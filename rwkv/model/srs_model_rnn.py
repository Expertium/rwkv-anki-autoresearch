import math
import os

import numpy as np
from rwkv.architecture import AnkiRWKVConfig
from rwkv.data_processing import (
    CARD_FEATURE_COLUMNS,
)
from rwkv.model.rwkv_rnn_model import RWKV7RNN
from rwkv.model.srs_model import is_excluded
import torch

# An RNN implementation of srs_model.

# Deploy-side counterfactual button probes (2026-07-26). Same construction as the training
# probes (rwkv/prepare_batch.py insert_probes) and the rectified eval path: the current
# review with the grade one-hot swapped and the current row's duration zeroed. Keeping the
# constants here rather than importing prepare_batch keeps the deploy path free of the
# LMDB/data-pipeline import chain.
_COL_DUR = CARD_FEATURE_COLUMNS.index("scaled_duration")
_COL_R1 = CARD_FEATURE_COLUMNS.index("rating_1")
assert [CARD_FEATURE_COLUMNS[_COL_R1 + k] for k in range(4)] == [
    "rating_1", "rating_2", "rating_3", "rating_4"
], "grade one-hot columns not contiguous"
# Duration encoding for the four counterfactual button probes. DEFAULT 0.0 = the pipeline's own
# "no press yet" encoding, the value every query row already carries (verified on the reference
# traces). ⚠ The default used to be -0.12079481388911952 = scale_duration(6433), the retired
# train-set median, while every actual run overrode it with RWKV_PROBE_DUR=0.0 -- so a caller who
# just imported this module got a DIFFERENT model than training and eval scored. That cost 15-20%
# on the button intervals and is exactly the divergence CLAUDE.md sec 9 case 2 records as already
# unified on 0.0. The env var remains, for experiments only.
_PROBE_DUR_SCALED = float(os.environ.get("RWKV_PROBE_DUR", "0.0"))


def __nop(ob):
    return ob


ModuleType = torch.nn.Module
FunctionType = __nop

# ModuleType = torch.jit.ScriptModule
# FunctionType = torch.jit.script_method


class SrsRWKVRnn(ModuleType):
    def __init__(self, anki_rwkv_config: AnkiRWKVConfig):
        super().__init__()
        self.card_features_dim = 92
        # RWKV_ZERO_FEATURES: same input-feature mask as SrsRWKV (srs_model.py, iter 15) so
        # the RNN/deploy path matches a model trained with dropped features. persistent=False:
        # not in state_dict, old checkpoints load unchanged.
        _zero_feats = [
            int(t) for t in os.environ.get("RWKV_ZERO_FEATURES", "").split(",") if t.strip()
        ]
        assert all(0 <= i < 92 for i in _zero_feats), f"RWKV_ZERO_FEATURES out of range: {_zero_feats}"
        self.input_feat_mask_on = len(_zero_feats) > 0
        _mask = torch.ones(92)
        for _i in _zero_feats:
            _mask[_i] = 0.0
        # Plain attribute (not a buffer), matching srs_model.py: keeps state_dict unchanged.
        self.input_feat_mask = _mask
        if self.input_feat_mask_on:
            print(f"[feat-mask] (rnn) zeroing input feature dims {_zero_feats}")
        # RWKV_MONO_CURVES: same ahead-residual cummin projection as SrsRWKV (srs_model.py,
        # iter 22) so the RNN/deploy path matches a model trained with monotone curves.
        self.mono_curve_on = os.environ.get("RWKV_MONO_CURVES", "0") == "1"
        # RWKV_NO_AHEAD_RESIDUAL: same disable as SrsRWKV (srs_model.py, Andrew 2026-07-16)
        # so the RNN/deploy path matches a model trained without the piecewise correction.
        self.no_ahead_residual = os.environ.get("RWKV_NO_AHEAD_RESIDUAL", "0") == "1"
        # RWKV_GRU_HEAD=N: same GRU-faithful curve head as SrsRWKV (srs_model.py, track-2
        # A3) so the RNN/deploy path matches. Legacy w_linear/ahead head become 1x1 dummies
        # (state_dict must stay SYMMETRIC with the training model for copy_downcast_).
        self.gru_n = int(os.environ.get("RWKV_GRU_HEAD", "0"))
        self.gru_on = self.gru_n > 0
        if self.gru_on:
            self.no_ahead_residual = True
            print(f"[gru] (rnn) GRU curve head ON: N={self.gru_n}")
        self.d_model = anki_rwkv_config.d_model
        self.features_fc_dim = anki_rwkv_config.features_fc_mult * anki_rwkv_config.d_model
        self.ahead_head_dim = anki_rwkv_config.head_fc_mult * self.d_model
        self.p_head_dim = anki_rwkv_config.head_fc_mult * self.d_model
        self.w_head_dim = anki_rwkv_config.head_fc_mult * self.d_model
        self.num_curves = anki_rwkv_config.num_curves
        if self.gru_on:
            self.num_curves = self.gru_n

        self.features2card = torch.nn.Sequential(
            torch.nn.Linear(self.card_features_dim, self.features_fc_dim),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(self.features_fc_dim),
            torch.nn.Linear(self.features_fc_dim, self.d_model),
            torch.nn.SiLU(),
        )
        # stamp each stream's name onto its config, exactly as SrsRWKV.__init__ does --
        # RWKV_STRIP_CMIX matches on "<stream_name>:<layer_id>", so without this the deploy
        # path would silently strip NOTHING and quietly diverge from the trained model
        for _name, _cfg in anki_rwkv_config.modules:
            _cfg.stream_name = _name
        self.rwkv_modules = torch.nn.ModuleList(
            [RWKV7RNN(config=config) for _, config in anki_rwkv_config.modules]
        )
        # iter 41 (RWKV_INTERLEAVE): deploy mirror of srs_model.py's round-robin layer
        # schedule -- same flag, same order, or the deploy path silently computes a
        # different model (the exact failure class the three-way-parity rule exists for).
        self.interleave_on = os.environ.get("RWKV_INTERLEAVE", "0") == "1"
        self.stream_depths = [config.n_layers for _, config in anki_rwkv_config.modules]
        if self.interleave_on:
            print(f"[interleave] (rnn) round-robin layer schedule ON: depths={self.stream_depths}")
        self.prehead_norm = torch.nn.LayerNorm(self.d_model)
        self.prehead_dropout = torch.nn.Dropout(p=anki_rwkv_config.dropout)
        if self.gru_on:
            self.head_ahead_logits = torch.nn.Sequential(
                torch.nn.Linear(1, 1),
                torch.nn.ReLU(),
            )
        else:
            self.head_ahead_logits = torch.nn.Sequential(
                torch.nn.Linear(self.d_model, self.ahead_head_dim),
                torch.nn.ReLU(),
            )
        self.head_w = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, 1 * self.d_model),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(1 * self.d_model),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(1 * self.d_model, self.w_head_dim),
        )
        self.head_p = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, self.p_head_dim),
            torch.nn.ReLU(),
        )

        self.max_e = 21
        self.point_spread = 18.5
        self.num_points = anki_rwkv_config.num_points
        if self.gru_on:
            self.ahead_linear = torch.nn.Linear(1, 1)
            self.w_linear = torch.nn.Linear(1, 1)
            _N = self.gru_n
            self.gru_w_weight = torch.nn.Parameter(torch.zeros(_N, self.w_head_dim))
            self.gru_w_bias = torch.nn.Parameter(torch.zeros(_N))
            self.gru_s_weight = torch.nn.Parameter(torch.zeros(_N, self.w_head_dim))
            self.gru_s_bias = torch.nn.Parameter(torch.zeros(_N))
            self.gru_d_weight = torch.nn.Parameter(torch.zeros(_N, self.w_head_dim))
            self.gru_d_bias = torch.nn.Parameter(torch.zeros(_N))
        else:
            self.ahead_linear = torch.nn.Linear(self.ahead_head_dim, self.num_points)

            self.w_linear = torch.nn.Linear(self.w_head_dim, self.num_curves)

        self.s_point_spread = 18.5
        self.s_max = 22

        self.p_linear = torch.nn.Linear(self.p_head_dim, 4)

        # RWKV_PAVA_LAMBDA: the rectifier's 3 junction thetas are MODEL parameters, not a
        # loss detail (Andrew 2026-07-26). They must exist here or load_state_dict (strict)
        # rejects any checkpoint trained with PAVA -- the deploy path could not even open
        # the champion. p_j = 2*tanh(theta_j); init p = 1 = classic PAVA.
        self.pava_lambda = float(os.environ.get("RWKV_PAVA_LAMBDA", "0"))
        if self.pava_lambda != 0.0:
            from rwkv.model.pava import theta_init
            self.pava_theta = torch.nn.Parameter(theta_init())
            print(f"[pava] (rnn) rectifier params present, lambda={self.pava_lambda}")

    def forgetting_curve(self, w, label_elapsed_seconds):
        s_space_raw = torch.exp(
            torch.linspace(0, self.s_point_spread, self.num_curves, device=w.device)
        )
        s_space = 0.1 + (s_space_raw - 1) * (np.e ** (self.s_max - self.s_point_spread))
        label_elapsed_seconds = torch.max(torch.tensor(1.0), label_elapsed_seconds)
        return 1e-5 + (1 - 2 * 1e-5) * torch.sum(
            w * 0.9 ** (label_elapsed_seconds / s_space), dim=-1
        )

    def gru_forgetting_curve(self, w, s_raw, d_raw, label_elapsed_seconds):
        # mirror of SrsRWKV.gru_forgetting_curve (srs_model.py) -- keep in sync
        s = torch.exp(torch.clamp(s_raw, min=-25.0, max=25.0))
        d = torch.exp(torch.clamp(d_raw, min=-25.0, max=25.0))
        t = torch.max(torch.tensor(1.0), label_elapsed_seconds)
        r = torch.sum(w * torch.exp(-d * torch.log1p(t / (1e-7 + s))), dim=-1)
        return 1e-5 + (1 - 2 * 1e-5) * r

    def interp(self, out_ahead_logits, label_elapsed_seconds):
        label_elapsed_seconds = torch.clamp(label_elapsed_seconds.contiguous(), min=1)
        point_space_raw = torch.exp(
            torch.linspace(
                0, self.point_spread, self.num_points, device=out_ahead_logits.device
            )
        )
        point_space = 0.5 + (point_space_raw - 1) * (
            np.e ** (self.max_e - self.point_spread)
        )
        right_idx = torch.searchsorted(point_space, label_elapsed_seconds)
        left_idx = torch.clamp(right_idx - 1, min=0)
        xl, xr = point_space[left_idx], point_space[right_idx]
        yl = torch.gather(out_ahead_logits, dim=-1, index=left_idx)
        yr = torch.gather(out_ahead_logits, dim=-1, index=right_idx)
        res = 1e-5 + (1 - 2 * 1e-5) * (
            yl + (yr - yl) * (label_elapsed_seconds - xl) / (xr - xl)
        )
        return res.squeeze(-1)

    def ahead_residual(self, out_ahead_logits, t):
        """The piecewise-linear correction term, or its exact constant when disabled.

        With RWKV_NO_AHEAD_RESIDUAL the logits are all zeros, and interp then returns
        1e-5 + (1 - 2e-5) * 0 = 1e-5 for EVERY t -- bit-identical to the short-circuit
        below. Taking the short-circuit also keeps the interval solver away from interp's
        searchsorted, whose right index runs off the end of point_space past ~1.3e9 s
        (the solver probes out to e^22 s).
        """
        if self.no_ahead_residual:
            return torch.full(
                (t.shape[0],), 1e-5, dtype=torch.float32, device=out_ahead_logits.device
            )
        return self.interp(out_ahead_logits.expand(t.shape[0], -1).contiguous(), t)

    def curve_p(self, out_ahead_logits, out_w, out_s_raw, out_d_raw, t):
        """Recall probability at each elapsed time in t (T, 1) -> (T,).

        The single curve formula for this file: run() and the deploy button API both go
        through it, so the two can never drift apart.
        """
        if self.gru_on:
            curve_probs_raw = self.gru_forgetting_curve(out_w, out_s_raw, out_d_raw, t)
        else:
            curve_probs_raw = self.forgetting_curve(out_w, t)
        curve_logits_raw = torch.log(curve_probs_raw / (1 - curve_probs_raw))
        return torch.sigmoid(curve_logits_raw + self.ahead_residual(out_ahead_logits, t))

    @torch.inference_mode()
    def button_heads(
        self,
        card_features,
        card_state,
        note_state,
        deck_state,
        preset_state,
        global_state,
    ):
        """Head outputs for the 4 counterfactual button presses of the CURRENT review.

        THE STATES ARE NOT ADVANCED. These are the deploy-side twins of the training probe
        rows, which are skip rows -- the WKV kernel restores the pre-step state on a skip
        (rwkv7_cuda.cu: `if (skip) state_xy = in_state_xy`), so asking for the four
        predictions perturbs nothing. Feed the real answer through review() afterwards, with
        its REAL duration, to advance the state.

        card_features: (92,) or (1, 92) -- the current review's row. The grade one-hot is
        overwritten per button and scaled_duration is zeroed here (RWKV_PROBE_DUR), so the
        caller cannot get the contract wrong: the four curves differ ONLY by the button.
        Duration is not yet observable at prediction time, and holding it fixed across the
        four is what makes the ordering comparison meaningful (Andrew 2026-07-26: "assume
        if the user presses Good and spent 7 seconds on reviewing, he would spend 7 seconds
        either way").

        Returns (ahead_logits, w, s_raw, d_raw, p_logits), each stacked over the 4 buttons
        in slot order Again, Hard, Good, Easy. The heads do not depend on the horizon t, so
        the curve can then be evaluated at any number of times for free.
        """
        if card_features.dim() == 1:
            card_features = card_features.unsqueeze(0)
        assert card_features.shape == (1, self.card_features_dim), card_features.shape

        base = card_features.clone()
        base[:, _COL_DUR] = _PROBE_DUR_SCALED
        base[:, _COL_R1:_COL_R1 + 4] = 0

        outs = []
        for k in range(4):
            row = base.clone()
            row[:, _COL_R1 + k] = 1
            # every button reads the SAME incoming state; next_* deliberately dropped
            out_ahead_logits, out_w, out_s_raw, out_d_raw, out_p_logits = self.review(
                row, card_state, note_state, deck_state, preset_state, global_state
            )[:5]
            outs.append((out_ahead_logits, out_w, out_s_raw, out_d_raw, out_p_logits))
        return tuple(torch.cat([o[j] for o in outs], dim=0) for j in range(5))

    def button_curves(self, heads, elapsed_seconds):
        """PAVA-rectified recall curves: (4, T) for T horizons, ordered Again<=..<=Easy.

        The rectifier is part of the MODEL, not the loss (Andrew 2026-07-26) -- training,
        eval and this path all apply it. Uniform pooling weights: iter 24 measured p-head
        weighting as null, so deploy keeps the simpler rectifier.
        """
        from rwkv.model.pava import pava_rectify

        ahead_logits, w, s_raw, d_raw, _ = heads
        t = torch.as_tensor(
            elapsed_seconds, dtype=torch.float32, device=w.device
        ).reshape(-1, 1)
        # (4, T): each button's curve over all horizons
        raw = torch.stack(
            [
                self.curve_p(ahead_logits[k:k + 1], w[k:k + 1], s_raw[k:k + 1],
                             d_raw[k:k + 1], t)
                for k in range(4)
            ]
        )
        powers = (
            2.0 * torch.tanh(self.pava_theta)
            if hasattr(self, "pava_theta")
            else torch.ones(3, device=w.device, dtype=torch.float32)
        )
        v = raw.t().float().contiguous()  # (T, 4) -- pava_rectify pools across buttons
        rect = pava_rectify(v, torch.ones_like(v), powers)
        return rect.t()

    def button_intervals(self, heads, desired_retention=0.9, max_seconds=None, iters=50):
        """Next-review interval per button (seconds), from the RECTIFIED curves.

        Geometric bisection on each button's rectified curve for R(t) = desired_retention.
        Cheap: the heads are already computed, so each step is closed-form arithmetic, and
        rectification is applied at every probe so the coupling between buttons is never
        skipped. Ordered by construction -- the curves are ordered at every t and each is
        decreasing in t -- and asserted, since a violated order is exactly the deploy bug
        the rectifier exists to prevent.
        """
        dev = heads[1].device
        hi_default = math.exp(self.s_max)
        lo = torch.ones(4, dtype=torch.float32, device=dev)
        hi = torch.full(
            (4,), hi_default if max_seconds is None else float(max_seconds),
            dtype=torch.float32, device=dev,
        )
        for _ in range(iters):
            mid = torch.sqrt(lo * hi)
            # button k evaluated at ITS OWN midpoint -> the diagonal of the (4 t, 4 button)
            # rectified matrix
            r = self.button_curves(heads, mid).diagonal()
            keep_going = r > desired_retention  # still above target -> interval can grow
            lo = torch.where(keep_going, mid, lo)
            hi = torch.where(keep_going, hi, mid)
        out = torch.sqrt(lo * hi)
        assert bool((out[1:] >= out[:-1] * (1 - 1e-4)).all()), (
            f"button intervals not ordered: {out.tolist()}"
        )
        return out

    def review(
        self,
        card_features,
        card_state,
        note_state,
        deck_state,
        preset_state,
        global_state,
    ):
        assert len(card_features.shape) == 2

        if self.input_feat_mask_on:
            card_features = card_features * self.input_feat_mask.to(
                card_features.device, card_features.dtype
            )
        card_rwkv_input = self.features2card(card_features)
        if self.interleave_on:
            import copy as _copy
            in_states = [card_state, deck_state, note_state, preset_state, global_state]
            states = []
            for _i, _st in enumerate(in_states):
                states.append(
                    _copy.deepcopy(_st) if _st is not None
                    else self.rwkv_modules[_i].init_state()
                )
            x_il = card_rwkv_input
            v0s = [torch.empty(0) for _ in range(len(states))]
            for _r in range(max(self.stream_depths)):
                for _i in range(len(states)):
                    if _r < self.stream_depths[_i]:
                        v0_in = torch.empty_like(x_il) if _r == 0 else v0s[_i]
                        x_il, v0s[_i] = self.rwkv_modules[_i].forward_layer(
                            _r, x_il, v0_in, states[_i]
                        )
            global_encoding = x_il
            (next_card_state, next_deck_state, next_note_state,
             next_preset_state, next_global_state) = states
        else:
            card_encoding, next_card_state = self.rwkv_modules[0].run(
                card_rwkv_input, card_state
            )
            deck_encoding, next_deck_state = self.rwkv_modules[1].run(
                card_encoding, deck_state
            )
            note_encoding, next_note_state = self.rwkv_modules[2].run(
                deck_encoding, note_state
            )
            preset_encoding, next_preset_state = self.rwkv_modules[3].run(
                note_encoding, preset_state
            )
            global_encoding, next_global_state = self.rwkv_modules[4].run(
                preset_encoding, global_state
            )

        x = self.prehead_dropout(self.prehead_norm(global_encoding))
        x_w = self.head_w(x).float()
        if self.gru_on:
            out_w_logits = torch.nn.functional.linear(x_w, self.gru_w_weight, self.gru_w_bias)
            out_s_raw = torch.nn.functional.linear(x_w, self.gru_s_weight, self.gru_s_bias)
            out_d_raw = torch.nn.functional.linear(x_w, self.gru_d_weight, self.gru_d_bias)
        else:
            out_w_logits = self.w_linear(x_w)
            out_s_raw = torch.zeros(1, dtype=torch.float32, device=x.device)
            out_d_raw = torch.zeros(1, dtype=torch.float32, device=x.device)
        out_w = torch.nn.functional.softmax(out_w_logits, dim=-1)
        if self.no_ahead_residual:
            out_ahead_logits = torch.zeros(
                x.shape[:-1] + (self.num_points,), dtype=torch.float32, device=x.device
            )
        else:
            out_ahead_logits = self.ahead_linear(self.head_ahead_logits(x).float())
            if self.mono_curve_on:
                out_ahead_logits, _ = torch.cummin(out_ahead_logits, dim=-1)

        x_p = self.head_p(x).float()
        out_p_logits = self.p_linear(x_p)
        return (
            out_ahead_logits,
            out_w,
            out_s_raw,
            out_d_raw,
            out_p_logits,
            next_card_state,
            next_note_state,
            next_deck_state,
            next_preset_state,
            next_global_state,
        )

    @torch.inference_mode()
    def run(self, df, dtype, device):
        print(
            "TODO: properly do id encode and time encode, it is just padded right now"
        )

        df = df.reset_index(drop=True)
        card_states = {}
        note_states = {}
        deck_states = {}
        preset_states = {}
        global_state = None
        ahead_ps = {}
        imm_ps = {}
        card_features_df = df[CARD_FEATURE_COLUMNS]
        card_features_all = torch.tensor(
            card_features_df.to_numpy(), dtype=dtype, device=device, requires_grad=False
        ).unsqueeze(0)
        label_elapsed_seconds_all = (
            torch.tensor(df["label_elapsed_seconds"], dtype=dtype, device=device)
            .to(torch.float32)
            .unsqueeze(0)
        )

        with torch.inference_mode():
            for i, row in df.iterrows():
                if i % 100 == 0:
                    print(i)
                card_id = row["card_id"]
                note_id = row["note_id"]
                deck_id = row["deck_id"]
                preset_id = row["preset_id"]

                if card_id not in card_states:
                    card_states[card_id] = None
                if note_id not in note_states:
                    note_states[note_id] = None
                if deck_id not in deck_states:
                    deck_states[deck_id] = None
                if preset_id not in preset_states:
                    preset_states[preset_id] = None

                card_features = card_features_all[:, i]
                (
                    out_ahead_logits,
                    out_w,
                    out_s_raw,
                    out_d_raw,
                    out_p_logits,
                    next_card_state,
                    next_note_state,
                    next_deck_state,
                    next_preset_state,
                    next_global_state,
                ) = self.review(
                    card_features,
                    card_states[card_id],
                    note_states[note_id],
                    deck_states[deck_id],
                    preset_states[preset_id],
                    global_state,
                )

                if not row["skip"]:
                    card_states[card_id] = next_card_state
                    note_states[note_id] = next_note_state
                    deck_states[deck_id] = next_deck_state
                    preset_states[preset_id] = next_preset_state
                    global_state = next_global_state

                curve_prob = self.curve_p(
                    out_ahead_logits,
                    out_w,
                    out_s_raw,
                    out_d_raw,
                    label_elapsed_seconds_all[:, i].unsqueeze(0),
                )

                if row["has_label"]:
                    if row["is_query"]:
                        out_p_probs = torch.softmax(out_p_logits, dim=-1)
                        out_p_again, out_p_1, out_p_2, out_p_3 = out_p_probs.unbind(
                            dim=-1
                        )
                        out_p_binary = torch.clamp(
                            1.0 - out_p_again, min=1e-5, max=1.0 - 1e-5
                        )
                        imm_ps[row["label_review_th"]] = out_p_binary.item()
                    else:
                        ahead_ps[row["label_review_th"]] = curve_prob.item()

        return ahead_ps, imm_ps

    def copy_downcast_(self, master_model, dtype):
        master_params = dict(master_model.named_parameters())
        with torch.no_grad():
            for name, param in self.named_parameters():
                target_dtype = torch.float32 if is_excluded(name) else dtype
                assert param.dtype == target_dtype
                param.data.copy_(master_params[name].to(target_dtype))
                assert param.dtype == target_dtype

    def selective_cast(self, dtype):
        for name, module in self.named_modules():
            if len(name) == 0:
                # Skip the root module
                continue
            if not is_excluded(name):
                if dtype == torch.bfloat16:
                    module = module.to(dtype)
                elif dtype == torch.half:
                    raise ValueError("not tested.")
                elif dtype == torch.float32:
                    pass
        return self
