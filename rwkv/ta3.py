"""TA3 -- Training-Time Testing (Dang 2026, https://dangxingyu.github.io/blog/t3.html).

Per optimizer step, split the batch into halves A and Q (here: by position on the batch's
time axis, through the `has_label` field, so the split needs no data-stream change; each loss
group is cut at the midpoint of its OWN labelled rows -- see `_mode_cuts` -- because a single
raw time cut can leave a phase with no rows of one group, which aborts the step):

  1. L_A at the CURRENT weights, backward -> g_A.
  2. Take a NORMALISED trial step DOWNHILL on A: u = unit(g_A) over a chosen subset of
     matrices; master <- master - eta*u.
  3. L_Q at the PERTURBED weights, backward -> g_Q (first-order MAML: no gradient through
     the shift).
  4. Restore the master weights exactly (no add/sub round trip), then let the ordinary
     Muon/AdamW pipeline step on the row-count-weighted average
         g = (n_A*g_A + n_Q*g_Q) / (n_A + n_Q)
     which is the gradient of the full-batch row mean, so splitting the batch does not
     change the objective's normalisation. (Each loss group carries its own phase/full
     row-count ratio; a group with a single labelled row cannot be split and lands whole in
     phase A at weight 1, which is exactly its full-batch gradient.)

Everything is env-gated and inert by default, so an unset run is byte-identical:

  RWKV_TA3=1                 enable the training hook (default off: no path entered)
  RWKV_TA3_ETA=0.3           trial-step length (global L2 norm of the displacement)
  RWKV_TA3_SCOPE=cmix        which matrices the trial step moves: cmix | trunk | all
  RWKV_TA3_FRAC=0.5          where the A/Q cut sits on the time axis
  RWKV_TA3_PROBE=<n>         measure-only mode: probe n steps, print JSON, exit 0
  RWKV_TA3_PROBE_ETAS=...    eta list for the probe; zero/random/uphill controls are added

The probe is a plumbing/mechanism check on the REAL training data path. It never steps the
optimizer and it is not accuracy evidence.
"""
import dataclasses
import json
import math
import os
import sys

import torch

from rwkv import sam as _sam  # save_rng/restore_rng only; SAM itself stays off here

# global_labels column layout (data_processing.create_sample, also clock_fix.py):
#   les, led, label_y, label_rating, has_label, label_is_equalize, is_query
_HAS_LABEL = 4
_IS_QUERY = 6



def _env(name, default=""):
    return (os.environ.get(name, "") or default).strip()


ETA = float(_env("RWKV_TA3_ETA", "0.3") or 0.3)
FRAC = float(_env("RWKV_TA3_FRAC", "0.5") or 0.5)
SCOPE = _env("RWKV_TA3_SCOPE", "cmix") or "cmix"
PROBE_STEPS = int(_env("RWKV_TA3_PROBE", "0") or 0)
PROBE_ETAS = tuple(
    float(x) for x in _env("RWKV_TA3_PROBE_ETAS", "0.03,0.1,0.3,1.0").split(",") if x.strip()
)
ON = _env("RWKV_TA3", "0") == "1"
PROBE_ON = PROBE_STEPS > 0
_probe_seen = [0]  # how many steps the measure-only probe has already reported

if ON or PROBE_ON:
    print(f"[ta3] Training-Time Testing ON: eta={ETA} scope={SCOPE} frac={FRAC} "
          f"probe_steps={PROBE_STEPS}", flush=True)
if PROBE_ON and ON:
    raise RuntimeError("RWKV_TA3_PROBE is measure-only; do not combine it with RWKV_TA3=1")


# --------------------------------------------------------------------------- scope


def scope_names(model):
    """Names of the trainable WEIGHT MATRICES the trial step is allowed to move.

    Only matrices, never norms/biases/interpolation parameters: the blog's trial step moves the
    MLP matrices, and silently moving every parameter under `.channel_mixer.` (norms, biases,
    the 3-D lerp) would be a different intervention. The selected list is logged once so the
    experiment description matches what ran; unknown scope names are rejected, not defaulted.
    """
    if SCOPE not in ("cmix", "trunk", "all"):
        raise RuntimeError(f"RWKV_TA3_SCOPE={SCOPE!r} is not one of cmix|trunk|all")
    names = []
    for name, param in model.named_parameters():
        if not param.requires_grad or param.dim() < 2:
            continue
        if SCOPE == "all":
            names.append(name)
        elif SCOPE == "trunk":
            if name.startswith("rwkv_modules"):
                names.append(name)
        elif ".channel_mixer." in name and name.endswith(".weight"):
            names.append(name)
    if not names:
        raise RuntimeError(f"TA3 scope {SCOPE!r} selected no parameters")
    return names


def scope_report(model):
    names = scope_names(model)
    params = dict(model.named_parameters())
    total = sum(params[n].numel() for n in names)
    print(f"[ta3] scope={SCOPE}: {len(names)} weight matrices, {total} parameters", flush=True)
    for name in names:
        print(f"[ta3]   {name} {tuple(params[name].shape)}", flush=True)
    return names


# --------------------------------------------------------------------------- A/Q split


def _rank_split(mask, frac, batch_dim):
    """Boolean "belongs to phase A" for the True rows of `mask`, split by RANK along the time axis.

    The first floor(frac*n) rows in (time, batch) order go to phase A and the rest to phase Q; for
    n >= 2 at least one row stays on each side. Ranking, not a time threshold: several batch rows can
    share one time position (B > 1), and a threshold then keeps or drops whole tied groups together
    -- the first revision of this fix did exactly that and could still leave a phase empty (caught by
    `scratchpad/ta3/test_split.py`).
    """
    idx = mask.nonzero(as_tuple=False)  # (n, 2): [batch row, time position]
    out = torch.zeros_like(mask)
    n = int(idx.shape[0])
    if n == 0:
        return out
    if n < 2:
        out[idx[0, 0], idx[0, 1]] = True  # a group that cannot be split lands whole in phase A
        return out
    n_a = max(1, min(n - 1, int(math.floor(frac * n))))
    key = idx[:, 1] * batch_dim + idx[:, 0]  # sort by time, batch row breaks ties
    chosen = idx[torch.argsort(key)[:n_a]]
    out[chosen[:, 0], chosen[:, 1]] = True
    return out


def _mode_split(labels, frac):
    """Phase-A mask for the batch: every loss group is split at the midpoint OF ITS OWN rows.

    A single cut on the raw time axis can leave a phase with no rows of one group, which aborts the
    step. That happened on 2026-11-15 at step 4930 of the first chain epoch (`RuntimeError: TA3:
    mode 'ahead' has no labelled rows in this phase`) and cost ~12 h of idle GPU. Each group is split
    by rank over its own labelled rows, which keeps the intended reading -- earlier rows of a group
    train the trial direction, later rows test it -- and always leaves both phases a row when the
    group has two or more.
    """
    has = labels[..., _HAS_LABEL] > 0.5
    isq = labels[..., _IS_QUERY] > 0.5
    batch_dim = int(labels.shape[0])
    in_a = torch.zeros_like(has)
    for mask in (has & ~isq, has & isq):
        in_a = in_a | _rank_split(mask, frac, batch_dim)
    return in_a


def split_batch(batch, frac=None):
    """Return (batch_A, batch_Q): the same batch with `has_label` kept on one side of the split.

    The split is a rank rule on the batch's time axis (one per loss group, see `_mode_split`), so the
    fetch workers, the group order and every other batch field stay untouched; only the loss mask
    moves. Probe rows only exist when `RWKV_PROBE_DENSITY > 0` (no current arm sets it); they are
    ranked among themselves so their terms are not double-counted.
    """
    if frac is None:
        frac = FRAC
    labels = batch.labels
    T = int(labels.shape[1])
    if T < 2:
        raise RuntimeError("TA3 needs at least 2 time steps per batch")
    has = labels[..., _HAS_LABEL] > 0.5
    in_a = _mode_split(labels, frac)
    probe_a = None
    if batch.probe_rows is not None and batch.probe_rows.numel() > 0:
        probe_t = batch.probe_target % T
        order = torch.argsort(probe_t)
        n = int(probe_t.numel())
        n_a = n if n < 2 else max(1, min(n - 1, int(math.floor(frac * n))))
        probe_a = torch.zeros_like(probe_t, dtype=torch.bool)
        probe_a[order[:n_a]] = True
    out = []
    for first_half in (True, False):
        keep = in_a if first_half else (has & ~in_a)
        lab = labels.clone()
        lab[..., _HAS_LABEL] = lab[..., _HAS_LABEL] * keep.to(lab.dtype)
        extra = {}
        if probe_a is not None:
            sel = probe_a if first_half else ~probe_a
            extra = {
                "probe_rows": batch.probe_rows[sel],
                "probe_target": batch.probe_target[sel],
                "probe_pressed": batch.probe_pressed[sel],
                "probe_query": batch.probe_query[sel],
            }
        out.append(dataclasses.replace(batch, labels=lab, **extra))
    return out[0], out[1]


# --------------------------------------------------------------------------- master weights


_reported = [False]


def _maybe_report(master_model):
    if not _reported[0]:
        _reported[0] = True
        scope_report(master_model)


def _master_map(master_model):
    return dict(master_model.named_parameters())


def _unit_direction(master_model, names):
    """u = unit(g) with ONE global L2 norm over the scoped matrices (the blog's u_A)."""
    params = _master_map(master_model)
    total = 0.0
    for name in names:
        grad = params[name].grad
        if grad is None:
            raise RuntimeError(f"TA3: no gradient for scoped parameter {name}")
        total += float(grad.detach().float().pow(2).sum().item())
    norm = math.sqrt(total)
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(f"TA3: phase-A gradient norm is {norm!r}")
    return [params[name].grad.detach() / norm for name in names], norm


def _shift(master_model, names, dirs, scale):
    """Move the master weights DOWNHILL by `scale` along `dirs` (p -= scale*d).

    Convention, used by both the training hook and the probe: a POSITIVE scale is a
    descent step. An earlier probe revision passed the sign in a second place and its
    "downhill" row was really the ascent, which inverts the whole reading of the probe.
    """
    params = _master_map(master_model)
    with torch.no_grad():
        for name, direction in zip(names, dirs):
            params[name].data.add_(direction, alpha=-scale)


def _snapshot(master_model, names):
    params = _master_map(master_model)
    return [params[name].detach().clone() for name in names]


def _restore(master_model, names, saved):
    params = _master_map(master_model)
    with torch.no_grad():
        for name, value in zip(names, saved):
            params[name].data.copy_(value)  # copy, never add/sub: exact restore


def _clear_master_grads(master_model):
    with torch.no_grad():
        for param in master_model.parameters():
            param.grad = None


# --------------------------------------------------------------------------- training hook


def _mode_counts(batch):
    """Count the two loss groups' rows in a batch: ahead rows (has_label & ~is_query) and
    immediate/query rows (has_label & is_query). These are exactly the denominators the model's
    loss uses, provided no re-weighting lever is active (guarded in `_assert_no_reweighting`)."""
    labels = batch.labels
    has = labels[..., _HAS_LABEL] > 0.5
    isq = labels[..., _IS_QUERY] > 0.5
    return float((has & ~isq).sum().item()), float((has & isq).sum().item())


def _assert_no_reweighting(model, batch):
    """TA3's exact merge needs the loss masks to be plain 0/1 row counts."""
    problems = []
    if getattr(model, "equalize_loss_on", False):
        problems.append("RWKV_EQUALIZE_LOSS_W")
    if getattr(model, "ahead_probe_only", False):
        problems.append("RWKV_AHEAD_PROBE_ONLY")
    if float(getattr(model, "pbin_scale", 0.0) or 0.0) != 0.0:
        problems.append("pbin_scale")
    if float(getattr(model, "ord_lambda", 0.0) or 0.0) != 0.0:
        problems.append("RWKV_ORD_LAMBDA")
    if float(getattr(model, "rank1_reg_lambda", 0.0) or 0.0) != 0.0:
        problems.append("RWKV_QAT_RANK1_REG")
    if float(getattr(model, "tail_reg_lambda", 0.0) or 0.0) != 0.0:
        problems.append("RWKV_QAT_TAIL_PENALTY")
    if float(getattr(model, "pava_lambda", 0.0) or 0.0) != 0.0:
        problems.append("RWKV_PAVA")
    if batch.user_weight is not None:
        weights = batch.user_weight.float()
        if float((weights - 1.0).abs().max().item()) > 1e-6:
            problems.append("RWKV_USER_WEIGHT")
    if problems:
        raise RuntimeError(
            "TA3 requires the plain row-count objective; these levers change the loss masks or "
            "denominators and would break the exact merge: " + ", ".join(problems)
        )


def _mask_mode(batch, mode):
    """Copy of the batch with only ONE loss group's rows labelled.

    The model reports its per-term averages DETACHED, so the two groups' gradients cannot be
    separated after the fact. Masking the other group's rows out makes each pass's loss exactly
    one group over its own row count, which is what the per-group weights need. `skips` is left
    alone, so the forward is unchanged; only the loss mask moves.
    """
    labels = batch.labels.clone()
    has = labels[..., _HAS_LABEL]
    isq = labels[..., _IS_QUERY]
    if mode == "ahead":
        keep = (has > 0.5) & (isq < 0.5)
    elif mode == "imm":
        keep = (has > 0.5) & (isq > 0.5)
    else:
        raise ValueError(mode)
    labels[..., _HAS_LABEL] = torch.where(keep, has, torch.zeros_like(has))
    return dataclasses.replace(batch, labels=labels)


def _group_grad(model, batch, mode, kd_args, kd_mix_args, transfer, master_model):
    """One forward + one backward for one loss group, scaled by that group's phase/full ratio.

    The batch passed here is already mode-masked, so the model's own `average_loss` is exactly the
    group's loss over the group's own row count. The other group must have ZERO rows, otherwise the
    backward would carry its gradient too and the per-group weights would be wrong.
    """
    n_ahead, n_imm = _mode_counts(batch)
    other = n_imm if mode == "ahead" else n_ahead
    if other != 0:
        raise RuntimeError(f"TA3: mode-masked batch still has {other} {mode}-excluded rows labelled")
    if (n_ahead if mode == "ahead" else n_imm) == 0:
        raise RuntimeError(f"TA3: mode {mode!r} has no labelled rows in this phase")
    _clear_master_grads(master_model)
    stats = model.get_loss(batch, kd=kd_args, kd_mix=kd_mix_args)
    if stats is None:
        raise RuntimeError(f"TA3: {mode} loss is None")
    if not torch.isfinite(stats.average_loss):
        raise RuntimeError(f"TA3: non-finite {mode} loss")
    stats.average_loss.backward()
    transfer()
    grads = _grad_map(master_model)
    _clear_master_grads(master_model)
    return stats, grads


def _combine(grad_sets):
    """Weighted sum of (grad_map, weight) pairs; None means a zero gradient for that term."""
    out = {}
    for grads, weight in grad_sets:
        for name, grad in grads.items():
            term = grad * weight
            out[name] = term if name not in out else out[name] + term
    return out


def training_step(model, master_model, prepared_batch, kd_args, kd_mix_args,
                  transfer, copy_downcast, dtype, eta=None):
    """A complete TA3 optimizer step's gradient. Returns the A-phase statistics.

    g = (N_a^A/N_a)*g_A^ahead + (N_a^Q/N_a)*g_Q^ahead
      + (N_i^A/N_i)*g_A^imm   + (N_i^Q/N_i)*g_Q^imm
    where A is the trial step (at w) and Q is the test (at w - eta*unit(g_A)), N are the group row
    counts of the WHOLE batch and N^P those of phase P. At eta=0 this is the ordinary full-batch
    gradient exactly, which is what the gate tests.
    """
    if eta is None:
        eta = ETA
    _maybe_report(master_model)
    _assert_no_reweighting(model, prepared_batch)
    names = scope_names(master_model)
    all_names = [name for name, _ in master_model.named_parameters()]
    master = _master_map(master_model)
    batch_a, batch_q = split_batch(prepared_batch)

    n_a_full, n_i_full = _mode_counts(prepared_batch)
    a_cnt, i_cnt = _mode_counts(batch_a)
    q_cnt, j_cnt = _mode_counts(batch_q)
    if n_a_full + n_i_full <= 0:
        raise RuntimeError("TA3: the batch has no labelled rows")
    # Each group's rows must be covered by the two phases exactly. A group absent from a phase
    # carries weight 0 there, which is the normal case for a group with a single labelled row
    # (`_mode_cuts` cannot split it; its whole gradient then lands in phase A at weight 1).
    for label, full, cnt_a, cnt_q in (("ahead", n_a_full, a_cnt, q_cnt),
                                      ("imm", n_i_full, i_cnt, j_cnt)):
        if abs((cnt_a + cnt_q) - full) > 1e-9:
            _clear_master_grads(master_model)
            raise RuntimeError(f"TA3: phase counts for the {label} group do not cover the batch "
                               f"({cnt_a} + {cnt_q} != {full})")
    if a_cnt + i_cnt <= 0:
        raise RuntimeError("TA3: the batch has no labelled rows in phase A")

    def _phase(phase_batch, counts):
        """(unweighted gradient map, weighted merge terms, stats by group) for one phase.

        The unweighted map is the exact gradient of this phase's own objective (each group's
        masked pass already carries the model's own term scales), which is the direction the probe
        validated. The weighted terms are what the merge sums.
        """
        grad_map, terms, stats_by_mode = {}, [], {}
        for mode, cnt, full in counts:
            if cnt <= 0:
                continue
            group_stats, grads = _group_grad(model, _mask_mode(phase_batch, mode), mode,
                                            kd_args, kd_mix_args, transfer, master_model)
            stats_by_mode[mode] = group_stats
            grad_map = _combine(((grad_map, 1.0), (grads, 1.0)))
            terms.append((grads, cnt / full))
        return grad_map, terms, stats_by_mode

    g_a, terms, by_mode = _phase(batch_a, (("ahead", a_cnt, n_a_full), ("imm", i_cnt, n_i_full)))
    stats_a = by_mode.get("ahead")
    if stats_a is None:
        stats_a = by_mode["imm"]
    total = 0.0
    for name in names:
        grad = g_a.get(name)
        if grad is None:
            raise RuntimeError(f"TA3: no phase-A gradient for scoped parameter {name}")
        total += float(grad.double().pow(2).sum().item())
    norm_a = math.sqrt(total)
    if not math.isfinite(norm_a) or norm_a <= 0.0:
        raise RuntimeError(f"TA3: phase-A gradient norm is {norm_a!r}")
    dirs = [g_a[name] / norm_a for name in names]

    if q_cnt + j_cnt > 0:
        saved_w = _snapshot(master_model, names)
        failure = None
        try:
            _shift(master_model, names, dirs, eta)
            copy_downcast(master_model, dtype)
            _, terms_q, _ = _phase(batch_q, (("ahead", q_cnt, n_a_full), ("imm", j_cnt, n_i_full)))
            terms.extend(terms_q)
        except BaseException as exc:
            failure = exc
        finally:
            _restore(master_model, names, saved_w)
            copy_downcast(master_model, dtype)
        if failure is not None:
            _clear_master_grads(master_model)
            raise failure

    combined = _combine(terms)
    _clear_master_grads(master_model)
    for name in all_names:
        master[name].grad = combined.get(name)
    return stats_a


# --------------------------------------------------------------------------- probe


def _batch_loss(model, batch, kd_args, kd_mix_args):
    with torch.no_grad():
        stats = model.get_loss(batch, kd=kd_args, kd_mix=kd_mix_args)
    if stats is None:
        return None, 0.0
    return stats, float(stats.average_loss.item())


def probe(model, master_model, batch, step, kd_args, kd_mix_args, transfer,
          copy_downcast, dtype, out_path=None):
    """Measure whether a downhill A-step lowers Q's loss, against equal-norm controls.

    Reports, per step: the A and Q losses at the current weights, then Q's loss after
    zero / downhill / random / uphill trial steps of length eta, for a list of eta values,
    plus the g_A-g_Q inner product and the implied per-element displacement.
    """
    _maybe_report(master_model)
    batch_a, batch_q = split_batch(batch)
    names = scope_names(master_model)
    master = _master_map(master_model)

    # Pin the dropout masks so every phase and every control sees the SAME masks: the whole
    # probe is a paired comparison, and unpinned masks would put run-to-run dropout noise
    # (this repo has measured it) into the one quantity being measured.
    rng_state = _sam.save_rng()

    _clear_master_grads(master_model)
    stats_a = model.get_loss(batch_a, kd=kd_args, kd_mix=kd_mix_args)
    if stats_a is None or not torch.isfinite(stats_a.average_loss):
        raise RuntimeError("TA3 probe: non-finite A-phase loss")
    stats_a.average_loss.backward()
    transfer()
    dirs, norm_a = _unit_direction(master_model, names)
    g_a = {name: master[name].grad.detach().clone() for name, _ in master_model.named_parameters()
           if master[name].grad is not None}

    # g_Q at the CURRENT weights (alignment reference), then averaged with g_A in the master.
    _clear_master_grads(master_model)
    _sam.restore_rng(rng_state)
    stats_q = model.get_loss(batch_q, kd=kd_args, kd_mix=kd_mix_args)
    if stats_q is None or not torch.isfinite(stats_q.average_loss):
        raise RuntimeError("TA3 probe: non-finite Q-phase loss")
    stats_q.average_loss.backward()
    transfer()
    dot = 0.0
    gg_a = 0.0
    gg_q = 0.0
    dot_s = 0.0
    gg_a_s = 0.0
    gg_q_s = 0.0
    # All-parameter alignment: <g_A, g_Q> over every parameter with a gradient.
    for name in g_a:
        grad_q = master[name].grad
        if grad_q is None:
            continue
        a = g_a[name].float()
        q = grad_q.float()
        dot += float((a * q).sum().item())
        gg_a += float(a.pow(2).sum().item())
        gg_q += float(q.pow(2).sum().item())
    # Scope-restricted alignment: the trial step only moves SCOPED matrices, so the first-order
    # prediction for Q's loss is -eta * <unit(g_A[scope]), g_Q[scope]>. The all-parameter cosine
    # can look healthy while the scoped part is anti-aligned, which would make TA3 harmful.
    for name in names:
        a = g_a.get(name)
        if a is None:
            continue
        grad_q = master[name].grad
        if grad_q is None:
            continue
        a = a.float()
        q = grad_q.float()
        dot_s += float((a * q).sum().item())
        gg_a_s += float(a.pow(2).sum().item())
        gg_q_s += float(q.pow(2).sum().item())
    _clear_master_grads(master_model)

    n_a = float(stats_a.ahead_n + stats_a.imm_n)
    n_q = float(stats_q.ahead_n + stats_q.imm_n)
    loss_a = float(stats_a.average_loss.item())
    loss_q = float(stats_q.average_loss.item())
    # per-element displacement implied by a unit-norm trial step over the scoped matrices
    n_scope = sum(master[name].numel() for name in names)
    saved_w = _snapshot(master_model, names)
    rand_gen = torch.Generator(device="cpu").manual_seed(1234 + step)
    random_dirs = [torch.randn(p.shape, generator=rand_gen).to(p.device, p.dtype) for p in
                   (master[name] for name in names)]
    rand_norm = math.sqrt(sum(float(d.float().pow(2).sum().item()) for d in random_dirs))
    random_dirs = [d / rand_norm for d in random_dirs]

    rows = []
    try:
        for eta in PROBE_ETAS:
            # `_shift` is a DESCENT by scale, so "downhill" is +eta and "uphill" is -eta.
            for tag, scale_dirs, sign in (
                ("downhill", dirs, +1.0),
                ("uphill", dirs, -1.0),
                ("random", random_dirs, +1.0),
            ):
                _sam.restore_rng(rng_state)
                _shift(master_model, names, scale_dirs, sign * eta)
                copy_downcast(master_model, dtype)
                _, loss = _batch_loss(model, batch_q, kd_args, kd_mix_args)
                _restore(master_model, names, saved_w)
                copy_downcast(master_model, dtype)
                rows.append({"eta": eta, "control": tag, "loss_q": loss,
                             "delta_q": loss - loss_q})
    finally:
        _restore(master_model, names, saved_w)
        copy_downcast(master_model, dtype)
        _clear_master_grads(master_model)

    rec = {
        "step": step, "eta_default": ETA, "scope": SCOPE, "frac": FRAC,
        "n_scope_params": n_scope, "per_element_eta_default": ETA / math.sqrt(max(1, n_scope)),
        "loss_a": loss_a, "loss_q": loss_q, "n_a": n_a, "n_q": n_q,
        "grad_norm_a": norm_a,
        "grad_norm_q": math.sqrt(max(0.0, gg_q)),
        "cos_gA_gQ": (dot / math.sqrt(gg_a * gg_q)) if gg_a > 0 and gg_q > 0 else None,
        "cos_gA_gQ_scope": (dot_s / math.sqrt(gg_a_s * gg_q_s)) if gg_a_s > 0 and gg_q_s > 0 else None,
        "scope_align": dot_s,  # <g_A[scope], g_Q[scope]>; positive => descent on A helps Q
        "first_order_slope": (-dot_s / math.sqrt(max(1e-30, gg_a_s))),  # dL_Q/dscale at scale 0
        "trial_rows": rows,
    }
    print("[ta3-probe] " + json.dumps(rec), flush=True)
    if out_path:
        with open(out_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec) + "\n")
    _probe_seen[0] += 1
    return rec


def probe_should_stop(step=None):
    """The probe counts the steps IT measured, not the global step: a resumed run starts at a
    large global step, so comparing that counter with RWKV_TA3_PROBE would never fire."""
    return PROBE_ON and _probe_seen[0] >= PROBE_STEPS


# --------------------------------------------------------------------------- check mode


CHECK_STEPS = int(_env("RWKV_TA3_CHECK", "0") or 0)
CHECK_ON = CHECK_STEPS > 0
_check_seen = [0]  # check steps already reported (the global step is offset in a resumed run)


def check_should_stop():
    return CHECK_ON and _check_seen[0] >= CHECK_STEPS


def _grad_map(master_model):
    return {name: param.grad.detach().clone()
            for name, param in master_model.named_parameters() if param.grad is not None}


def _l2(grads, names=None):
    total = 0.0
    for name, grad in grads.items():
        if names is not None and name not in names:
            continue
        total += float(grad.double().pow(2).sum().item())
    return math.sqrt(total)


def _state_checksum(optimizer):
    total = 0.0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and value.is_floating_point():
                total += float(value.double().sum().item())
    return total


def check_step(model, master_model, optimizer, prepared_batch, step, kd_args, kd_mix_args,
               transfer, copy_downcast, dtype, eta=None):
    """The gates the advisor required before any multi-day TA3 arm.

    1. eta = 0 must reproduce the ordinary full-batch gradient (relative L2 and cosine).
       Run this check with dropout OFF (RWKV_DROPOUT_SCALE=0): with dropout on, the A and Q
       phases necessarily draw different masks, which is a noise floor no implementation
       error can be separated from.
    2. A nonzero eta must CHANGE the gradient while restoring the weights exactly.
    3. The trial perturbation must leave the optimizer state untouched.
    """
    if eta is None:
        eta = ETA
    _maybe_report(master_model)
    names = scope_names(master_model)
    scope_set = set(names)
    before = {name: param.detach().clone() for name, param in master_model.named_parameters()}
    state_before = _state_checksum(optimizer)

    # Reference: the ordinary path, one full-batch forward/backward.
    _clear_master_grads(master_model)
    rng_at_plain = _sam.save_rng()
    stats_plain = model.get_loss(prepared_batch, kd=kd_args, kd_mix=kd_mix_args)
    if stats_plain is None or not torch.isfinite(stats_plain.average_loss):
        raise RuntimeError("TA3 check: non-finite plain loss")
    stats_plain.average_loss.backward()
    transfer()
    g_plain = _grad_map(master_model)
    loss_plain = float(stats_plain.average_loss.item())

    def _ta3_grad(eta_value):
        stats = training_step(model, master_model, prepared_batch, kd_args, kd_mix_args,
                              transfer, copy_downcast, dtype, eta=eta_value)
        return _grad_map(master_model), stats

    # Diagnostic: is the two-GROUP decomposition exact on its own (no A/Q split, no weights)?
    # If this is not ~float noise, the problem is in the group masks; if it is, the problem is
    # in the per-phase weights.
    _clear_master_grads(master_model)
    stats_ah = model.get_loss(_mask_mode(prepared_batch, "ahead"), kd=kd_args, kd_mix=kd_mix_args)
    stats_ah.average_loss.backward()
    transfer()
    g_ah = _grad_map(master_model)
    _clear_master_grads(master_model)
    stats_im = model.get_loss(_mask_mode(prepared_batch, "imm"), kd=kd_args, kd_mix=kd_mix_args)
    stats_im.average_loss.backward()
    transfer()
    g_im = _grad_map(master_model)
    _clear_master_grads(master_model)
    g_groups = {}
    for name in set(g_ah) | set(g_im):
        left, right = g_ah.get(name), g_im.get(name)
        if left is None:
            g_groups[name] = right
        elif right is None:
            g_groups[name] = left
        else:
            g_groups[name] = left + right
    diag_diff = {n: g_groups[n] - g_plain[n] for n in g_groups if n in g_plain}
    # Repeatability floor: the SAME plain forward/backward, run twice, no masks involved. Any
    # difference here is the noise floor of every comparison in this check.
    _clear_master_grads(master_model)
    _sam.restore_rng(rng_at_plain)
    try:  # the RWKV forward keeps a rotation cache when that lever is on; clear it like the loop does
        from rwkv.model import rwkv_model as _rc_mod
        _rc_mod.shift_rot_cache_clear()
    except Exception:
        pass
    stats_rep = model.get_loss(prepared_batch, kd=kd_args, kd_mix=kd_mix_args)
    stats_rep.average_loss.backward()
    transfer()
    g_rep = _grad_map(master_model)
    _clear_master_grads(master_model)
    rep_diff = {n: g_rep[n] - g_plain[n] for n in g_rep if n in g_plain}
    rep_rel_l2 = _l2(rep_diff) / max(_l2(g_plain), 1e-30)
    diag_cos = (sum(float((g_groups[n].double() * g_plain[n].double()).sum().item())
                    for n in g_groups if n in g_plain)
                / max(_l2(g_groups) * _l2(g_plain), 1e-30))
    diag = {"groups_only_rel_l2": _l2(diag_diff) / max(_l2(g_plain), 1e-30),
            "groups_only_cos": diag_cos,
            "ahead_rows_full": _mode_counts(prepared_batch)[0],
            "imm_rows_full": _mode_counts(prepared_batch)[1],
            # Value-level check: do the six terms' values add up to average_loss?
            "loss_groups_value": (
                0.5 * float(stats_plain.ahead_avg.item())
                + 0.5 * float(stats_plain.ahead_raw_avg.item())
                + 1e-5 * float(stats_plain.w_loss_avg.item())
                + 1e-4 * float(stats_plain.ahead_logits_mag_loss_avg.item())
                + 1e-3 * float(stats_plain.ahead_logits_diff_loss_avg.item())
                + 1.0 * float(stats_plain.imm_avg.item())),
            "loss_value": loss_plain,
            "masked_group_loss_sum": float(stats_ah.average_loss.item() + stats_im.average_loss.item()),
            "repeat_loss_delta": abs(float(stats_rep.average_loss.item()) - loss_plain),
            "repeat_grad_rel_l2": rep_rel_l2}

    g_zero, stats_a0 = _ta3_grad(0.0)
    drift_after_zero = max(float((param.detach() - before[name]).abs().max().item())
                           for name, param in master_model.named_parameters())
    g_eta, stats_a = _ta3_grad(eta)
    drift_after_eta = max(float((param.detach() - before[name]).abs().max().item())
                          for name, param in master_model.named_parameters())
    state_after = _state_checksum(optimizer)

    def _compare(other):
        diff = {name: g_zero[name] - other[name] for name in g_zero if name in other}
        norm_diff = _l2(diff)
        norm_plain = max(_l2(g_plain), 1e-30)
        dot = sum(float((g_zero[n].double() * other[n].double()).sum().item())
                  for n in g_zero if n in other)
        cos = dot / max(_l2(g_zero) * _l2(other), 1e-30)
        diff_scoped = {name: value for name, value in diff.items() if name in scope_set}
        return {
            "rel_l2_zero_vs_plain": norm_diff / norm_plain,
            "rel_l2_zero_vs_plain_scope": _l2(diff_scoped) / max(_l2(g_plain, scope_set), 1e-30),
            "cos_zero_vs_plain": cos,
        }

    rec = {
        "step": step, "eta": eta, "scope": SCOPE, "n_scope_params": sum(
            dict(master_model.named_parameters())[n].numel() for n in names),
        "loss_plain": loss_plain,
        "loss_a": float(stats_a0.average_loss.item()),
        "grad_norm_plain": _l2(g_plain),
        "grad_norm_zero": _l2(g_zero),
        "grad_norm_eta": _l2(g_eta),
        "grad_norm_plain_scope": _l2(g_plain, scope_set),
        "grad_norm_zero_scope": _l2(g_zero, scope_set),
        "grad_norm_eta_scope": _l2(g_eta, scope_set),
        "weight_drift_zero": drift_after_zero,
        "weight_drift_eta": drift_after_eta,
        "optimizer_state_delta": state_after - state_before,
        "eta_minus_zero_vs_plain": {
            "rel_l2": _l2({n: g_eta[n] - g_zero[n] for n in g_eta}) / max(_l2(g_plain), 1e-30),
            "rel_l2_scope": _l2({n: g_eta[n] - g_zero[n] for n in g_eta if n in scope_set})
            / max(_l2(g_plain, scope_set), 1e-30),
        },
    }
    rec.update(_compare(g_plain))
    rec.update(diag)
    _check_seen[0] += 1
    print("[ta3-check] " + json.dumps(rec), flush=True)

    # Restore the ordinary gradient so the caller's step (if any) is unaffected, and prove the
    # weights/optimizer are exactly where they were.
    _clear_master_grads(master_model)
    for name, param in master_model.named_parameters():
        if name in g_plain:
            param.grad = g_plain[name].clone()
    return rec


def exit_after_probe(step):
    """Probe mode never trains: exit cleanly once the requested steps are measured."""
    if probe_should_stop(step):
        print(f"[ta3-probe] done after {step} probed steps; no optimizer step was taken",
              flush=True)
        sys.exit(0)
