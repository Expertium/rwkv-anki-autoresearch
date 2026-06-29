mod model;

use anyhow::Result;
use candle_core::{Device, Tensor};
use model::{stack_stream_states, BatchedStreamState, Model, StreamState};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

const REF_USERS: [i64; 3] = [107, 136, 156];

#[derive(serde::Serialize)]
struct UserPreds {
    user: i64,
    review_th: Vec<i64>,
    pred_imm: Vec<f32>,
    pred_ahead: Vec<Option<f32>>,
}

fn run_user(model: &Model, user: i64) -> Result<()> {
    let dev = Device::Cpu;
    let trace = format!("reference/trace_user_{user}.safetensors");
    let t = candle_core::safetensors::load(&trace, &dev)?;

    let feats_imm = t.get("feats_imm").unwrap(); // (N,92)
    let feats_proc = t.get("feats_proc").unwrap();
    let route: Vec<Vec<i64>> = t.get("route").unwrap().to_vec2()?; // (N,4): [card,note,deck,preset]
    let elapsed: Vec<f32> = t.get("elapsed_seconds").unwrap().to_vec1()?;
    let review_th: Vec<i64> = t.get("review_th").unwrap().to_vec1()?;
    let n = review_th.len();

    // RWKV states keyed by dense per-stream id; global is a singleton.
    let mut s_card: HashMap<i64, StreamState> = HashMap::new();
    let mut s_deck: HashMap<i64, StreamState> = HashMap::new();
    let mut s_note: HashMap<i64, StreamState> = HashMap::new();
    let mut s_preset: HashMap<i64, StreamState> = HashMap::new();
    let mut s_global: Option<StreamState> = None;
    // stored forgetting curve per card: (out_ahead_logits, out_w)
    let mut curve: HashMap<i64, (Tensor, Tensor)> = HashMap::new();

    let mut pred_imm = vec![0.0f32; n];
    let mut pred_ahead = vec![None; n];

    let t0 = Instant::now();
    for i in 0..n {
        let cidx = route[i][0];
        let nidx = route[i][1];
        let didx = route[i][2];
        let pidx = route[i][3];

        // ahead prediction from the card's previously stored curve
        if let Some((al, ow)) = curve.get(&cidx) {
            pred_ahead[i] = Some(model.predict_ahead(al, ow, elapsed[i])?);
        }

        // states in chain order: [card, deck, note, preset, user]
        let states: [Option<StreamState>; 5] = [
            s_card.get(&cidx).cloned(),
            s_deck.get(&didx).cloned(),
            s_note.get(&nidx).cloned(),
            s_preset.get(&pidx).cloned(),
            s_global.clone(),
        ];

        // immediate forward (state read-only)
        let fi = feats_imm.narrow(0, i, 1)?; // (1,92)
        let (_, _, out_p, _) = model.review(&fi, &states)?;
        pred_imm[i] = model.imm_prob(&out_p)?;

        // ahead-of-time forward (updates state, stores curve)
        let fp = feats_proc.narrow(0, i, 1)?;
        let (al, ow, _, new_states) = model.review(&fp, &states)?;
        let [n0, n1, n2, n3, n4] = new_states;
        s_card.insert(cidx, n0);
        s_deck.insert(didx, n1);
        s_note.insert(nidx, n2);
        s_preset.insert(pidx, n3);
        s_global = Some(n4);
        curve.insert(cidx, (al, ow));

        if (i + 1) % 1000 == 0 {
            let rate = (i + 1) as f64 / t0.elapsed().as_secs_f64();
            println!("  user {user}: {}/{n}  ({rate:.1} rev/s)", i + 1);
        }
    }

    let out = UserPreds {
        user,
        review_th,
        pred_imm,
        pred_ahead,
    };
    let path = format!("reference/rust_pred_{user}.json");
    std::fs::write(&path, serde_json::to_string(&out)?)?;
    let rate = n as f64 / t0.elapsed().as_secs_f64();
    println!("user {user}: {n} reviews in {:.1}s ({rate:.1} rev/s) -> {path}", t0.elapsed().as_secs_f64());
    Ok(())
}

/// Throughput bench: replay user `user`'s trace (full forward work per review) in a loop for
/// `secs` wall-clock seconds, single-thread, B=1; print the review count. The Wilcoxon driver
/// runs several of these simultaneously (before vs after) for paired timed trials.
fn bench(model: &Model, user: i64, secs: f64) -> Result<()> {
    let dev = Device::Cpu;
    let t = candle_core::safetensors::load(&format!("reference/trace_user_{user}.safetensors"), &dev)?;
    let feats_imm = t.get("feats_imm").unwrap();
    let feats_proc = t.get("feats_proc").unwrap();
    let route: Vec<Vec<i64>> = t.get("route").unwrap().to_vec2()?;
    let elapsed: Vec<f32> = t.get("elapsed_seconds").unwrap().to_vec1()?;
    let n = route.len();
    let mut total: u64 = 0;
    let t0 = Instant::now();
    'outer: while t0.elapsed().as_secs_f64() < secs {
        let mut s_card: HashMap<i64, StreamState> = HashMap::new();
        let mut s_deck: HashMap<i64, StreamState> = HashMap::new();
        let mut s_note: HashMap<i64, StreamState> = HashMap::new();
        let mut s_preset: HashMap<i64, StreamState> = HashMap::new();
        let mut s_global: Option<StreamState> = None;
        let mut curve: HashMap<i64, (Tensor, Tensor)> = HashMap::new();
        for i in 0..n {
            let (cidx, nidx, didx, pidx) = (route[i][0], route[i][1], route[i][2], route[i][3]);
            if let Some((al, ow)) = curve.get(&cidx) {
                let _ = model.predict_ahead(al, ow, elapsed[i])?;
            }
            let states: [Option<StreamState>; 5] = [
                s_card.get(&cidx).cloned(),
                s_deck.get(&didx).cloned(),
                s_note.get(&nidx).cloned(),
                s_preset.get(&pidx).cloned(),
                s_global.clone(),
            ];
            let fi = feats_imm.narrow(0, i, 1)?;
            let (_, _, _op, _) = model.review(&fi, &states)?;
            let fp = feats_proc.narrow(0, i, 1)?;
            let (al, ow, _, new_states) = model.review(&fp, &states)?;
            let [n0, n1, n2, n3, n4] = new_states;
            s_card.insert(cidx, n0);
            s_deck.insert(didx, n1);
            s_note.insert(nidx, n2);
            s_preset.insert(pidx, n3);
            s_global = Some(n4);
            curve.insert(cidx, (al, ow));
            total += 1;
            if t0.elapsed().as_secs_f64() >= secs {
                break 'outer;
            }
        }
    }
    let el = t0.elapsed().as_secs_f64();
    println!("BENCH reviews={total} secs={el:.2} rev_s={:.1}", total as f64 / el);
    Ok(())
}

/// Warmed per-user state after a full sequential replay (the JSchoreels "warmup" phase).
struct Warmed {
    s_card: HashMap<i64, StreamState>,
    s_deck: HashMap<i64, StreamState>,
    s_note: HashMap<i64, StreamState>,
    s_preset: HashMap<i64, StreamState>,
    s_global: StreamState,
    card_route: HashMap<i64, (i64, i64, i64)>, // card -> (note, deck, preset) last seen
    card_feat_row: HashMap<i64, usize>,        // card -> last review row (into feats_imm)
    card_order: Vec<i64>,                      // distinct cards, first-seen order
    feats_imm: Tensor,                         // (N,92)
}

/// Replay a user's full trace sequentially (state-updating "ahead" path) to build warmed states.
fn warmup(model: &Model, user: i64) -> Result<Warmed> {
    let dev = Device::Cpu;
    let t = candle_core::safetensors::load(&format!("reference/trace_user_{user}.safetensors"), &dev)?;
    let feats_imm = t.get("feats_imm").unwrap().clone();
    let feats_proc = t.get("feats_proc").unwrap();
    let route: Vec<Vec<i64>> = t.get("route").unwrap().to_vec2()?;
    let n = route.len();

    let mut s_card: HashMap<i64, StreamState> = HashMap::new();
    let mut s_deck: HashMap<i64, StreamState> = HashMap::new();
    let mut s_note: HashMap<i64, StreamState> = HashMap::new();
    let mut s_preset: HashMap<i64, StreamState> = HashMap::new();
    let mut s_global: Option<StreamState> = None;
    let mut card_route: HashMap<i64, (i64, i64, i64)> = HashMap::new();
    let mut card_feat_row: HashMap<i64, usize> = HashMap::new();
    let mut card_order: Vec<i64> = Vec::new();
    let mut seen: HashSet<i64> = HashSet::new();

    for i in 0..n {
        let (cidx, nidx, didx, pidx) = (route[i][0], route[i][1], route[i][2], route[i][3]);
        let states: [Option<StreamState>; 5] = [
            s_card.get(&cidx).cloned(),
            s_deck.get(&didx).cloned(),
            s_note.get(&nidx).cloned(),
            s_preset.get(&pidx).cloned(),
            s_global.clone(),
        ];
        let fp = feats_proc.narrow(0, i, 1)?;
        let (_, _, _, new_states) = model.review(&fp, &states)?;
        let [n0, n1, n2, n3, n4] = new_states;
        s_card.insert(cidx, n0);
        s_deck.insert(didx, n1);
        s_note.insert(nidx, n2);
        s_preset.insert(pidx, n3);
        s_global = Some(n4);
        card_route.insert(cidx, (nidx, didx, pidx));
        card_feat_row.insert(cidx, i);
        if seen.insert(cidx) {
            card_order.push(cidx);
        }
    }
    Ok(Warmed {
        s_card,
        s_deck,
        s_note,
        s_preset,
        s_global: s_global.expect("user had no reviews"),
        card_route,
        card_feat_row,
        card_order,
        feats_imm,
    })
}

/// Gather, for a list of cards, the per-card B=1 state arrays + the batched (B,...) states + (B,92)
/// feats. Chain order is [card, deck, note, preset, global] (matches Model::review).
#[allow(clippy::type_complexity)]
fn gather_batch(
    wm: &Warmed,
    cards: &[i64],
) -> Result<(Vec<[Option<StreamState>; 5]>, [Option<BatchedStreamState>; 5], Tensor)> {
    let mut per_card: Vec<[Option<StreamState>; 5]> = Vec::with_capacity(cards.len());
    let (mut card_v, mut deck_v, mut note_v, mut preset_v, mut global_v) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut feat_rows: Vec<Tensor> = Vec::with_capacity(cards.len());
    for &c in cards {
        let (n, d, p) = wm.card_route[&c];
        let cs = wm.s_card[&c].clone();
        let ds = wm.s_deck[&d].clone();
        let ns = wm.s_note[&n].clone();
        let ps = wm.s_preset[&p].clone();
        let gs = wm.s_global.clone();
        per_card.push([
            Some(cs.clone()),
            Some(ds.clone()),
            Some(ns.clone()),
            Some(ps.clone()),
            Some(gs.clone()),
        ]);
        card_v.push(cs);
        deck_v.push(ds);
        note_v.push(ns);
        preset_v.push(ps);
        global_v.push(gs);
        feat_rows.push(wm.feats_imm.narrow(0, wm.card_feat_row[&c], 1)?);
    }
    let batched: [Option<BatchedStreamState>; 5] = [
        Some(stack_stream_states(&card_v)?),
        Some(stack_stream_states(&deck_v)?),
        Some(stack_stream_states(&note_v)?),
        Some(stack_stream_states(&preset_v)?),
        Some(stack_stream_states(&global_v)?),
    ];
    let feats_b = Tensor::cat(&feat_rows, 0)?; // (B,92)
    Ok((per_card, batched, feats_b))
}

/// Assert the batched query forward matches the B=1 query forward per card (max |imm diff|).
fn verify_batched(model: &Model, user: i64) -> Result<()> {
    let wm = warmup(model, user)?;
    let cards = wm.card_order.clone();
    let b = cards.len();
    let (per_card, batched, feats_b) = gather_batch(&wm, &cards)?;

    // B=1 reference imm per card
    let mut ref_imm = Vec::with_capacity(b);
    for (idx, st) in per_card.iter().enumerate() {
        let fi = feats_b.narrow(0, idx, 1)?;
        let (_, _, out_p, _) = model.review(&fi, st)?;
        ref_imm.push(model.imm_prob(&out_p)?);
    }
    // batched imm
    let (_, _, out_p_b, _) = model.review_batched(&feats_b, &batched)?;
    let imm_b = model.imm_prob_batched(&out_p_b)?;

    let mut maxdiff = 0f32;
    let mut argmax = 0usize;
    for i in 0..b {
        let d = (ref_imm[i] - imm_b[i]).abs();
        if d > maxdiff {
            maxdiff = d;
            argmax = i;
        }
    }
    println!(
        "verify-batched user {user}: B={b} distinct cards, max|imm_batched - imm_B1| = {maxdiff:.3e} \
         (card {} B1={:.6} batched={:.6})",
        cards[argmax], ref_imm[argmax], imm_b[argmax]
    );
    if maxdiff < 1e-4 {
        println!("  PASS (batched matches B=1 within 1e-4)");
    } else {
        println!("  FAIL (diff exceeds 1e-4)");
    }
    Ok(())
}

/// Throughput: B=1 single-step queries vs one batched (B,C) query, both over `bmax` cards (cards
/// are cycled to reach bmax if the user has fewer). Reports rev/s for each and the speedup.
fn bench_batched(model: &Model, user: i64, secs: f64, bmax: usize) -> Result<()> {
    let wm = warmup(model, user)?;
    if wm.card_order.is_empty() {
        anyhow::bail!("user {user} has no cards");
    }
    // cycle distinct cards to fill bmax (throughput only; correctness uses real cards via verify)
    let cards: Vec<i64> = (0..bmax)
        .map(|i| wm.card_order[i % wm.card_order.len()])
        .collect();
    let (per_card, batched, feats_b) = gather_batch(&wm, &cards)?;
    let b = cards.len();

    // B=1 single-step query loop
    let mut total1: u64 = 0;
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < secs {
        for (idx, st) in per_card.iter().enumerate() {
            let fi = feats_b.narrow(0, idx, 1)?;
            let (_, _, out_p, _) = model.review(&fi, st)?;
            let _ = model.imm_prob(&out_p)?;
            total1 += 1;
        }
    }
    let el1 = t0.elapsed().as_secs_f64();
    let rps1 = total1 as f64 / el1;

    // batched (B,C) query loop
    let mut total_b: u64 = 0;
    let t1 = Instant::now();
    while t1.elapsed().as_secs_f64() < secs {
        let (_, _, out_p_b, _) = model.review_batched(&feats_b, &batched)?;
        let _ = model.imm_prob_batched(&out_p_b)?;
        total_b += b as u64;
    }
    let elb = t1.elapsed().as_secs_f64();
    let rpsb = total_b as f64 / elb;

    println!("BENCH-BATCHED user {user} B={b}");
    println!("  B=1   queries: {total1} in {el1:.2}s -> {rps1:.1} rev/s");
    println!("  batch queries: {total_b} in {elb:.2}s -> {rpsb:.1} rev/s");
    println!("  speedup: {:.2}x", rpsb / rps1);
    Ok(())
}

/// Sweep batch size B = 1,2,4,...,maxb through the SAME batched query kernel, warming up once.
/// Prints "B<TAB>rev_s<TAB>speedup_vs_B1" so a plotter can read throughput-vs-batch-size.
fn sweep_batched(model: &Model, user: i64, secs: f64, maxb: usize) -> Result<()> {
    let wm = warmup(model, user)?;
    if wm.card_order.is_empty() {
        anyhow::bail!("user {user} has no cards");
    }
    let mut bs = Vec::new();
    let mut b = 1usize;
    while b <= maxb {
        bs.push(b);
        b *= 2;
    }
    eprintln!("warmup done ({} distinct cards); sweeping B=1..{maxb}", wm.card_order.len());
    println!("# user {user} secs_per_B {secs}");
    println!("# B\trev_s\tspeedup");
    let mut rps1 = 0.0f64;
    for (i, &bb) in bs.iter().enumerate() {
        let cards: Vec<i64> = (0..bb).map(|j| wm.card_order[j % wm.card_order.len()]).collect();
        let (_per, batched, feats_b) = gather_batch(&wm, &cards)?;
        let mut total: u64 = 0;
        let t0 = Instant::now();
        while t0.elapsed().as_secs_f64() < secs {
            let (_, _, out_p, _) = model.review_batched(&feats_b, &batched)?;
            let _ = model.imm_prob_batched(&out_p)?;
            total += bb as u64;
        }
        let rps = total as f64 / t0.elapsed().as_secs_f64();
        if i == 0 {
            rps1 = rps;
        }
        println!("{bb}\t{rps:.1}\t{:.3}", rps / rps1);
        eprintln!("  B={bb:5}  {rps:8.1} rev/s  ({:.2}x)", rps / rps1);
    }
    Ok(())
}

/// Build synthetic batched states of the correct shapes for a batch of B cards (random values --
/// dense-matmul timing and allocation depend only on shapes, not values). Lets a RAM/speed sweep
/// skip the expensive per-B warmup replay while measuring identical compute and memory.
fn synth_states(model: &Model, b: usize, dev: &Device) -> Result<[Option<BatchedStreamState>; 5]> {
    let (h, k, c) = model.dims();
    let layers = model.stream_layers().to_vec();
    let mut arr: Vec<Option<BatchedStreamState>> = Vec::with_capacity(5);
    for m in 0..5 {
        let mut st: BatchedStreamState = Vec::with_capacity(layers[m]);
        for _ in 0..layers[m] {
            st.push(model::BatchedLayerState {
                t_xshift: Tensor::rand(-1f32, 1f32, (b, c), dev)?,
                t_state: Tensor::rand(-1f32, 1f32, (b, h, k, k), dev)?,
                c_xshift: Tensor::rand(-1f32, 1f32, (b, c), dev)?,
            });
        }
        arr.push(Some(st));
    }
    let arr: [Option<BatchedStreamState>; 5] =
        arr.try_into().map_err(|_| anyhow::anyhow!("stream count"))?;
    Ok(arr)
}

/// Synthetic batched query throughput at a fixed B (no warmup). Prints "rev_s <value>" for a driver.
fn bench_synth(model: &Model, secs: f64, b: usize) -> Result<()> {
    let dev = Device::Cpu;
    let states = synth_states(model, b, &dev)?;
    let feats_b = Tensor::rand(0f32, 1f32, (b, 92), &dev)?;
    // one untimed warm call (allocate buffers) then timed loop
    let _ = model.review_batched(&feats_b, &states)?;
    let mut total: u64 = 0;
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < secs {
        let (_, _, out_p, _) = model.review_batched(&feats_b, &states)?;
        let _ = model.imm_prob_batched(&out_p)?;
        total += b as u64;
    }
    let el = t0.elapsed().as_secs_f64();
    println!("B {b} rev_s {:.1} reviews {total} secs {el:.2}", total as f64 / el);
    Ok(())
}

/// Warm up a user, grab a REAL card's card_id-stream WKV state, and print it fp32 vs int2 (ternary)
/// so the int2 quantization is visible: the 1024 floats collapse to 3 levels {-scale, 0, +scale}.
fn dump_card_state(model: &Model, user: i64, card_pos: usize) -> Result<()> {
    let wm = warmup(model, user)?;
    let cid = wm.card_order[card_pos.min(wm.card_order.len() - 1)];
    let st = &wm.s_card[&cid]; // card stream (1 layer in iter36/39)
    let t_state = &st[0].t_state; // (H,K,K) = (1,32,32) for d=32
    let dims = t_state.dims().to_vec();
    let fp32: Vec<f32> = t_state.flatten_all()?.to_vec1()?;
    let n = fp32.len();

    let (codes, scale) = model::quant_codes(t_state, 1.0)?; // int2: qmax=1 -> codes in {-1,0,1}
    let deq: Vec<f32> = codes.iter().map(|c| (*c as f64 * scale) as f32).collect();
    let amax = fp32.iter().fold(0f32, |m, x| m.max(x.abs()));

    let (mut nneg, mut nzero, mut npos) = (0usize, 0usize, 0usize);
    for c in &codes {
        if *c < -0.5 { nneg += 1 } else if *c > 0.5 { npos += 1 } else { nzero += 1 }
    }
    // reconstruction error
    let mse: f32 = fp32.iter().zip(&deq).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / n as f32;

    let (h, k, _c) = model.dims(); // (H,K,K); for d=32 this is (1,32,32)
    println!("user {user}  card_pos {card_pos} (dense card id {cid})");
    println!("card_id-stream WKV state shape {dims:?} = {n} floats = a {k}x{k} matrix (H={h} head)");
    println!("fp32 amax = {amax:.6}   int2 scale = {scale:.6}  (the 3 levels are -{scale:.5}, 0, +{scale:.5})");

    for hh in 0..h {
        let off = hh * k * k;
        // ---- full KxK fp32 matrix ----
        println!("\n=== fp32 {k}x{k} WKV state (head {hh}) ===");
        for r in 0..k {
            let mut line = String::new();
            for c in 0..k {
                line.push_str(&format!("{:7.3}", fp32[off + r * k + c]));
            }
            println!("{line}");
        }
        // ---- full KxK int2 code grid (-1 / 0 / +1 shown as  -  .  + ) ----
        println!("\n=== int2 {k}x{k} code grid (head {hh}):  '+' = +scale, '-' = -scale, '.' = 0 ===");
        for r in 0..k {
            let mut line = String::new();
            for c in 0..k {
                let code = codes[off + r * k + c] as i32;
                line.push(' ');
                line.push(match code {
                    1 => '+',
                    -1 => '-',
                    _ => '.',
                });
            }
            println!("{line}");
        }
    }

    println!();
    println!("--- the NONZERO int2 entries (row,col): the other {nzero} of {n} are 0 ---");
    println!("{:>3} {:>3}  {:>12}  {:>5}  {:>12}", "r", "c", "fp32", "code", "int2 deq");
    for i in 0..n {
        let code = codes[i] as i32;
        if code != 0 {
            let (r, c) = ((i % (k * k)) / k, i % k);
            println!("{:>3} {:>3}  {:>12.6}  {:>+5}  {:>+12.6}", r, c, fp32[i], code, deq[i]);
        }
    }
    println!();
    println!("int2 code histogram: -1: {nneg}   0: {nzero}   +1: {npos}   (sum {})", nneg + nzero + npos);
    println!("storage: {n} codes x 2 bits = {} bytes = {:.3} KiB  (+ one fp32 scale)",
             n * 2 / 8, (n * 2 / 8) as f64 / 1024.0);
    println!("reconstruction MSE (int2 deq vs fp32) = {mse:.3e}");

    // ===== RANK-2 LOW-RANK view (the deploy card-state format): the quantized factors, by eye =====
    {
        use nalgebra::DMatrix;
        let rank = 2usize;
        let q = 7.0f64; // int4 factors
        let a = DMatrix::<f32>::from_row_slice(k, k, &fp32[0..k * k]); // head 0
        let svd = a.clone().svd(true, true);
        let u = svd.u.as_ref().unwrap();
        let vt = svd.v_t.as_ref().unwrap();
        let sv = &svd.singular_values;
        let mut uf = DMatrix::<f32>::zeros(k, rank);
        let mut vf = DMatrix::<f32>::zeros(k, rank);
        for j in 0..rank {
            let sj = sv[j].max(0.0).sqrt();
            for i in 0..k {
                uf[(i, j)] = u[(i, j)] * sj; // A_r = (U sqrt S)(V sqrt S)^T
                vf[(i, j)] = vt[(j, i)] * sj;
            }
        }
        // int4-quantize each factor (symmetric per-matrix scale), column-major codes
        let quant = |m: &DMatrix<f32>| -> (Vec<i32>, f64, DMatrix<f32>) {
            let amax = m.iter().fold(0f32, |a, &x| a.max(x.abs())) as f64;
            let scale = (amax / q).max(1e-12);
            let mut codes = Vec::with_capacity(m.len());
            let mut deq = DMatrix::<f32>::zeros(m.nrows(), m.ncols());
            for c in 0..m.ncols() {
                for r in 0..m.nrows() {
                    let code = ((m[(r, c)] as f64) / scale).round().clamp(-q, q) as i32;
                    codes.push(code);
                    deq[(r, c)] = (code as f64 * scale) as f32;
                }
            }
            (codes, scale, deq)
        };
        let (uc, us, uq) = quant(&uf);
        let (vc, vs, vq) = quant(&vf);
        let ar = &uq * vq.transpose();
        let fro = |m: &DMatrix<f32>| m.iter().map(|x| (x * x) as f64).sum::<f64>().sqrt();
        let anorm = fro(&a);
        let recon_fp = &uf * vf.transpose();
        let err_fp = fro(&(&a - &recon_fp)) / anorm;
        let err_q = fro(&(&a - &ar)) / anorm;
        let tot_e: f64 = sv.iter().map(|s| (s * s) as f64).sum();
        println!("\n===================== RANK-2 LOW-RANK (deploy card-state format) =====================");
        let topsv: Vec<f32> = (0..6.min(sv.len())).map(|i| (sv[i] * 1000.0).round() / 1000.0).collect();
        println!("singular values (top 6): {topsv:?}");
        println!("rank-2 keeps top 2 -> Frobenius energy {:.4}", (sv[0].powi(2) + sv[1].powi(2)) as f64 / tot_e);
        println!("\n--- U factor (Kx2 = U[:,:2]*sqrt(S)) and its int4 codes  (scale {us:.5}) ---");
        println!("{:>3}  {:>9} {:>9}   {:>4} {:>4}", "i", "U[:,0]", "U[:,1]", "c0", "c1");
        for i in 0..k {
            println!("{:>3}  {:>9.4} {:>9.4}   {:>+4} {:>+4}", i, uf[(i, 0)], uf[(i, 1)], uc[i], uc[i + k]);
        }
        println!("\n--- V factor (Kx2) and its int4 codes  (scale {vs:.5}) ---");
        println!("{:>3}  {:>9} {:>9}   {:>4} {:>4}", "i", "V[:,0]", "V[:,1]", "c0", "c1");
        for i in 0..k {
            println!("{:>3}  {:>9.4} {:>9.4}   {:>+4} {:>+4}", i, vf[(i, 0)], vf[(i, 1)], vc[i], vc[i + k]);
        }
        println!("\n--- reconstruction A_r = dequant(Uq) @ dequant(Vq)^T  (KxK, from the int4 factors) ---");
        for r in 0..k {
            let mut line = String::new();
            for c in 0..k {
                line.push_str(&format!("{:7.3}", ar[(r, c)]));
            }
            println!("{line}");
        }
        println!("\nrelative Frobenius error ||A-A_r||/||A||:  rank-2 fp32 factors = {err_fp:.4}   rank-2 int4 factors = {err_q:.4}");
        let int2_bytes = k * k * 2 / 8;
        let lr_bytes = 2 * k * rank * 4 / 8; // 2 factors, K x rank, int4
        println!("storage (WKV matrix only):  int2 full = {int2_bytes} B   vs   rank-2 int4 factors = 2x{k}x{rank}x4bit/8 = {lr_bytes} B   ({:.1}x smaller)",
                 int2_bytes as f64 / lr_bytes as f64);
    }
    Ok(())
}

fn main() -> Result<()> {
    let weights_owned = std::env::var("RWKV_WEIGHTS")
        .unwrap_or_else(|_| "reference/rwkv_ref_558.safetensors".to_string());
    let weights = weights_owned.as_str();
    let model = Model::load(weights, Device::Cpu)?;
    println!("model loaded from {weights}");

    // --bench <secs> [user]: timed throughput trial
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().map(|s| s.as_str()) == Some("--bench") {
        let secs: f64 = argv.get(1).map(|s| s.parse().unwrap()).unwrap_or(20.0);
        let user: i64 = argv.get(2).map(|s| s.parse().unwrap()).unwrap_or(107);
        return bench(&model, user, secs);
    }

    // --verify-batched [user]: assert batched query == B=1 query per card
    if argv.first().map(|s| s.as_str()) == Some("--verify-batched") {
        let user: i64 = argv.get(1).map(|s| s.parse().unwrap()).unwrap_or(107);
        return verify_batched(&model, user);
    }

    // --bench-batched <secs> [user] [B]: B=1 vs batched single-step query throughput
    if argv.first().map(|s| s.as_str()) == Some("--bench-batched") {
        let secs: f64 = argv.get(1).map(|s| s.parse().unwrap()).unwrap_or(20.0);
        let user: i64 = argv.get(2).map(|s| s.parse().unwrap()).unwrap_or(107);
        let bmax: usize = argv.get(3).map(|s| s.parse().unwrap()).unwrap_or(512);
        return bench_batched(&model, user, secs, bmax);
    }

    // --sweep-batched <secs_per_B> [user] [maxB]: throughput vs batch size, warm up once
    if argv.first().map(|s| s.as_str()) == Some("--sweep-batched") {
        let secs: f64 = argv.get(1).map(|s| s.parse().unwrap()).unwrap_or(5.0);
        let user: i64 = argv.get(2).map(|s| s.parse().unwrap()).unwrap_or(107);
        let maxb: usize = argv.get(3).map(|s| s.parse().unwrap()).unwrap_or(2048);
        return sweep_batched(&model, user, secs, maxb);
    }

    // --dump-card-state [user] [card_pos]: print a real card's WKV state, fp32 vs int2 (ternary)
    if argv.first().map(|s| s.as_str()) == Some("--dump-card-state") {
        let user: i64 = argv.get(1).map(|s| s.parse().unwrap()).unwrap_or(107);
        let card_pos: usize = argv.get(2).map(|s| s.parse().unwrap()).unwrap_or(0);
        return dump_card_state(&model, user, card_pos);
    }

    // --bench-synth <secs> <B>: synthetic-state batched throughput at a single B (no warmup).
    // One subprocess per B lets an external driver measure peak RSS for a speed-vs-RAM frontier.
    if argv.first().map(|s| s.as_str()) == Some("--bench-synth") {
        let secs: f64 = argv.get(1).map(|s| s.parse().unwrap()).unwrap_or(4.0);
        let b: usize = argv.get(2).map(|s| s.parse().unwrap()).unwrap_or(512);
        return bench_synth(&model, secs, b);
    }

    if std::env::var("RWKV_DEBUG").is_ok() {
        // one-shot: review 0 of user 107 with zero state, dump intermediates
        let dev = Device::Cpu;
        let t = candle_core::safetensors::load("reference/trace_user_107.safetensors", &dev)?;
        let fi = t.get("feats_imm").unwrap().narrow(0, 0, 1)?;
        let states: [Option<StreamState>; 5] = [None, None, None, None, None];
        let (_, _, out_p, _) = model.review(&fi, &states)?;
        eprintln!("imm = {:.6}", model.imm_prob(&out_p)?);
        return Ok(());
    }

    // optional CLI user ids, else the default reference set
    let args: Vec<String> = std::env::args().skip(1).collect();
    let users: Vec<i64> = if args.is_empty() {
        REF_USERS.to_vec()
    } else {
        args.iter().map(|s| s.parse().unwrap()).collect()
    };
    for u in users {
        run_user(&model, u)?;
    }
    Ok(())
}
