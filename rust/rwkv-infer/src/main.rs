mod model;

use anyhow::Result;
use candle_core::{Device, Tensor};
use model::{Model, StreamState};
use std::collections::HashMap;
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
