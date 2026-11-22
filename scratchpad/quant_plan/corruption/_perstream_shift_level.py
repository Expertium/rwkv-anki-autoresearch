"""Make RWKV_STATE_SHIFT_LEVEL PER-STREAM in the Rust deploy engine.

Today it takes ONE bare level ("int4") and a "card:int4,note:int8" string is silently ignored
(`parse_level` returns None). That is the blocker behind the loss-only rows 31/34/37 and row 35
(n1i8s3, the smallest passing row): their card and note want DIFFERENT shift widths, so the engine
cannot be pointed at them. The fixed-basis rows want the same thing (card int4 + note int4 is fine,
but a mixed card/note candidate cannot be expressed).

Same string shape as RWKV_STATE_QUANT_SCOPE: a bare level still means EVERY stream (so every past
number reproduces), while "card:int4,note:int8" sets only the named streams -- the rest follow their
WKV factor level, exactly as today.

Fields renamed: CompressCfg::shift_qmax_override / Model::shift_qmax_override (both Option<f64>)
-> shift_level: HashMap<usize, f64>.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "rust" / "rwkv-infer" / "src" / "model.rs"
FAST = ROOT / "rust" / "rwkv-infer" / "src" / "fast.rs"

CFG_OLD = b"""    pub shift_qmax_override: Option<f64>,
    pub percol: bool,
"""
CFG_NEW = b"""    pub shift_level: std::collections::HashMap<usize, f64>,
    pub percol: bool,
"""

MODEL_FIELD_OLD = b"""    shift_qmax_override: Option<f64>,
    // RWKV_LOWRANK_PERCOL=1: quantize low-rank factors with a PER-COLUMN (per rank-component) scale
"""
MODEL_FIELD_NEW = b"""    // RWKV_STATE_SHIFT_LEVEL, per stream: module_idx -> shift qmax. A BARE level in the env fills all
    // five streams; a "card:int4,note:int8" list fills only the named ones. See CompressCfg::shift_level.
    shift_level: std::collections::HashMap<usize, f64>,
    // RWKV_LOWRANK_PERCOL=1: quantize low-rank factors with a PER-COLUMN (per rank-component) scale
"""

PARSE_OLD = b"""        let shift_qmax_override = std::env::var("RWKV_STATE_SHIFT_LEVEL")
            .ok()
            .and_then(|v| parse_level(v.as_str()));
"""
PARSE_NEW = b"""        // BARE level ("int4") = every compressed stream; PER-STREAM list ("card:int4,note:int8") =
        // only the named streams, the rest follow their WKV factor level. Same shape as the quant scope.
        let mut shift_level: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        for entry in std::env::var("RWKV_STATE_SHIFT_LEVEL").unwrap_or_default().split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            match entry.split_once(':') {
                Some((n, lvl)) => {
                    if let Some(q) = parse_level(lvl) {
                        shift_level.insert(name_to_idx(n), q);
                    }
                }
                None => {
                    if let Some(q) = parse_level(entry) {
                        for i in 0..5 {
                            shift_level.insert(i, q);
                        }
                    }
                }
            }
        }
"""

CONSUMER_OLD = b"""                base.map(|b| self.shift_qmax_override.unwrap_or(b))"""
CONSUMER_NEW = b"""                base.map(|b| self.shift_level.get(&module_idx).copied().unwrap_or(b))"""

FAST_CONSUMER_OLD = b"""let shift_qmax = base.map(|b| cfg.shift_qmax_override.unwrap_or(b));"""
FAST_CONSUMER_NEW = b"""let shift_qmax = base.map(|b| cfg.shift_level.get(&m).copied().unwrap_or(b));"""


def repl(raw: bytes, old: bytes, new: bytes, eol: bytes, count: int, label: str) -> bytes:
    ob = old.replace(b"\n", eol)
    n = raw.count(ob)
    if n < count:
        raise SystemExit(f"{label}: anchor matched {n} times, need {count}")
    return raw.replace(ob, new.replace(b"\n", eol), count)


raw = MODEL.read_bytes()
if b"shift_level" in raw:
    raise SystemExit("model.rs already carries shift_level")
raw = repl(raw, CFG_OLD, CFG_NEW, b"\r\n", 1, "CompressCfg field")
raw = repl(raw, MODEL_FIELD_OLD, MODEL_FIELD_NEW, b"\r\n", 1, "Model field")
raw = repl(raw, PARSE_OLD, PARSE_NEW, b"\r\n", 1, "env parse")
raw = repl(raw, CONSUMER_OLD, CONSUMER_NEW, b"\r\n", 1, "candle consumer")
# the two init sites: CompressCfg first (gets a clone), then Self (moves)
raw = repl(raw, b"            shift_qmax_override,\n", b"            shift_level: shift_level.clone(),\n",
           b"\r\n", 1, "compress_cfg init")
raw = repl(raw, b"            shift_qmax_override,\n", b"            shift_level,\n", b"\r\n", 1, "Self init")
MODEL.write_bytes(raw)
print(f"patched model.rs ({len(raw)} bytes)")

raw = FAST.read_bytes()
raw = repl(raw, FAST_CONSUMER_OLD, FAST_CONSUMER_NEW, b"\n", 1, "fast consumer")
FAST.write_bytes(raw)
print(f"patched fast.rs ({len(raw)} bytes)")
