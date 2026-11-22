"""Finish the model.rs half of the Rust finiteness wiring (the file is CRLF, main.rs is LF)."""
import re
from pathlib import Path

MODEL = Path(__file__).resolve().parents[3] / "rust" / "rwkv-infer" / "src" / "model.rs"


def block_re(block: str) -> str:
    """Escape a multi-line block into a regex that accepts LF or CRLF between lines."""
    return r"\r?\n".join(re.escape(line) for line in block.split("\n"))


QV_OLD = """pub fn quant_vec_inplace(v: &mut [f32], qmax: f64) {
    let amax = v.iter().fold(0f32, |m, &x| m.max(x.abs())) as f64;
    let scale = (amax / qmax).max(1e-12);
    for x in v.iter_mut() {
        *x = (((*x as f64) / scale).round().clamp(-qmax, qmax) * scale) as f32;
    }
}"""

QV_NEW = """pub fn quant_vec_inplace(v: &mut [f32], qmax: f64) {
    crate::finite::vec_raw(v); // observation only (RWKV_FINITE_CHECK)
    let amax = v.iter().fold(0f32, |m, &x| m.max(x.abs())) as f64;
    // A non-finite input makes amax non-finite, hence scale inf, hence every output NaN - so the exit
    // count below is the honest "did quantization produce garbage" number. The value itself is kept.
    let scale = (amax / qmax).max(1e-12);
    for x in v.iter_mut() {
        *x = (((*x as f64) / scale).round().clamp(-qmax, qmax) * scale) as f32;
    }
    crate::finite::vec_quant(v);
}"""

ADD_OLD = """        let a_r = &uf * vf.transpose(); // (k,k)
        for rr in 0..k {
            for cc in 0..k {
                data[off + rr * k + cc] = a_r[(rr, cc)];
            }
        }
"""

text = MODEL.open(encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in text else "\n"

pairs = [
    (block_re(QV_OLD), QV_NEW.replace("\n", nl), "quant_vec_inplace in full"),
    (r"    let r = rank\.min\(k\);",
     "    crate::finite::state_raw(data); // observation only (RWKV_FINITE_CHECK)" + nl + "    let r = rank.min(k);",
     "compress_wkv_state entry"),
    (block_re(ADD_OLD),
     ADD_OLD.replace("\n", nl) + "        crate::finite::state_quant(&data[off..off + k * k]); "
     "// observation only" + nl,
     "compress_wkv_state exit"),
]

for pat, rep, what in pairs:
    text, n = re.subn(pat, lambda _m, r=rep: r, text, count=1)
    if n != 1:
        raise SystemExit(f"{what}: expected 1 match, got {n}")
    print(f"ok: {what}")

with MODEL.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("model.rs patched")
