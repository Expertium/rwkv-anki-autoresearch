"""Wire the new finite.rs observation module into the deploy engine (Andrew, 2026-11-22 12:02).

main.rs: declare the module, count every p(recall) through san(), announce the mode, report per user and
in total, and exit 3 in STRICT mode when anything was non-finite.
model.rs: count the state entering a compression, the state written back, and the vectors handled by
quant_vec_inplace (token shifts / full-matrix state).

Replacement strings are used LITERALLY (function replacement in re.subn), so no \\1 group references:
each site is rewritten in full and every anchor must match exactly once.
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "rust" / "rwkv-infer" / "src"


def patch(path: Path, pairs) -> None:
    text = path.open(encoding="utf-8", newline="").read()
    for pat, rep, what in pairs:
        text, n = re.subn(pat, lambda _m, r=rep: r, text, count=1)
        if n != 1:
            raise SystemExit(f"{path.name}: {what}: expected 1 match, got {n}")
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    print(f"patched {path.name}")


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
    // a non-finite input makes amax non-finite and every output NaN, so the exit count below is the
    // honest "did quantization produce garbage" number (the engine keeps the value as-is either way).
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

ADD_NEW = ADD_OLD + "        crate::finite::state_quant(&data[off..off + k * k]); // observation only\n"

patch(SRC / "main.rs", [
    (r"mod fast;\nmod model;\nmod pava;\n",
     "mod fast;\nmod finite;\nmod model;\nmod pava;\n",
     "mod declaration"),

    (r"fn san\(p: f32\) -> f32 \{\n    if p\.is_finite\(\) \{",
     "fn san(p: f32) -> f32 {\n"
     "    // observation only (RWKV_FINITE_CHECK): the deploy gate must know how often this hides a\n"
     "    // non-finite p(recall) as 0.5. Does not change the value.\n"
     "    crate::finite::p(p);\n"
     "    if p.is_finite() {",
     "san() counting"),

    (r'println!\("model loaded from \{weights\}"\);',
     'println!("model loaded from {weights}");\n    finite::announce();',
     "announce"),

    (r"    for u in users \{\n        run_user\(&model, u\)\?;\n    \}",
     "    for u in users {\n"
     "        run_user(&model, u)?;\n"
     "        finite::report_user(u);\n"
     "    }",
     "per-user report"),

    (r"(\n    Ok\(\(\)\)\n\}\n$)",
     "\n    let bad = finite::report_total();\n"
     "    if finite::strict() && bad > 0 {\n"
     "        eprintln!(\"[finite] STRICT FAIL: {bad} non-finite value(s) - see the per-user lines above\");\n"
     "        std::process::exit(3);\n"
     "    }\n"
     "    Ok(())\n"
     "}\n",
     "total report + strict exit"),
])

patch(SRC / "model.rs", [
    (re.escape(QV_OLD), QV_NEW, "quant_vec_inplace in full"),
    (r"    let r = rank\.min\(k\);",
     "    crate::finite::state_raw(data); // observation only (RWKV_FINITE_CHECK)\n"
     "    let r = rank.min(k);",
     "compress_wkv_state entry"),
    (re.escape(ADD_OLD), ADD_NEW, "compress_wkv_state exit"),
])
