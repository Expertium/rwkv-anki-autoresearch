"""Fix the fixed-basis Rust unit test that measured truncation instead of quantization (and drop one
needless `mut`). A random FULL-RANK state loses ~m/K of its energy to the subspace projection alone
(rel 0.96 at m=8), so it can only measure the truncation; a state that LIVES in the basis's top-m
subspace is reproduced exactly at m and then the leftover error IS the coefficient quantization.
"""
from pathlib import Path

MODEL = Path(__file__).resolve().parents[3] / "rust" / "rwkv-infer" / "src" / "model.rs"
EOL = "\r\n"

OLD = '''        let mut take = |it: &mut std::str::SplitWhitespace| -> Vec<f32> {'''
NEW = '''        let take = |it: &mut std::str::SplitWhitespace| -> Vec<f32> {'''

OLD_T = '''    #[test]
    fn quantization_hits_the_coefficient_block() {
        // int8 on the m x m block is a small precision error; int2 on the same block is much larger.
        let p = orth(11);
        let q = orth(22);
        let orig = orth(55);
        let mut a = orig.clone();
        let mut c = orig.clone();
        fixedbasis_roundtrip(&mut a, 1, K, 8, &p, &q, K, Some(127.0));
        fixedbasis_roundtrip(&mut c, 1, K, 8, &p, &q, K, Some(1.0));
        let ea = rel_err(&a, &orig);
        let ec = rel_err(&c, &orig);
        assert!(ea < 0.35, "int8 on the block must be mild, rel={ea}");
        assert!(ec > ea * 2.0, "int2 must be clearly worse than int8 ({ec} vs {ea})");
    }'''

NEW_T = '''    #[test]
    fn quantization_hits_the_coefficient_block() {
        // A state that LIVES in the basis's top-8 subspace (decaying modes along the basis's own first
        // 8 directions): m=8 is then EXACT, so whatever error is left is the coefficient quantization
        // alone -- which is what this check is about. A random full-rank state can only measure the
        // subspace projection (~m/K of the energy), not the quantization.
        let p = orth(11);
        let q = orth(22);
        let mut orig = vec![0f32; K * K];
        let mut sigma = 3.0f32;
        for s in 0..8 {
            for i in 0..K {
                for j in 0..K {
                    orig[i * K + j] += sigma * p[i * K + s] * q[j * K + s];
                }
            }
            sigma *= 0.8;
        }
        let mut exact = orig.clone();
        fixedbasis_roundtrip(&mut exact, 1, K, 8, &p, &q, K, None);
        let e0 = rel_err(&exact, &orig);
        assert!(e0 < 1e-5, "m=8 on an in-subspace state must be exact, rel={e0}");
        let mut a = orig.clone();
        let mut c = orig.clone();
        fixedbasis_roundtrip(&mut a, 1, K, 8, &p, &q, K, Some(127.0));
        fixedbasis_roundtrip(&mut c, 1, K, 8, &p, &q, K, Some(1.0));
        let ea = rel_err(&a, &orig);
        let ec = rel_err(&c, &orig);
        assert!(ea < 0.05, "int8 on the coefficient block must be a small precision error, rel={ea}");
        assert!(ec > ea * 2.0, "int2 must be clearly worse than int8 ({ec} vs {ea})");
    }'''

raw = MODEL.read_bytes()
for name, old, new in (("mut take", OLD, NEW), ("quant test", OLD_T, NEW_T)):
    ob = old.replace("\n", EOL).encode()
    n = raw.count(ob)
    if n != 1:
        raise SystemExit(f"{name}: anchor matched {n} times (need 1)")
    raw = raw.replace(ob, new.replace("\n", EOL).encode(), 1)
MODEL.write_bytes(raw)
print(f"patched {MODEL} ({len(raw)} bytes)")
