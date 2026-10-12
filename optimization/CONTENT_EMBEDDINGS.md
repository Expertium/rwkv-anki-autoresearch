# Content-aware RWKV — the embedding decision (contingent)

**Status: CONTINGENT.** Andrew, 2026-08-19: *"there is a slim possibility that we'll get a dataset
with card content"*. Nothing here is scheduled. This exists so the decision does not have to be
re-derived if that dataset appears.

## ★ THE DECISION

> **Andrew, 2026-08-19: "paraphrase-multilingual-MiniLM-L12 quantized to int8 (no pruning)."**

* model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
* quantization: **int8**, weights ≈ **118 MB**
* **no vocabulary pruning** — the full 250k-token table ships
* output: **384 dims**, native (see "reduction" below — do not naively truncate)
* licence: Apache-2.0, so shipping it inside Anki (AGPL) is clean

### Why multilingual, and why it outranked the bias concern

Andrew ruled out training our own embedder: a content dataset would come largely via AnkiHub and be
**heavily skewed toward med students**, so a model trained on a general corpus generalizes better.
Correct — and the same argument kills a *learned* dimensionality reduction, which would be fitted on
the same skewed data. Any reduction must be **data-independent**.

The larger risk is **language coverage**, which is specific to Anki: language learning is a flagship
use case, and a large share of real collections are Japanese / Chinese / Spanish vocabulary.
`all-MiniLM-L6-v2` is English-only and does not degrade gracefully on those — it produces near-noise
rather than a worse-but-usable vector. That outweighs the med skew.

### Why no pruning, despite the 5x size

| model | vocab table | encoder | int8 total |
|---|---|---|---|
| all-MiniLM-L6 (English) | 11.7M (52%) | 11.0M | 23 MB |
| **paraphrase-multilingual-MiniLM-L12** | **96.0M (82%)** | 21.7M | **118 MB** |

82% of the multilingual model is the vocabulary table, and pruning to the tokens a collection
actually uses would bring it to ~29 MB int8 at a 20k working vocab — i.e. English-MiniLM cost.

**That option was considered and declined.** Pruning buys ~90 MB and introduces a whole failure
mode: a per-user vocabulary that must be re-derived whenever someone adds a language, with silent
degradation (missing tokens) if the re-derivation is missed or deferred. 118 MB flat has no such
state. The trade is 90 MB against removing a class of silent failure, on a device tier where the
model is loaded occasionally rather than held resident.

## Why RAM is not the binding constraint

**The embedding model does not need to be resident during scheduling.** Card text changes rarely, so
a note is embedded once on create/edit, and the scheduler only ever reads a stored vector.

Andrew's constraint (2026-08-19) is that users edit cards **on mobile**, so the model must be
**available on-device at all times** rather than only at sync — which is what rules out the
Qwen3-Embedding class:

| candidate | int8 weights | verdict |
|---|---|---|
| Qwen3-Embedding-4B | ~4000 MB | out |
| Qwen3-Embedding-0.6B | ~600 MB (int4 ~300) | out on 2016-tier: 1-2 GB total RAM, and per-app heap growth limits were commonly 128-192 MB |
| **multilingual MiniLM-L12** | **118 MB** | **chosen** |
| model2vec / static embeddings | 4-15 MB | the floor, if 118 MB ever proves too much |

Storage of the vectors themselves is a non-issue: per **note** (siblings share content), 20k notes
at 384-dim int8 is 7.7 MB.

## ★ What it costs OUR model, which is the real constraint

The trunk is 558,212 params. Ingesting the vector widens `features2card`:

| input dim | added params | share of the model |
|---|---|---|
| 384 (native) | 30,720 | **+5.5%** |
| 128 | 10,240 | +1.8% |
| 64 | 5,120 | +0.9% |

+5.5% for the native 384 is affordable. **If it must be cheaper, use a FIXED RANDOM PROJECTION** —
data-independent, so it survives the anti-skew argument. Do **not** use PCA or a learned head (fitted
on skewed data), and do **not** naively slice the 384 dims: MiniLM is **not** Matryoshka-trained, so
its dimensions are not ordered by importance. (Qwen3-Embedding *is* MRL-trained and could be
truncated — but it is out on size.)

**It does not touch the deploy contract.** An embedding is an *input*, not state, so the frozen
9 B/card and 27 B/note state budgets are unaffected.

## Open design issues — settle these before building, not after

1. **Cards are not sentences.** Many are one or two words. A sentence embedder's behaviour on 1-3
   token inputs is not the behaviour it was trained for.
2. **Markup.** Cloze cards carry `{{c1::...}}` and fields carry HTML. Both need stripping — and the
   cloze structure is itself signal that stripping discards.
3. **Content-free cards.** Image-only and audio-only cards have no text. This needs an explicit
   "no content" encoding, and it must be a value a real embedding cannot produce. ⚠ Two sentinel
   bugs were fixed on 2026-08-19 alone (the cumulative-elapsed sentinel summed as a magnitude, and a
   rounding path that could mint a fake sentinel), so treat sentinel design here as a first-class
   decision rather than a default.
4. **Per note or per card?** Content is a note property; siblings share it. Storing per note is
   ~2x cheaper and matches the note stream the model already has.
5. **Re-embedding on edit** must invalidate the cached vector, and the scheduler must tolerate a
   stale-or-missing vector without failing.

⚠ The parameter counts and vocab size above are from memory of the model cards; confirm them against
the published cards before designing to exact numbers. The *shape* of the argument — that the vocab
table dominates a multilingual model, and that the input dim is what costs our trunk — does not
depend on those exact figures.

## HOW MUCH CAN CARD CONTENT BUY? AnkiHub's 4B-review comparison (Andrew 2026-09-19)

Source: https://community.ankihub.net/t/comparing-memory-models-on-a-4b-reviews-dataset/606464 (95.8 M post-cutoff validation reviews, success rate 80.3%). **The one clean pair: `memory_transformer_v1` (random identity vectors per card = content-free, full cross-card history) vs `v2` (the same model with frozen Qwen3 Q+A embeddings of the card version shown): log-loss 0.392 -> 0.335 (-0.057, -14.5%), AUC 0.804 -> 0.866, Brier 0.124 -> 0.106.** v4 (adds a trainable per-card-version delta table) 0.332 and v5 (2048-wide 6-layer delta-rule backbone, no delta table) 0.329: once the text is in, a learned per-card term adds ~0.003 and 10x+ the backbone adds ~0.006. FSRS-6 per user 0.458, so v1 sits at 86% of FSRS-6's loss -- the same ratio as our RWKV ahead vs FSRS-6 in srs-benchmark, i.e. v1 is a fair stand-in for "a strong content-free sequence model".

**Extrapolation, with Andrew's caveat that their methodology differs a lot from srs-benchmark** (log-loss averaged over REVIEWS, not by user; a TEMPORAL split with the same users on both sides; 80% success rate vs ~85-90% here, so compare RELATIVE changes only): -14.5% applied to our leak-free 0.2978 / 0.2642 would be about -0.043 / -0.038. **Discount it:** AnkiHub users study the SAME shared decks, so a text embedding there also works as a cross-user card identity (the crowd's difficulty for this exact card), and v4's near-zero gain says the frozen text already carries that. On personal decks only the text-difficulty part transfers. A third to a half of the effect = **-0.015 to -0.03**, which is still 5-10x everything the algorithmic loop found, and the same order as the whole gap between FSRS-6 and RWKV. It stays contingent: no dataset with card content and the 10k users' structure exists.

