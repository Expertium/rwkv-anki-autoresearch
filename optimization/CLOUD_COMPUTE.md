# Renting GPUs for the chain -- what our workload needs, what it costs, which card (2026-09-17)

> Andrew 2026-09-17: *"browse some cloud computing renting websites. Preferably Russian, but if you can find
> cheap compute elsewhere, that probably works too, I have a card for international payments. As for which
> GPU offers the best price/performance ratio for our case, I'll leave that to you (probably not H200,
> lol...unless?)"* Prices below were read on 2026-09-17 and change hourly on the marketplaces; the RUB/USD
> rate used is ~90.

## 1. What our workload actually is (so the card choice follows from it)

* A WS step on the 4070 is ~1.1 s: **~80% GPU kernel time in small ELEMENTWISE kernels** (the d=80
  interleaved trunk: 892 ms/step of kernels after PermGather), plus **18,756 kernel launches costing ~218 ms
  of CPU dispatch**. FLOPS are irrelevant -- the matmuls are ~5% (2026-07-03 profile). So a step is bound by
  GPU MEMORY BANDWIDTH and by launch latency, not by tensor cores.
* VRAM: 9.4 GB peak at MAX=65536 (12 GB card). Any card with >= 16 GB is fine; more VRAM buys nothing unless
  MAX goes up, which changes the recipe (a phase-5 lever, not a speedup).
* RAM: trainer ~10.6 GB + 2 fetch workers at ~3.2 GB private + a ~9 GB LMDB mapping each -> 32 GB is enough
  for training. **A full-VAL eval needs ~64 GB** (giant users hold ~40 GB) -- so evals stay LOCAL: a decayed
  checkpoint is 2.4 MB, download it and score it here.
* Disk on the remote box: train db 159 GB real (sparse map; measure with GetCompressedFileSize), the -id
  dataset 16 GB, the label filter 3.9 GB, a 10-user validation db (minutes to build) -> **~200 GB** if evals
  stay local; ~360 GB if the test db (161 GB) is also rebuilt there.
* Software: Linux + torch cu130 + the CUDA kernel rebuilt with nvcc for the card (sm_120 for a 5090); the
  kernel was born on Linux (srs-benchmark's 3090). **The runners are `.cmd` files**: each remote experiment
  needs a bash runner with the same env block and guards (~a day of work for the first one, then a template).

### The speed estimate per card (kernel part scales with bandwidth, the 218 ms launch part does not)

| card | memory bandwidth | est. step vs 4070 | why |
|---|---|---|---|
| RTX 4070 (ours) | 504 GB/s | 1.0x | reference |
| RTX 4090 | 1,008 GB/s | ~1.7x | 892/2 + 218 |
| RTX 5090 | 1,792 GB/s | ~2.1x | 892/3.5 + 218 |
| A100 80GB | ~2,000 GB/s | ~2.2x | + a slower host CPU on most clouds |
| H100 SXM | 3,350 GB/s | ~2.6x | launch-bound floor |
| H200 | 4,800 GB/s | ~2.8x | launch-bound floor |

**These are estimates. A 30-minute benchmark on a rented card (`RWKV_BENCH` mode, ~$0.30) settles the real
number before anything is committed.** The launch floor means an H200 cannot be more than ~3x a 4070 for us.

## 2. Prices (2026-09-17)

### International marketplaces (international card)

| provider | RTX 4090 | RTX 5090 | A100 80GB | H100 | H200 | notes |
|---|---|---|---|---|---|---|
| Vast.ai | ~$0.35-0.40/h | ~$0.40-0.65/h | ~$0.8-1.2/h | ~$1.5-2.5/h | ~$2.5-3.5/h | supply/demand pricing; host RAM/disk/CPU vary per listing (filter for >= 48 GB RAM, >= 250 GB disk, a fast CPU); interruptible bids are cheaper but can be killed |
| RunPod community / secure | $0.34 / $0.74 | $0.69 / $0.99 | $1.19 / $1.59 | $1.99 / $2.89 | $3.59 / $4.59 | per-second; 5090 pods come with 9 vCPU + 35 GB RAM, 4090 with 6 vCPU + 41 GB; network volume ~$0.05-0.07/GB/month |
| 20-provider median (getdeploying) | -- | $0.65/h | -- | -- | -- | cheapest listed: HyperAI $0.35, Vast $0.40, GPU.ai $0.41, Salad $0.50 |

### Russian providers (Russian card, SBP, rubles; prices include VAT where stated)

| provider | RTX 4090 | RTX 5090 | A100 | H100 / H200 | notes |
|---|---|---|---|---|---|
| immers.cloud | ~83-87 RUB/h | **~131 RUB/h** (8 vCPU / 32 GB / 60 GB); 64 GB RAM ~140 RUB/h | ~212-223 RUB/h | H200 ~423 RUB/h | per-second billing, deposit from 100 RUB, stop/start keeps the disk; **disk tops out at 160 GB in the standard configs** (ask about an extra volume); 25% off for 30-day prepay, 35% for 60 |
| Intelion Cloud | 4090 48 GB from 75 RUB/h | -- | -- | H100 from 352 RUB/h | per-second, Russian cards, prices incl. 22% VAT; 8-25% prepay discounts |
| Selectel | ~57,000 RUB/month (~78 RUB/h) | -- | yes (cloud) | -- | hourly or monthly; enterprise-shaped |
| HOSTKEY | ~29,000 RUB/month | ~48,000 RUB/month | -- | -- | dedicated, mostly monthly; Russian cards |
| Timeweb Cloud | T4 only; A100/H100/H200 by pre-order | -- | pre-order | pre-order | not useful for us |

Russian hourly prices are **~2-3x the international marketplaces** for the same card; the trade is payment
in rubles, a Russian card, and a datacenter next door.

## 3. Price per unit of OUR work (price / speed factor = cost of one 4070-hour of training)

| card | Vast.ai | RunPod community | immers.cloud |
|---|---|---|---|
| RTX 4090 | ~$0.21 | $0.20 | ~49 RUB |
| **RTX 5090** | **~$0.24** | $0.33 | **~62 RUB** |
| A100 80GB | ~$0.45 | $0.54 | ~98 RUB |
| H100 | ~$0.75 | $0.77 | -- |
| H200 | ~$1.1 | $1.28 | ~150 RUB |

**Verdict: RTX 5090.** Per unit of work it ties the 4090 (the 4090 is a hair cheaper), and it halves the
wall-clock, which is what we are short of. A100/H100/H200 cost 2-5x more per unit of our work because our
step has a launch-latency floor that bandwidth cannot buy down. "Unless?" -- no: the H200's 4.8 TB/s would
only matter if the step were purely bandwidth-bound, and 218 ms of it is CPU.

## 4. What the queue would cost on one rented 5090 (evals local)

| job | 4070-hours | 5090 hours (~2.1x) | Vast.ai (~$0.55/h) | immers (~131 RUB/h) |
|---|---|---|---|---|
| setup: kernel build + train-db rebuild (~5 h CPU, GPU idle) | -- | ~6 h | ~$3 | ~800 RUB |
| the warmup experiment `lfwu20k` (10-ep WS + 1-ep decay; eval local) | ~42 | ~20 h | ~$11 | ~2,600 RUB |
| its control on the SAME hardware (a w10lf/lfd1 twin, see 5) | ~42 | ~20 h | ~$11 | ~2,600 RUB |
| link D's d=128 at 10+2 (`d128e10`, ~4 days here) | ~90 | ~43 h | ~$24 | ~5,600 RUB |
| the three QAT arms' decays (lfq1 / lfqkd1 / b13, ~16 h each at L*=1) | ~48 | ~23 h | ~$13 | ~3,000 RUB |
| **whole queue** | ~220 | **~4.5 days** | **~$60-70** | **~15,000 RUB** |

A month of a 5090 at immers with the 30-day prepay is ~70,000 RUB; on Vast.ai ~$400. Neither is needed --
the queue is ~5 days of one card, or ~2 days of three cards.

## 5. Three caveats that decide whether the numbers mean anything

1. **Numerics differ across GPUs.** bf16 reductions and cuBLAS algorithm choice change with the card, and
   every gate here is a PAIRED comparison. A candidate trained remotely against a control trained locally is
   confounded at roughly the noise floor (~1e-4). **Run the control on the same box** (the twin of w10lf /
   lfd1, ~$11), or accept that only remote-vs-remote pairs are clean.
2. **Reproducibility is per box.** Our RWKV_DETERMINISTIC=1 gives run-to-run variance ~0 on ONE machine; a
   Vast.ai listing is a different machine every time. Keep a rented instance alive (stop/start on immers
   keeps the disk) for a whole pair.
3. **The upload.** ~20 GB up (dataset + label filter + repo); at 50 Mbit/s that is ~1 h, at 10 Mbit/s ~5 h.
   Measure the uplink before choosing between "upload the dataset and rebuild remotely" (~5 h CPU) and
   "upload the 159 GB train db" (only sane above ~200 Mbit/s).

## 5b. The cheapest RTX 5090 / RTX 4090 anywhere (Andrew 2026-09-17: "Look for the cheapest RTX 5090 (anywhere) ... or 4090, if it's cheaper"; read 2026-09-17 ~10:15)

Live listings (getdeploying.com's stock tracker, dated 2026-09-17; the marketplaces' own pages are JavaScript and
the vast.ai domain is blocked in the browser pane). Only offers with >= 32 GB host RAM are usable for a WS;
Salad (containers, 8 GB RAM, no disk) is not.

| RTX 5090 | $/h | host | RTX 4090 | $/h | host |
|---|---|---|---|---|---|
| **Vast.ai SPOT (interruptible)** | **from $0.12** | varies; can be killed any time | **Vast.ai SPOT** | **from $0.18** | varies |
| HyperAI | $0.35 | 16 vCPU / 40 GB | Lium (Bittensor marketplace, per-minute, prepaid) | $0.30 | 24 vCPU / 55 GB |
| Vast.ai reserved 3 mo | $0.39 | -- | Novita | $0.33 | 16 vCPU / 62 GB |
| Vast.ai on-demand | $0.43 | 32 vCPU / 63 GB | RunPod community | $0.34 | 6 vCPU / 41 GB |
| GPU.ai (aggregator, per-second) | $0.43 (2x) / $0.54 (1x) | not stated | GPU.ai | $0.34 | not stated |
| GPUhub | $0.46 | -- | Vast.ai on-demand | $0.36 | 24 vCPU / 31 GB |
| Lium | $0.50 | 24 vCPU / 63 GB | Vast.ai reserved | $0.39 | -- |
| RunPod community / secure | $0.69 / $0.99 | 9 vCPU / 35 GB | RunPod secure | $0.74 | -- |
| Database Mart dedicated (monthly) | $479/mo = $0.66 | 36 cores / 256 GB / 10 TB | Database Mart | $0.51 | -- |
| immers.cloud (RU) | ~131 RUB = $1.46 | 8 vCPU / 32 GB | immers.cloud | ~83-87 RUB = $0.93 | -- |

**Per unit of OUR work (price / est. speed factor 2.1x for the 5090, 1.7x for the 4090), on-demand:**
5090 at $0.35-0.43 -> **$0.17-0.20 per 4070-hour**; 4090 at $0.30-0.36 -> **$0.18-0.21**. A tie. The 5090
finishes in half the wall-clock, so it wins on-demand. **Spot is the real discount:** Vast.ai spot 5090 from
$0.12/h = ~$0.06 per 4070-hour, i.e. the whole ~220 4070-hour queue for ~$15 -- IF the WS survives being
killed. Our WS loop already resumes from its 1000-step checkpoints (make_resume + RWKV_RESUME_SKIP_GROUPS),
so an interrupted WS costs <= 1000 steps + a restart; a spot box needs a supervisor that relaunches the runner
on boot and a persistent volume (Vast.ai bills storage while stopped, ~$0.10-0.20/GB/mo). The decay and the
eval are short enough to run on-demand. ⚠ A spot instance that is killed and re-won lands on a DIFFERENT host
= different numerics mid-run (caveat 1 applies twice); for a gate-critical pair, on-demand on one box.

## 5c. DECIDED (Andrew 2026-09-17 ~12:40): Vast.ai, INTERRUPTIBLE, "Unverified Machines" ON, $100 at the end of Friday 2026-09-18

> **CORRECTION (Andrew 2026-09-17 ~13:10): "Interruptible doesn't seem to be cheaper than On-Demand. It's 'unverified' that
> makes a difference."** So rent **ON-DEMAND with "Unverified Machines" ticked** -- same ~$0.26/h, no interruptions, one host
> for a whole pair (which also removes the mid-run numerics caveat). The supervisor stays as a crash/reboot net only.
> Also his instruction the same hour: NO more prep for now (weekly usage at ~98%); the kit in `scratchpad/cloud/` is enough.

* **✅ Tick "Unverified Machines" in the Vast.ai filter panel.** The default hides unverified hosts, and every
  listing under $0.35/h is one. Also switch the "On-Demand" dropdown to "Interruptible" (bids). Seen live at
  12:40 with both set: a 5090 at **$0.257/h** (host 125418, CN: EPYC 7K62, 24 of 192 vCPU, 36 of 290 GB RAM,
  773 GB disk, 99 Mbps up / 66 Mbps down, reliability 98.5%, max duration 6 months) plus two Slovak hosts at
  $0.355-0.369 (8 vCPU / 31 GB RAM -- RAM too tight for a WS with 2 fetch workers) and the US host at $0.356
  (10 Mbps up). My earlier "$0.12 spot" came from a stale tracker and is withdrawn; ~$0.26 is the live floor.
* **Budget: $100, bought when Andrew's weekly Claude Code limit resets (end of Friday 2026-09-18).** At $0.26/h
  that is ~385 GPU-hours = **~16 days of one 5090**, i.e. the whole remaining queue (~4.5 5090-days) three
  times over, plus bandwidth (metered per GB on Vast.ai; the CN host's 99 Mbps up means the 20 GB upload takes
  ~30 min; check the "plus bandwidth" price on the listing).
* **What must be ready by Friday (my side, CPU, while lfq1 runs):** the Linux setup script (venv, torch cu130,
  the CUDA kernel via nvcc for sm_120), the remote rebuild of the train db + a 10-user validation db from the
  uploaded -id dataset + label filter, a bash runner that mirrors `run_lfwu20k.cmd`'s env and guards, and a
  supervisor that relaunches the runner after an interruption (the WS resumes from its 1000-step checkpoints).
  First pair on the box: `lfwu20k` and a twin of lfd1 (its control on the SAME hardware). Evals stay local.
* **36 GB RAM on the $0.257 host is enough for a WS** (trainer ~10.6 GB + 2 workers at ~3.2 GB private; the
  LMDB mapping is file-backed) but NOT for a full-VAL eval -- which stays local by design.

## 5d. ALLOWED (Andrew 2026-09-17 ~13:40): on the 5090 "it's ok to change MAX or any other batch-related settings to utilize VRAM better"

* 32 GB VRAM fits MAX_TRAIN_GLOBAL_LEN well above 65536 (9.4 GB there). Measured on the 4070: MAX 98304 = +14.3% rev/s but -34% optimizer
  steps per epoch, so **a MAX change is a RECIPE change, not a free speedup**: the MAX=65536 move cost ~0.0003 both modes until the LR was re-tuned
  (iter 34), and both LRs (AdamW PEAK_LR and RWKV_MUON_LR) must move together.
* Rule for the box: FIRST pair at MAX 65536 (comparable with every local run); a larger-MAX arm only as its own pair, with its control at the
  SAME MAX on the same box, and the LR scaled (start from sqrt scaling, confirm on the validation trace). d=128 (link D) is where it pays most:
  it is stuck at MAX 32768 on 12 GB, and 65536 on the 5090 removes the pxe PREREG's one named confound (batch vs width).

**THE BUDGET RULE FOR THAT (Andrew, same hour): "fine to adapt the recipe to 5090 and tweak hyperparameters, as long as tweaking doesn't take
longer than whatever we stand to gain from 12->32 GB of VRAM."** So: (1) the 15-minute benchmark measures rev/s at MAX 65536 / 98304 / 131072 /
196608 FIRST (free: a few hundred steps each; the launch floor of ~218 ms/step is amortised by a bigger batch, so the 5090's gain can exceed the
4070's +14%); (2) tune only if the best MAX is >= ~25% faster -- a re-tune is ~4 one-epoch runs (~2 h each on the box = ~8 h), which a 25% gain
repays after ~2 full WS runs (each ~20 h); below that, stay at 65536 and spend the hours on experiments; (3) the re-tune moves both LRs together
and is judged on the validation trace, like phase 5.

## 5e. 4090 vs 5090 at the live unverified prices (Andrew 2026-09-17 ~14:00: a 4090 at $0.14/h vs a 5090 at $0.272/h)

* Per unit of our work (est. 1.7x / 2.1x a 4070): 4090 $0.14/1.7 = **$0.08**, 5090 $0.272/2.1 = **$0.13** -- at half the rate the 4090 WINS by ~37%,
  and the queue is many INDEPENDENT runs, so two 4090s in parallel ($0.28/h) beat one 5090 on wall-clock too.
* **The HOST matters as much as the card** (our step has a ~218 ms CPU launch floor): a Ryzen 9 9950X host is ~1.5x a 5950X single-thread and
  far ahead of an EPYC 7282, so a 4090 on a 9950X may match a 5090 on a slow EPYC. The benchmark decides, not the card name.
* **Listing filters that are hard requirements:** host RAM >= 32 GB (the Texas $0.14 hosts have 16 GB: trainer 10.6 + 2 workers x 3.2 GB private
  does not fit, and the rebuild needs more), **Max CUDA >= 13.0** (our wheel is torch cu130; the JP 5090 host shows 12.8 = driver too old),
  disk >= 250 GB, upload >= 50 Mbps (14 Mbps = 3+ h for the 20 GB), reliability >= ~95% (87% = expect a dead run), max duration >= the queue.

## 6. Recommended first step, if Andrew wants to try it

Rent ONE RTX 5090 on immers.cloud (rubles, per-second, 64 GB RAM config, ask for a 250 GB volume) or on
Vast.ai (cheaper, international card): build the kernel, run the 30-minute benchmark, decide from the
measured steps/s. Then the warmup experiment + its control as the first pair (~$25 / ~5,500 RUB). The bash
runner port is the work item on my side; nothing is started without Andrew's go.

Sources (read 2026-09-17): immers.cloud 5090 and 4090 pages, RunPod pricing, getdeploying's 5090
comparison, tproger's 2026 Russian GPU-cloud roundup, ailist.ru's 2026 provider list, gpuradar.ru's hourly
guide, the Vast.ai 5090 page. Links in the chat reply of 2026-09-17.

## 5f. MEASURED on the rented box (2026-09-18/19): RTX 4090 + Ryzen 9 7950X = 2.08x the PC

`scratchpad/cloud/bench.sh` (cw10lf's exact env and WS toml, 150 s warm-up, 180 s measured, test-db build finished before the numbers that count):

| MAX | steps/s | rows/s-equivalent | VRAM | vs MAX 65536 |
|---|---|---|---|---|
| 65536 | **1.900** (PC: 0.913 = **2.08x**) | 124,516 | 7.9 GB | -- |
| 98304 | 1.422 | 139,807 | 6.8 GB | +12% |
| 131072 | 1.144 | 150,001 | 14.6 GB | +20% |
| 196608 | 0.811 | 159,467 | 21.3 GB | +28% |

Params 561,732 and 10,935 groups on the box-built db = the PC's numbers exactly. **Decision by the 5d budget rule: MAX stays 65536.** Only 196608 clears the 25% bar, at a third of the optimizer steps and 21.3 of 24 GB; re-tuning both LRs for it costs more box-hours than +28% returns inside a $100 budget, and it would break comparability with every local run. The 2.08x is above the memo's 1.7x estimate for a 4090 -- the 7950X's single-thread speed pays on the launch-bound share. A 10-epoch WS is ~16 h (PC 33 h); db builds: train 57 min, VAL-half test 6 min (PC ~4 h). First contact fixes: CRLF in the archived `.sh` files, `wandb` missing from the pip list, a `grep -c | grep` guard that fails under pipefail, `import rwkv.model.RWKV_CUDA` needs `import torch` first.

**Two runs at once on the 4090 (Andrew's idea, measured 2026-09-19):** solo cw10lf 1.87 steps/s (util ~70%, 150 W of 450 W); cw10lf + lfwu20k side by side 1.16 + 1.16 = **2.32 steps/s, +24% total** (util 100%, 183 W, 15.8 GB VRAM, RAM fine). The processes time-slice the GPU (no MPS in the container), so the gain is only the idle launch gaps. Kept: the pair ends in ~26 h instead of ~33 h, and both arms of the pair see the same co-tenancy. The runners take a per-tag lock and an eval waits for any other eval (a giant user holds ~40 GB of RAM).


## 5g. MEASURED on the rented box (2026-09-20): the numbers TRANSFER, but the EVALS do not, unchanged

Three findings from running a full 10+1 pair (`cw10lf`, `lfwu20k`) end to end on the 4090. Each one is a property of
the machine class, not of these runs, so carry them to any future rental.

**1. A box number can be compared with a PC number directly.** `cw10lf` is w10lf's recipe from scratch on the box;
the local `lfd1` is the same recipe on the 4070. Paired over 2,487 users: **-0.000023 ahead / -0.000021 imm**
(p 0.73 / 0.90) -- a third of the 7.5e-5 noise floor, and an upper bound, since cw10lf lost 20 WS steps to a
co-tenant OOM that lfd1 never saw. The old rule "controls must run on the same box" can be relaxed to "prefer the
same box"; a cross-hardware comparison costs ~2e-5, not the ~0.0004 cross-seed spread.

**2. ⚠ THE BOX CANNOT EVALUATE THE GIANT USERS, AND WINDOWS IS THE REASON IT LOOKS EASY LOCALLY.** A 1,048,576-token
mega-chunk needs ~22 GB, and the 4090 has 23.5 -- but the 12 GB 4070 runs the same users because **Windows spills GPU
memory into the WDDM shared pool** (which is also why a local giant-user eval wants ~40 GB of free system RAM).
**Linux has no such spill, so 24 GB is a hard wall.** 12 of the 2,499 VAL users hit it: 5626, 5859, 5991, 6198, 6431,
6630, 6714, 6720, 6810, 7033, 7221, 7238. `get_result` now skips such a user WHOLE (like the NaN guard; partial
batches would change its equalized `size`) into `result/RWKV-<tag>.oomskip.jsonl`, `eval_sharded` counts those as
accounted for, and the runners' user-count guard adds them back. **Budget n = 2,487 for a box eval and pair on the
intersection.** A bigger card (48-80 GB) would take all 2,499; the alternative is to evaluate those 12 on the PC.

**3. ⚠ `ulimit -n` 1024 kills a long eval, silently.** `torch.multiprocessing` passes every batch tensor to the fetch
workers as a shared-memory FILE DESCRIPTOR. On the container's default soft limit a worker dies with "Too many open
files" after ~45 users, and the parent then waits FOREVER on an empty queue -- GPU idle, log frozen, no marker (it
cost 1 h 22 min before anyone looked). Every generated runner now starts with `ulimit -n 65535`. Windows never shows
this, so no local run could have caught it.
