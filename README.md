# moe-miss-substitution

Code, run scripts and result records for the paper

**Which Expert Misses Are Worth Waiting For? Substitution and Caching for Mixture-of-Experts Inference on a Consumer GPU**
ChaeWoo Son. Research Square preprint, 2026. https://doi.org/10.21203/rs.3.rs-11268552/v1

The paper studies what happens when a mixture-of-experts (MoE) model with hundreds of experts per layer
(mainly Qwen3-Next-80B-A3B) runs on a 24 GB consumer GPU with only a quarter of its experts in GPU memory and the
rest read from an NVMe SSD. Experts missing from GPU memory are either fetched synchronously or substituted with
experts already in memory, and the paper asks which substitutions account for the quality loss and whether the
substituted gate weight (mass) can guide cache policies.

This repository keeps the whole path of the project, including early explorations that the paper does not use,
in the order in which the runs were made.

## Layout

| Folder | Contents |
|---|---|
| `engine/` | The llama.cpp patch (expert cache, "xcache") and how to build it (`BUILD.md`) |
| `code/` | Every script used: routing traces, the PyTorch emulation, the router-logit simulator, pod run scripts, analysis and tests. Kept flat, as they were run |
| `code/ref_dsv4/` | DeepSeek-V4-Flash reference inference code used to check our emulation adapter |
| `results/` | Run logs and result files (JSON, logs, analysis outputs) |
| `figures/` | The paper's figures, the values they plot (`figdata.json`) and the script that drew them (it also reads research notes that are not included) |

## The path of the project

Runs are numbered R1-R21 in the paper. Inside the code, comments and log headers call them "Part n" or "round n"
(Part n = Rn). Internal names: `xc` / `xcache` = the engine's expert cache; E = fetch every miss (reference);
T = the 0.15 rule; C = the first-choice rule; F = frequency-based eviction; A = background admission (rent-or-buy);
Q = T+F+A (quality-first); S = C+F+A (speed-first); H = C with a 12.5% cache.

| Run | What we did | Main files | Results |
|---|---|---|---|
| R1 | Routing map of Qwen3-30B-A3B: how well LRU and optimal caches would hit | `trace_routes.py`, `analyze.py`, `report.py`, `prep_data.py`, `run.sh` | - |
| R2 | Routing map of Qwen3-Next-80B-A3B (linear attention, shared expert) | `trace_next.py`, `run_next.sh` | - |
| R3 | Predicting experts from a window of tokens, without training | `window_router.py`, `run_wr.sh` | - |
| R4 | Emulation: route only within the resident experts (substitution), Qwen3-Next | `masked_route.py`, `run_mr.sh`, `summarize_mr.py` | - |
| R5 | Emulation extended to gpt-oss-120b and DeepSeek-V4-Flash | `mr_run.py`, `ref_dsv4/`, `run_go.sh`, `run_go2.sh`, `run_ds2.sh`, `run_ds3.sh` | `dsv4_mr*.json`, `gptoss_mr.json` |
| R6 | PyTorch prototype engine on an RTX 3090 with NVMe reads | `live.py`, `run_live.sh`, `summarize_live.py` | `live_3090.*` |
| R7 | Expert cache inside llama.cpp, first measurements (A40) | `../engine/llama_xcache.patch`, `run_xc.sh`, `summarize_xc.py` | `xc_a40.*` |
| R8 | Engine improvements: fetch once, then only first choices | `run_xc2.sh`, `summarize_xc2.py` | `xc2_3090.*` |
| R9 | Time to first token | `run_xc3.sh`, `summarize_xc3.py` | `xc3_3090.*` |
| R10 | Prompt streaming: only the experts a prompt uses, from SSD to VRAM | `run_xc4.sh`, `summarize_xc4.py` | `xc4_3090.*` |
| R11 | Full answer time, 32 GB RAM, automatic choice of read path | `run_xc5.sh`, `summarize_xc5.py` | `xc5_3090.*` |
| R12 | Comparison with stock llama.cpp at the same VRAM (machine with slower reads) | `run_xc6.sh`, `summarize_xc6.py` | `xc6_3090.*` |
| R13 | Read-path decision and RAM conditions on one machine (main comparison) | `run_xc7.sh`, `summarize_xc7.py` | `xc7_3090.*` |
| R14 | Page cache filled only by our own process; worst-case conditions | `run_xc8.sh`, `summarize_xc8.py`, `evict.py` | `xc8_3090.*` |
| R15 | Cache policies as rules on recorded router logits (simulator) | `xdump.cpp`, `xsim.cpp`, `make_seq.py`, `run_xc9.sh`, `run_xc9b.sh`, `summarize_xc9*.py` | `xc9_sim.log`, `xc9b_sim.log` |
| R16 | Rent-or-buy background admission and per-layer slots (simulator) | `run_xc10.sh`, `summarize_xc10.py` | `xc10_sim.log` |
| R17 | Engine: admission, frequency scores, per-layer slots (6 sessions) | `run_xc11*.sh`, `ci_xc11*.py`, `summarize_xc11.py`, `test_trace.py` | `xc11_3090.log` |
| R18 | Engine: reproduction on 24 sessions and new topic-change sequences | `run_xc12.sh`, `ci_xc12.py`, `merge_runs.py` | `xc12_3090.log` |
| R19 | Engine: 24 topic-change sequences | `run_xc13.sh`, `ci_xc13.py`, `make_seq.py` | `xc13_3090.log` |
| - | Post-hoc analysis of mass and perplexity in R18 and R19 | `posthoc_mass.py` | `posthoc_mass_2026-10-04.out` |
| R20 | Greedy generation on GSM8K and HumanEval, with hypotheses and decision rules fixed before the runs | `prep_tasks.py`, `run_xc14.sh`, `score_tasks.py`, `collect14.py` | `xc14_*`, `xc14_pooled_tasks/` |
| R20s | HumanEval regeneration after an extraction defect was found | `run_xc14he.sh`, `dump14.py`, `undump14.py`, `score_fix.py`, `verify14he.py`, `explore14he.py` | `xc14he_*` |
| R21 | Experiment B: equal substituted mass, different composition (emulation) | `expb.py`, `analyze_b.py`, `test_expb.py`, `run_expb_prefetch.sh`, `run_expb.sh`, `run_expb_aux.sh` | `expb_*` |

The paper uses R4, R5, R7 and R12-R21; the other runs were earlier explorations. Results that the paper reports as
exploratory, chosen from the results, or planned in advance are marked that way in the paper.

## Running

- The experiments ran on rented cloud GPUs (RTX 3090 and A40 pods). Each `run_*.sh` is the script a pod ran: it
  downloads the model and data from Hugging Face, builds what it needs, runs, and prints its results into the log.
  Paths such as `/workspace` refer to the pod's volume.
- `test_*.py` are local checks on tiny random models (for example, that our layer-wise forward pass matches
  transformers). They write temporary models under `/tmp/moe_test/`.
- Python 3 with PyTorch and transformers (4.57.6 for R21); the emulation of Qwen3-Next reuses transformers modules.
- Dataset and model revisions were not pinned in most runs (see Appendix C of the paper); R20s and R21 record them.

## Data and models

Sessions are built by `prep_data.py` from public datasets: Wikipedia (Korean and English, wikimedia/wikipedia
20231101), codeparrot/codeparrot-clean-valid, HuggingFaceH4/ultrachat_200k, maywell/koVast and AI-MO/NuminaMath-CoT.
The generation test uses GSM8K and HumanEval. Models: Qwen3-Next-80B-A3B-Instruct (BF16, and the Q4_K_M GGUF by
unsloth), DeepSeek-V4-Flash and gpt-oss-120b.

## Notes

- The code and the run scripts were written and maintained with Claude (Anthropic), as described in the paper's
  section on the use of an AI system.
- Some comments refer to planning documents and research notes that are not part of this repository.
- License: MIT (see `LICENSE`). The engine patch applies to llama.cpp, which is also MIT-licensed.

## Citation

```bibtex
@misc{son2026expertmisses,
  author       = {Son, ChaeWoo},
  title        = {Which Expert Misses Are Worth Waiting For? Substitution and Caching for Mixture-of-Experts Inference on a Consumer GPU},
  year         = {2026},
  howpublished = {Research Square preprint},
  doi          = {10.21203/rs.3.rs-11268552/v1},
  url          = {https://doi.org/10.21203/rs.3.rs-11268552/v1}
}
```
