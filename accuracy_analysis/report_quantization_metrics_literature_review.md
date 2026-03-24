# Literature review: are layer-output norm and perplexity enough for quantization evaluation?

**Date:** 2026-03-22  
**Scope:** recent LLM quantization literature, with emphasis on FP4/NVFP4-era work and evaluation methodology  
**Review type:** targeted narrative/scoping review  
**Status:** links manually checked from source pages; not DOI-verified with a citation script

**Figure status:** AI-generated schematic requested by the literature-review workflow was not generated in this environment because `OPENROUTER_API_KEY` is not set for the bundled `scientific-schematics` tool.

---

## Executive summary

Short answer: **no, layer output norm + perplexity are not enough as the full evaluation stack**.

- **Layer output norm / hidden-state L2** is useful as a **cheap diagnostic / sensitivity proxy** for ranking layers or experts.
- **Perplexity** is useful as a **model-level language modeling proxy**, and several papers explicitly show it often correlates with downstream quality.
- But the recent literature, especially stronger and more recent LLM quantization work, almost always evaluates **more than perplexity**:
  - **zero-shot / few-shot downstream tasks**
  - **instruction / reasoning / coding benchmarks**
  - **task-level accuracy recovery relative to BF16/FP8 baselines**
  - **system metrics** such as throughput, latency, memory, and sometimes energy

For your use case, the literature supports this recommendation:

1. Keep **layer output norm** for fast layer/expert screening.  
2. Keep **perplexity** for model-level sanity checking and coarse ranking.  
3. Add a **downstream benchmark suite** before concluding a quantization recipe is “good”.  
4. For NVFP4 specifically, add **hardware-facing metrics** too, because many NVFP4 papers and practical reports argue value through the **accuracy–throughput trade-off**, not accuracy alone.

---

## What the current repo measures

From `accuracy_analysis/sensitivity.py`, the current analysis already uses a small hierarchy of metrics:

1. **`layer_out_norm`**  
   single-layer local sensitivity via output difference norm.

2. **`layerwise_model_out_norm`**  
   quantize one layer at a time, run the full model, and measure the last hidden-state L2 difference.

3. **`layerwise_perplexity`**  
   quantize one layer at a time and measure perplexity increase versus BF16.

4. **`quantize_all`**  
   quantize all experts and report **perplexity + mean KL divergence**.

So the current setup is actually:

- **local proxy:** output norm  
- **model proxy:** perplexity  
- **distributional proxy:** KL divergence (full-model path)

What is still missing relative to the literature is mainly:

- **downstream tasks**
- **instruction / reasoning / coding evaluation**
- **hardware efficiency metrics**

---

## Main conclusion from the literature

### 1) Output norm is useful, but mostly as an internal proxy

I did **not** find strong recent quantization papers that treat plain layer-output L2 as the primary final evaluation metric for LLM quantization. It is much more common as:

- an internal proxy for sensitivity analysis
- an auxiliary objective or calibration signal
- an error-analysis quantity alongside MSE / Hessian-weighted error / activation outlier statistics

That means your current use of layer output norm is reasonable for **screening layers/expert blocks**, but literature does **not** support using it as the only evidence that a quantized model is good.

### 2) Perplexity is standard, but not sufficient alone

Perplexity is still one of the most common quantization evaluation metrics, especially for:

- WikiText-2
- C4
- PTB / language-model validation sets

However, recent papers increasingly say one of two things:

- **Perplexity is a useful proxy on many tasks**, especially for base models.
- **Perplexity alone misses important behavior**, especially for instruction-tuned models, reasoning, coding, alignment, and task robustness.

### 3) The stronger recent papers evaluate downstream tasks explicitly

The most convincing recent quantization papers usually report some combination of:

- **perplexity / validation loss**
- **zero-shot or few-shot accuracy**
- **instruction or reasoning tasks**
- **coding benchmarks**
- **latency / throughput / memory**

This is especially true for **NVFP4-era work**, because the point of native FP4 Tensor Core support is not just “acceptable perplexity”, but **better deployment/training efficiency at similar task quality**.

---

## Recommended metric stack for your project

### A. For fast sensitivity ranking

Keep:

- layer output norm
- layerwise model-output norm
- layerwise perplexity

Use them to rank:

- sensitive layers
- sensitive expert blocks
- candidates for mixed precision or exclusions

### B. For deciding whether a quantized model is actually acceptable

Add downstream tasks. A practical minimal suite would be:

- **knowledge / QA:** MMLU or MMLU-Pro
- **commonsense:** HellaSwag, PIQA, Winogrande, ARC-C
- **reasoning / math:** GSM8K, MATH or AIME-style set if relevant
- **coding:** HumanEval or LiveCodeBench if code ability matters
- **instruction following / robustness:** IFEval / TruthfulQA / MT-Bench style subset if chat behavior matters

### C. For NVFP4 specifically

Also measure:

- throughput / tokens per second
- latency
- memory footprint
- kernel fallback rate / backend used

For NVFP4, those system metrics are part of the claim.

---

## Key papers and what they evaluate

Below is a curated list focused on papers that are recent, influential, or directly relevant to FP4/NVFP4.

### 1) A Comprehensive Evaluation of Quantization Strategies for Large Language Models (ACL Findings 2024)
- **Link:** https://aclanthology.org/2024.findings-acl.726.pdf
- **Why it matters:** directly studies whether perplexity tracks downstream benchmarks for quantized LLMs.
- **Metrics used:** perplexity, ten downstream benchmarks, efficiency dimension.
- **Key finding:** **perplexity can serve as a proxy on most benchmarks**, but the paper still argues for a **structured evaluation framework** across knowledge/capacity, alignment, and efficiency.
- **Takeaway for you:** supports keeping perplexity, but **not using it alone**.

### 2) SpinQuant: LLM Quantization with Learned Rotations (2024)
- **Link:** https://arxiv.org/abs/2405.16406
- **Metrics used:** WikiText-2 perplexity, average zero-shot reasoning accuracy, task suites such as ARC, PIQA, Winogrande, HellaSwag and related commonsense tasks.
- **Reported result:** for 4-bit quantization of weights/activations/KV-cache, SpinQuant narrows the gap to full precision to **2.9 points** on zero-shot reasoning for LLaMA-2 7B, outperforming SmoothQuant and QuaRot.
- **Takeaway:** a strong modern PTQ paper does **not** stop at perplexity.

### 3) Post Training Quantization of Large Language Models with Enhanced Methods in Microscaling Formats (2024)
- **Link:** https://arxiv.org/html/2405.07135v2
- **Metrics used:** C4/WikiText-style perplexity plus **eight zero-shot commonsense reasoning tasks**.
- **Key finding:** combinations such as SmoothQuant+GPTQ and AWQ+GPTQ can be synergistic, especially at low bit-widths and with MX formats.
- **Takeaway:** even when the core question is quantization mechanics, evaluation still includes **downstream tasks**.

### 4) Benchmarking Post-Training Quantization of Large Language Models with Microscaling Formats (2026)
- **Link:** https://arxiv.org/html/2601.09555v1
- **Metrics used:** WikiText2 perplexity plus zero-shot tasks including PIQA, Winogrande, HellaSwag, ARC-Easy, ARC-Challenge and others.
- **Why it matters:** explicitly benchmarks PTQ methods under MX/FP4-like settings rather than only classic INT quantization.
- **Takeaway:** the benchmark design itself assumes **perplexity + downstream tasks**.

### 5) FP4 All the Way: Fully Quantized Training of LLMs (2025)
- **Link:** https://arxiv.org/abs/2505.19115
- **Metrics used:** validation/training loss, perplexity, and **zero-shot downstream tasks**.
- **Reported result:** FP4-trained models achieve **downstream performance comparable to BF16** after the proposed training recipe; the paper also argues that NVFP4 is the best FP4 design among tested options.
- **Takeaway:** for FP4-era work, the claim is validated with **task performance**, not just LM loss.

### 6) Pretraining Large Language Models with NVFP4 (2026 technical report)
- **Link:** https://arxiv.org/html/2509.25149v2
- **Metrics used:** training loss, validation perplexity, and broad **downstream task accuracies** across general, reasoning, math, coding, and multilingual domains.
- **Reported result:** example numbers in the report show **MMLU-Pro 62.58 vs 62.62** for NVFP4 vs FP8, with similarly close results on other domains.
- **Takeaway:** one of the clearest NVFP4 papers, and it evaluates **loss + downstream tasks**, not just perplexity.

### 7) Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling (2026)
- **Link:** https://arxiv.org/abs/2512.02010
- **Metrics used:** pretraining loss, WikiText-2/C4 perplexity for PTQ, and **downstream tasks** on Llama/Qwen models.
- **Reported result:** improves NVFP4 behavior during both pretraining and PTQ; snippets report that it can bring perplexity noticeably closer to BF16 and improve downstream performance across tasks.
- **Takeaway:** even a paper centered on a low-level NVFP4 scaling trick still validates with **task-level outcomes**.

### 8) Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery (2026 technical report)
- **Link:** https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf
- **Metrics used:** validation loss for checkpoint selection and downstream evaluations such as **AIME24, AIME25, LiveCodeBench-v6**.
- **Reported result:** NVFP4 PTQ shows gaps on some tasks, and QAD narrows them materially.
- **Takeaway:** particularly relevant to your concern: **validation loss/perplexity is not the whole story**, especially on hard reasoning/coding tasks.

### 9) Bridging the Gap Between Promise and Performance for FP4 Quantization / MR-GPTQ (ICLR 2026 under review)
- **Link:** https://openreview.net/pdf/826643f2babc65767f15bd1e2842bbee49ba8c5a.pdf
- **Metrics used:** accuracy/performance study across PTQ methods, with discussion of quantization error metrics, downstream tasks, and end-to-end speedups on RTX 5090.
- **Reported result:** emphasizes that FP4 formats are promising but fragile; proposes MR-GPTQ and reports both **accuracy** and **performance**.
- **Takeaway:** this is exactly the type of work showing that **error metrics alone are not enough** for native FP4 evaluation.

### 10) Exploring the Trade-Offs: Quantization Methods, Task Difficulty, and Model Size in LLMs From Edge to Giant (IJCAI 2025)
- **Link:** https://www.ijcai.org/proceedings/2025/0902.pdf
- **Metrics used:** 13 datasets across six task categories, including commonsense QA, complex knowledge/language understanding, instruction following, hallucination detection, and newer leaderboards.
- **Key finding:** much recent work relied too heavily on perplexity and old benchmarks; harder and newer tasks reveal more nuanced quantization effects.
- **Takeaway:** strong direct support for your intuition that **downstream-task evaluation is the proper final test**.

### 11) Evaluating the Generalization Ability of Quantized LLMs: Benchmark, Analysis, and Toolbox (ICLR 2025 submission)
- **Link:** https://openreview.net/forum?id=ClkfwM3STw
- **Metrics used:** more than **40 datasets** across in-distribution and out-of-distribution scenarios.
- **Why it matters:** shows the field is moving toward **benchmark suites**, not single-metric evaluation.
- **Takeaway:** if your analysis is intended to generalize beyond one calibration corpus, downstream and OOD evaluation matter.

### 12) A Survey on Model Compression for Large Language Models (TACL 2024)
- **Link:** https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00704/125482/A-Survey-on-Model-Compression-for-Large-Language
- **Why it matters:** survey-level summary of common compression benchmarks and metrics.
- **Metrics noted:** perplexity datasets, zero-shot tasks, reasoning datasets, evaluation harnesses such as LM Harness.
- **Takeaway:** the survey perspective also treats **perplexity as one metric among several**.

---

## What NVFP4-specific work tends to measure

Across the NVFP4 papers/reports above, the recurring metric pattern is:

1. **training loss / validation perplexity**  
   to show numerical stability and language-model quality

2. **downstream tasks**  
   to show preserved capability in reasoning, knowledge, coding, or multilingual performance

3. **hardware/system metrics**  
   to justify native FP4 Tensor Core use

That is important because NVFP4’s value proposition is inherently **joint**:

- acceptable model quality
- plus materially better runtime efficiency

If you evaluate NVFP4 using only output norm or only perplexity, you are only validating part of the claim.

---

## Answer to your concrete question

### Are the current two metrics enough?

**For sensitivity analysis only:** maybe close, but still incomplete.

- `layer_out_norm` is okay as a fast local proxy.
- `layerwise_perplexity` is much better than output norm alone.

If the goal is to **rank layers / experts for mixed precision**, those metrics are useful.

**For overall model evaluation:** no.

The literature suggests the final decision should not be based only on:

- hidden-state difference
- perplexity

It should include **downstream tasks** and, for NVFP4, **system performance**.

### Is your intuition about downstream tasks correct?

**Yes.** Recent papers strongly support it.

The strongest practical recommendation is:

- use proxies for search and calibration  
- use downstream tasks for final acceptance

That is the pattern followed by the better recent quantization papers, and especially by FP4/NVFP4 work.

---

## Suggested evaluation protocol for your repo

### Stage 1: cheap screening

- `layer_out_norm`
- `layerwise_model_out_norm`
- `layerwise_perplexity`
- full-model `quantize_all` perplexity + KL

Purpose: identify sensitive layers/expert projections and compare candidate quantization recipes quickly.

### Stage 2: acceptance gate

Run a compact downstream suite on the fully quantized model:

- **MMLU / MMLU-Pro**
- **HellaSwag**
- **PIQA**
- **Winogrande**
- **ARC-C**
- **GSM8K**
- **HumanEval or LiveCodeBench** if code matters

If this is an instruction-tuned model, also consider:

- **IFEval**
- **TruthfulQA**
- **MT-Bench** or an equivalent internal eval

### Stage 3: NVFP4 deployment metrics

- throughput
- latency
- VRAM / memory footprint
- backend / kernel fallback information

---

## Bottom line

Your current metrics are **good proxies**, but **not a complete evaluation**.

The literature-supported stance is:

- **output norm** → useful for local sensitivity ranking  
- **perplexity** → useful model-level proxy, often correlated with downstream quality  
- **downstream tasks** → required for final claims about preserved model capability  
- **hardware metrics** → required for NVFP4-specific claims

If you want a single sentence recommendation:

> Keep layer-output norm and perplexity for analysis, but do not use them as the sole basis for accepting a quantized model; final evaluation should include downstream tasks, and for NVFP4 also include throughput/latency/memory measurements.

---

## Appendix: practical paper list

- ACL 2024 comprehensive evaluation of quantized LLMs: https://aclanthology.org/2024.findings-acl.726.pdf
- SpinQuant: https://arxiv.org/abs/2405.16406
- PTQ with enhanced MX methods: https://arxiv.org/html/2405.07135v2
- Benchmarking PTQ with MX formats: https://arxiv.org/html/2601.09555v1
- FP4 All the Way: https://arxiv.org/abs/2505.19115
- Pretraining LLMs with NVFP4: https://arxiv.org/html/2509.25149v2
- Four Over Six: https://arxiv.org/abs/2512.02010
- NVFP4 QAD report: https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf
- MR-GPTQ / FP4 quantization gap study: https://openreview.net/pdf/826643f2babc65767f15bd1e2842bbee49ba8c5a.pdf
- IJCAI 2025 trade-off study: https://www.ijcai.org/proceedings/2025/0902.pdf
- Quantized LLM generalization benchmark: https://openreview.net/forum?id=ClkfwM3STw
- TACL survey on model compression for LLMs: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00704/125482/A-Survey-on-Model-Compression-for-Large-Language
