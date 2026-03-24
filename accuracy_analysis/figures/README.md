# Figure generation blocker

The literature-review workflow requested an AI-generated schematic.

Attempted tool path:

- `/home/huanchen/claude-scientific-skills/scientific-skills/scientific-schematics/scripts/generate_schematic.py`

Current blocker:

- `OPENROUTER_API_KEY` is not set in this environment, and the schematic generator requires it.

Suggested command once credentials are available:

```bash
python /home/huanchen/claude-scientific-skills/scientific-skills/scientific-schematics/scripts/generate_schematic.py \
  "A clean academic diagram showing a three-stage quantization evaluation stack: (1) layer-level proxy metrics such as output norm, (2) model-level proxy metrics such as perplexity and KL, and (3) final downstream-task and hardware evaluation for NVFP4 including MMLU, GSM8K, HumanEval, throughput, latency, and memory" \
  -o accuracy_analysis/figures/quantization_eval_stack.png \
  --doc-type report
```
