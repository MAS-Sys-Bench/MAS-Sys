# Controlled workflow ranking (Fig. 1b)

This single-GPU experiment runs five fixed workflows under five serving
conditions. It starts one vLLM instance per condition and executes ten
sequential, deterministically shuffled blocks within each instance. All
conditions use the same generated factual context and workload definitions.
Workflow outputs feed downstream prompts, and the background load follows a
poll interval and active-request cap. The repetitions characterize variation
within a serving instance.

Use a Linux host with an H200, `taskset`, and the OpenAI and HTTPX Python
dependencies from the main requirements. The historical server reported
vLLM `0.19.2.dev0+gf150107.local`; point `--vllm-python` at a compatible
installation. Select an available GPU and a CPU core in the process affinity
mask. The default Hugging Face model revision is pinned to the historical
Qwen2.5-14B-Instruct snapshot.

From the repository root, run the paper configuration:

```bash
python experiments/Motivation/motivation_workflow_ranking.py \
  --vllm-python /path/to/vllm-env/bin/python --gpu 0 --cpu-core 0 \
  --seed 2027 --context-tokens 2048 --warmups 0 --repetitions 10 \
  --star-workers 14 --dag-width 6 --tool-cpu-work 3000 \
  --decode-context-fraction 0.08
```

Use `--model /path/to/local/snapshot` to use an already downloaded model.
Per-condition `runs.jsonl` files contain workflow and per-call timings.
`pressure.jsonl` records background requests for the two pressure conditions.
The harness writes aggregate ranking by median; the paper's appendix
also summarizes mean latency and within-instance intervals from the same
per-execution records. Outputs default to the ignored `results/` directory.

The executable and its extracted helpers derive from the original
workflow-ranking implementation (source commit `41b8f4ff5ea4`).
They are distributed under the [MIT License](LICENSE).
