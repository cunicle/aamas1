# Notes for Claude sessions on this repo

Project: "Silent Dissent" (AAMAS submission). When an agent flips to the majority
in multi-agent debate, does the Jacobian lens still read out its original answer?
Read README.md (Chinese) for the full pipeline, conditions and metrics.

## Running on the GPU box (RunPod A100)

- Work in `/workspace/aamas1` (the network volume); anything outside /workspace is lost on terminate.
- Main config: `configs/qwen35_4b.yaml` (Qwen3.5-4B + official pre-fitted J-lens from the Hub).
- First time: `bash scripts/setup_gpu.sh`. Do not start experiments unless
  `scripts/check_model.py` ends with `ALL CHECKS PASSED`; if it fails, report the output, don't work around it.
- Run long jobs inside tmux (`tmux new -s <name>`), log to a file, e.g.
  `python scripts/run_pressure.py --config C 2>&1 | tee results/pressure.log`.
  run_pressure / run_intervention do not resume mid-way, so never kill a running one casually.
- Do a `--limit 50` pass of run_pressure / run_intervention before the full run.

## Rules that protect the study's validity

- Step order: run_baseline -> select_layer -> **commit and push `prereg/*.json`** -> only then run_pressure.
  The readout layer must be fixed before anyone looks at pressure data.
- Never run `select_layer.py --force`, never edit prereg files, never pick layers after seeing pressure results.
  If a second layer is wanted, it goes into prereg before the pressure run.
- `results/` and `lenses/` are gitignored; only prereg files and code are committed.
  Push to branch `claude/pensive-dirac-d7zg2t` unless told otherwise.

## Cost

The GPU bills per minute. When a job finishes and nothing else is queued, tell the user
so they can terminate the pod (data on /workspace survives).
