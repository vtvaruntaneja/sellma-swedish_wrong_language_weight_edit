#!/usr/bin/env bash
# Reproduce everything end to end on an Apple-silicon Mac (MPS). Usage: ./run.sh [--quick]
# --quick uses 10 dev prompts per set (never a test file) and writes to results_quick/ instead of results/.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
PY=.venv/bin/python

$PY data.py                                    # data/ from data/source/ and wikimedia/wikipedia
$PY experiment.py langid "$@"                  # language-detector check
$PY experiment.py pilot --model qwen35 "$@"    # pilot, Qwen3.5-0.8B
$PY experiment.py pilot --model qwen25 "$@"    # pilot, Qwen2.5-0.5B-Instruct
$PY experiment.py dev128 "$@"                  # dev round 1: 128 tokens, 8 settings
$PY experiment.py base512 "$@"                 # base model on dev prompts at 512 tokens
$PY experiment.py dev512 "$@"                  # dev round 2: 8 settings, selection, English check
$PY experiment.py lora --skip-dev-eval "$@"    # LoRA baseline on dev prompts (its dev evaluation was not run)
$PY experiment.py test "$@"                    # all conditions on the test sets, once
$PY analyze.py sheet "$@"                      # blinded sheet for the hand annotation (understanding_labels.jsonl)
$PY analyze.py "$@"                            # summary.json, summary.md, examples.md
if [[ " $* " != *" --quick "* ]]; then $PY analyze.py note; fi   # results tables in NOTE.md
