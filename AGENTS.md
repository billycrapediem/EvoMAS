# Repository Guidelines

## Project Structure & Module Organization

EvoMAS evolves multi-agent system configurations and evaluates them on BBEH, WorkBench, and SWE-bench. `main.py` orchestrates selection, evolution, evaluation, and memory updates.

- `src/meta_model/`: evolutionary operators, configuration pools, rewards, and memory.
- `src/mas/`, `src/agents/`, `src/topology/`: configuration schemas, execution backends, and routing.
- `src/dataset/`, `src/models/`, `src/tools/`, `src/utils/`: benchmark adapters, model integrations, tools, and utilities.
- `src/prompts/templates/` and `mas_pools/`: prompt assets and benchmark-specific YAML configurations.
- `scripts/`: dataset preparation and benchmark launchers; shared defaults live in `scripts/common.sh`.

Downloaded data lives in `dataset/`; benchmark outputs default to `output_paper/`. Neither is needed for source-only edits.

## Build, Test, and Development Commands

Run commands from the repository root:

```bash
conda create -n mas python=3.11 -y
conda activate mas
pip install -r requirements.txt
scripts/prepare_datasets.sh --bbeh-only
python main.py --help
NUM_EVAL_TASKS=1 MAX_STEPS=1 WORKERS=1 scripts/run_bbeh.sh mini
```

These install dependencies, prepare BBEH data, display CLI options, and run a minimal integration smoke test. Launch other benchmarks with `scripts/run_workbench.sh email` or `scripts/run_swebench.sh lite`. Launchers use the `mas` Conda environment by default; override `CONDA_ENV` when needed. There is no separate build step.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, `PascalCase` classes, and uppercase constants. Follow neighboring type hints and docstrings. YAML uses two-space indentation; use descriptive configuration names such as `peer_review.yaml`. Keep changes focused and avoid unrelated reformatting. No formatter or linter configuration is checked in.

## Testing Guidelines

No automated unit-test suite, test framework configuration, or coverage threshold is tracked. Use the small benchmark run above for relevant integration changes; it requires prepared data and provider credentials and invokes model APIs. Record the dataset, models, seed, command, and outcome. For new regression tests, use `tests/test_<module>.py` and document execution; adjust `.gitignore`, which currently excludes `test*.py`, so tests are tracked.

## Commit & Pull Request Guidelines

History uses short, plain-language subjects, such as `Update paper title and mention license in README`; no formal prefix convention is established. Follow `CONTRIBUTING.md`: start from current `main`, check existing PRs, and discuss significant work in an issue. Keep PRs focused, link relevant issues, describe behavior changes, and report validation results and environment details.

## Configuration & Credentials

Keep credentials in local `.env` files or provider credential stores. Never commit secrets, downloaded repositories, or generated outputs. Override models through `META_MODEL`, `JUDGE_MODEL`, and `AGENT_MODELS` when needed.
