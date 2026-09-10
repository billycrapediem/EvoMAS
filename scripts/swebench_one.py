"""One-instance, uncached EvoMAS integration and local evaluation smoke test."""
import argparse
import contextlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from dotenv import load_dotenv
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_id", nargs="?", default="astropy__astropy-12907")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_paper" / "swebench_one")
    parser.add_argument("--step-limit", type=int, default=250)
    parser.add_argument("--cost-limit", type=float, default=3.0)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--workspace", type=Path, help="Reuse an instance workspace prepared by a previous smoke run")
    parser.add_argument("--reference-only", action="store_true", help="Validate the local test environment without model calls")
    args = parser.parse_args()
    os.chdir(ROOT)
    load_dotenv(ROOT / ".env", override=False)
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="run_", dir=args.output_root)).resolve()
    report = {"instance_id": args.instance_id, "dataset": "swe_bench_verified", "command": sys.argv,
              "versions": {p: importlib.metadata.version(p) for p in ("mini-swe-agent", "swebench")},
              "step_limit": args.step_limit, "cost_limit": args.cost_limit, "timeout": args.timeout,
              "seed": None, "temperature": 0.0}
    from src.agents.runners.minisweagent import redact
    class RedactedLog(logging.Filter):
        def filter(self, record):
            record.msg = redact(record.getMessage())
            record.args = ()
            return True
    handler = logging.FileHandler(out / "run.log")
    handler.addFilter(RedactedLog())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    print(f"Artifacts: {out}")
    try:
        if min(args.step_limit, args.cost_limit, args.timeout) <= 0:
            raise ValueError("Step, cost and timeout limits must be positive")
        data = json.loads((ROOT / "dataset/swe_bench_verified/test.json").read_text())
        task = next((t for t in data if t["id"] == args.instance_id), None)
        if task is None:
            raise ValueError("Instance is not in the prepared SWE-bench Verified dataset")
        if not args.reference_only:
            missing = [key for key in ("MODEL_ID", "BASE_URL", "API_KEY") if not os.getenv(key)]
            if missing:
                raise ValueError("Missing configuration in EvoMAS/.env: " + ", ".join(missing))
        from src.dataset.swe_smoke_setup import prepare_instance
        setup = prepare_instance(args.instance_id, args.workspace or out / "workspace")
        from src.dataset.local_swe_evaluator import evaluate_patch
        with (out / "reference.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            reference = evaluate_patch(args.instance_id, task["gt"], dataset="swe_bench_verified", use_cache=True)
        (out / "reference.json").write_text(json.dumps(reference, indent=2))
        test_log = Path("/tmp") / ("test_output_" + args.instance_id.replace("/", "_") + ".log")
        if test_log.exists():
            shutil.copyfile(test_log, out / "reference_tests.log")
        if reference.get("resolved") != "FULL" or reference.get("error"):
            raise RuntimeError("Reference patch did not pass; see reference.json and reference.log")
        if args.reference_only:
            report["status"] = "reference_passed"
            return 0
        os.environ["EVOMAS_REPOS_DIR"] = str(Path(setup["repo_path"]).parent)
        from src.agents.runners.minisweagent import load_config
        config = load_config("default")
        config["agent"].update(step_limit=args.step_limit, cost_limit=args.cost_limit,
                               wall_time_limit_seconds=args.timeout)
        config["environment"] = {"timeout": 60, "conda_env": setup["env_name"]}
        config["output_dir"] = str(out / "trajectories")
        agent_config = out / "agent.yaml"
        agent_config.write_text(yaml.safe_dump(config))
        mas = {"name": "single_minisweagent_swebench", "backend": "minisweagent",
               "agents": {"worker": {"id": "worker", "role": "worker", "agent_type": "DefaultAgent",
                                      "model_id": os.environ["MODEL_ID"], "temperature": 0.0}},
               "topology": {"reports_to": {}},
               "execution": {"parallel_workers": False, "max_retries": 0, "agent_config": str(agent_config)}}
        config_path = out / "mas.yaml"
        config_path.write_text(yaml.safe_dump(mas))
        report["model_id"] = os.environ["MODEL_ID"]
        from src.utils.mas_runner import MasRunner
        runner = MasRunner(config_path, "swe_bench_verified", output_dir=out, use_cache=False,
                           skip_evaluation=True, verbose=False, task_timeout=args.timeout)
        with (out / "agent.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = runner.run_single_task(args.instance_id)
        if result.error or not result.result:
            raise RuntimeError(result.error or "Agent returned no patch")
        report["agent_metadata"] = result.metadata
        patch = result.result
        from src.agents.runners.minisweagent import MinisweagentRunner
        if not MinisweagentRunner.validate_patch(Path(setup["repo_path"]), patch):
            raise RuntimeError("Agent output is not a patch applicable to the base commit")
        (out / "model.patch").write_text(patch)
        with (out / "evaluation.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            evaluation = evaluate_patch(args.instance_id, patch, dataset="swe_bench_verified", use_cache=True)
        test_log = Path("/tmp") / ("test_output_" + args.instance_id.replace("/", "_") + ".log")
        if test_log.exists():
            shutil.copyfile(test_log, out / "tests.log")
        (out / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
        if evaluation.get("error"):
            raise RuntimeError(evaluation["error"])
        report["status"] = "resolved" if evaluation.get("resolved") == "FULL" else "unresolved"
        return 0 if report["status"] == "resolved" else 1
    except Exception as exc:
        message = str(exc)
        if os.getenv("API_KEY"):
            message = message.replace(os.environ["API_KEY"], "[REDACTED]")
        report.update(status="infrastructure_error", error=message)
        print(message, file=sys.stderr)
        return 2
    finally:
        handler.flush()
        for artifact in out.rglob("*"):
            if artifact.is_file() and artifact.suffix in (".log", ".json", ".yaml", ".txt") and "workspace" not in artifact.relative_to(out).parts:
                artifact.write_text(redact(artifact.read_text(errors="replace")))
        (out / "summary.json").write_text(json.dumps(redact(report), indent=2))
        print(f"Status: {report.get('status', 'interrupted')}; report: {out / 'summary.json'}")


if __name__ == "__main__":
    sys.exit(main())
