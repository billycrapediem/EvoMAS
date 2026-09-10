"""Prepare one local Verified instance using pinned upstream installation recipes."""
import json
import fcntl
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def publish_configs(instance_id, setup, task):
    """Update only this instance and preserve other locally prepared entries."""
    from src.dataset.local_swe_evaluator import normalize_instance_id
    directory = ROOT / "dataset/swe_bench_verified"
    key = normalize_instance_id(instance_id)[1]
    with (directory / ".setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for filename, value in (("setup_map.json", setup), ("tasks_map.json", task)):
            path = directory / filename
            data = json.loads(path.read_text()) if path.exists() else {}
            data[key] = value
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(data, indent=2))
            temporary.replace(path)


def prepare_instance(instance_id, workspace):
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS, MAP_REPO_TO_INSTALL
    from swebench.harness.test_spec.python import get_environment_yml, get_requirements, get_test_directives

    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    ready_path = workspace / "ready.json"
    dataset_dir = ROOT / "dataset/swe_bench_verified"
    items = json.loads((dataset_dir / "test.json").read_text())
    item = next(t for t in items if t["id"] == instance_id)
    instance = dict(item["metadata"])
    instance["version"] = next(t.split(":", 1)[1] for t in item["tag"] if t.startswith("version:"))
    specs = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]
    repo = workspace / "repos" / instance["repo"].split("/")[-1]
    repo.parent.mkdir(exist_ok=True)
    task = dict(instance)
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        if isinstance(task[key], str):
            task[key] = json.loads(task[key])
    if ready_path.exists():
        ready = json.loads(ready_path.read_text())
        if ready["instance_id"] != instance_id or ready["setup"]["repo_path"] != str(repo):
            raise ValueError("Prepared workspace belongs to a different instance or path")
        os.environ["CONDA_ENVS_PATH"] = str(workspace / "envs")
        if not (workspace / "envs" / ready["setup"]["env_name"] / "bin/python").exists():
            raise ValueError("Prepared environment is missing; use a new workspace")
        publish_configs(instance_id, ready["setup"], task)
        return ready["setup"]
    # An isolated checkout prevents resets/evaluation from touching downloaded source trees.
    existing = Path(os.getenv("EVOMAS_REPOS_DIR", ROOT / "dataset/repos")) / repo.name
    source = str(existing) if (existing / ".git").exists() else f"https://github.com/{instance['repo']}.git"
    subprocess.run(["git", "clone", "--no-checkout", source, str(repo)], check=True, timeout=600)
    subprocess.run(["git", "checkout", "--detach", instance["base_commit"]], cwd=repo, check=True, capture_output=True)
    # Remove future history from the agent's checkout while retaining the requested base tree.
    subprocess.run(["git", "remote", "remove", "origin"], cwd=repo, check=True)
    refs = subprocess.check_output(["git", "for-each-ref", "--format=%(refname)", "refs/heads", "refs/tags"], cwd=repo, text=True)
    for ref in refs.splitlines():
        subprocess.run(["git", "update-ref", "-d", ref], cwd=repo, check=True)
    subprocess.run(["git", "reflog", "expire", "--expire=now", "--all"], cwd=repo, check=True)
    subprocess.run(["git", "gc", "--prune=now"], cwd=repo, check=True)

    env_name = "evomas_" + workspace.parent.name
    os.environ["CONDA_ENVS_PATH"] = str(workspace / "envs")
    conda_base = subprocess.check_output(["conda", "info", "--base"], text=True).strip()
    commands = ["set -euo pipefail", f"source {shlex.quote(conda_base + '/etc/profile.d/conda.sh')}"]
    packages = specs.get("packages", "")
    if packages == "environment.yml":
        env_file = workspace / "environment.yml"
        env_file.write_text(get_environment_yml(instance, env_name))
        if specs.get("no_use_env"):
            commands += [f"conda create -y -n {env_name} python={specs['python']}",
                         f"conda env update -n {env_name} -f {shlex.quote(str(env_file))}"]
        else:
            commands += [f"conda env create -n {env_name} -f {shlex.quote(str(env_file))}"]
    else:
        extra = "" if packages == "requirements.txt" else packages
        commands += [f"conda create -y -n {env_name} python={specs['python']} {extra}"]
    commands += [f"conda activate {env_name}"]
    if packages == "requirements.txt":
        req_file = workspace / "requirements.txt"
        req_file.write_text(get_requirements(instance))
        commands += [f"python -m pip install -r {shlex.quote(str(req_file))}"]
    if specs.get("pip_packages"):
        commands += ["python -m pip install " + " ".join(shlex.quote(p) for p in specs["pip_packages"])]
    commands += [f"cd {shlex.quote(str(repo))}"]
    if instance["repo"] in MAP_REPO_TO_INSTALL:
        commands += [MAP_REPO_TO_INSTALL[instance["repo"]]]
    commands += specs.get("pre_install", [])
    if specs.get("install"):
        commands += [specs["install"]]
    script = workspace / "setup.sh"
    script.write_text("\n".join(commands) + "\n")
    with (workspace.parent / "setup.log").open("w") as log:
        subprocess.run(["bash", str(script)], cwd=workspace, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=1800)
    subprocess.run(["git", "reset", "--hard", instance["base_commit"]], cwd=repo, check=True, capture_output=True)
    setup = {"repo_path": str(repo), "env_name": env_name, "python": specs["python"],
             "install": specs.get("install", ""), "pre_install": specs.get("pre_install", []),
             "test_cmd": specs["test_cmd"] + " " + " ".join(shlex.quote(t) for t in get_test_directives(instance)),
             "eval_commands": specs.get("eval_commands", [])}
    publish_configs(instance_id, setup, task)
    ready_path.write_text(json.dumps({"instance_id": instance_id, "setup": setup}, indent=2))
    return setup
