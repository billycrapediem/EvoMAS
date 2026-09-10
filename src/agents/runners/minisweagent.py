"""mini-swe-agent v2 backend for EvoMAS's prepared local repositories."""
import copy
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

import yaml
from dotenv import load_dotenv

from ..spec import AgentResult, AgentSpec
from .base import BaseAgentRunner

ROOT = Path(__file__).resolve().parents[3]


def redact(value):
    """Remove credentials from nested persisted data and exception messages."""
    if isinstance(value, dict):
        return {k: "[REDACTED]" if any(s in k.lower() for s in ("api_key", "authorization", "password", "secret"))
                else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for key, secret in os.environ.items():
            if secret and any(s in key.upper() for s in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
                value = value.replace(secret, "[REDACTED]")
    return value


def load_config(config_spec="default"):
    from minisweagent import __file__ as package_file
    aliases = {"default": "benchmarks/swebench.yaml", "swebench": "benchmarks/swebench.yaml",
               "simple": "benchmarks/swebench.yaml", "mini_default": "benchmarks/swebench.yaml"}
    if config_spec in aliases:
        path = Path(package_file).parent / "config" / aliases[config_spec]
    else:
        path = Path(config_spec).expanduser()
        if not path.is_absolute():
            path = ROOT / path
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("agent"), dict):
        raise ValueError(f"Invalid mini-swe-agent configuration: {path}")
    from minisweagent.agents.default import AgentConfig
    unknown = set(config["agent"]) - set(AgentConfig.model_fields)
    if unknown:
        raise ValueError(f"Unknown mini-swe-agent v2 agent options: {sorted(unknown)}")
    AgentConfig(**config["agent"])
    if config_spec == "simple":
        config["agent"]["step_limit"] = 20
    return config


def register_endpoint_pricing(model_id):
    """Use canonical model pricing for an OpenAI-compatible alias when available.

    This is an estimate for third-party endpoints; an explicit LiteLLM registry
    can override it with the endpoint's actual rates.
    """
    import litellm
    if not model_id.startswith("openai/") or model_id in litellm.model_cost:
        return model_id
    canonical = model_id.removeprefix("openai/")
    if canonical in litellm.model_cost:
        rates = dict(litellm.model_cost[canonical], litellm_provider="openai")
        litellm.register_model({model_id: rates})
        return canonical
    return model_id


class MinisweagentRunner(BaseAgentRunner):
    def __init__(self, working_dir=None, config="default", **kwargs):
        self.working_dir = Path(working_dir).resolve() if working_dir else None
        self.full_config = load_config(config)
        self.config_name = config
        self._last_agent = None

    def create_agent(self, spec: AgentSpec, working_dir: Path):
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models import get_model

        load_dotenv(ROOT / ".env", override=False)
        model_config = copy.deepcopy(self.full_config.get("model", {}))
        model_config.pop("model_name", None)
        model_id = spec.model_id
        if ":" in model_id:
            provider, name = model_id.split(":", 1)
            model_id = f"{provider}/{name}"
        elif os.getenv("BASE_URL") and "/" not in model_id:
            model_id = f"openai/{model_id}"
        model_kwargs = model_config.setdefault("model_kwargs", {})
        model_kwargs.update(temperature=spec.temperature, max_tokens=spec.max_tokens, timeout=60)
        if model_id.startswith("openai/") and os.getenv("BASE_URL"):
            model_kwargs["api_base"] = os.environ["BASE_URL"]
        if model_id.startswith("openai/") and os.getenv("API_KEY"):
            model_kwargs["api_key"] = os.environ["API_KEY"]
        model = get_model(model_id, model_config)
        pricing_source = register_endpoint_pricing(model_id) if os.getenv("BASE_URL") else model_id
        env_config = self.full_config.get("environment", {})
        env_vars = dict(env_config.get("env", {}))
        # The upstream Docker configuration sources /root/.bashrc; local mode must not.
        env_vars.pop("BASH_ENV", None)
        conda_env = env_config.get("conda_env")

        class PreparedEnvironment(LocalEnvironment):
            def execute(self, action, **kwargs):
                if conda_env:
                    command = action.get("command", "")
                    action = dict(action, command=f"conda run --no-capture-output -n {shlex.quote(conda_env)} bash -c {shlex.quote(command)}")
                return super().execute(action, **kwargs)

        env = PreparedEnvironment(cwd=str(working_dir), timeout=env_config.get("timeout", 60), env=env_vars)
        agent_config = copy.deepcopy(self.full_config["agent"])
        for key in ("system_template", "instance_template"):
            agent_config[key] = agent_config[key].replace("/testbed", str(working_dir))
        # Save only redacted trajectories, including intermediate steps and errors.
        agent_config.pop("output_path", None)
        output_dir = Path(self.full_config.get("output_dir", ROOT / "output" / "minisweagent")).resolve()
        if output_dir == working_dir or working_dir in output_dir.parents:
            raise ValueError("Trajectory output_dir must be outside the task repository")
        output_dir.mkdir(parents=True, exist_ok=True)
        trajectory_dir = Path(tempfile.mkdtemp(prefix="agent_", dir=output_dir))
        trajectory_path = trajectory_dir / "trajectory.json"

        class RecordedAgent(DefaultAgent):
            def save(self, path=None, *extra_dicts):
                data = redact(self.serialize(*extra_dicts))
                data.setdefault("info", {})["pricing_source"] = pricing_source
                trajectory_path.write_text(json.dumps(data, indent=2))
                return data

        agent = RecordedAgent(model=model, env=env, **agent_config)
        agent.trajectory_path = trajectory_path
        return agent

    @staticmethod
    def _extract_problem_statement(task):
        if "Problem Statement:" in task:
            task = task.split("Problem Statement:", 1)[1].strip()
        return task.split("\n\nCRITICAL INSTRUCTIONS:", 1)[0].strip()

    @staticmethod
    def _get_git_diff(repo):
        # Intent-to-add includes new source files without staging their contents.
        new_files = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=repo).decode().split("\0")
        files = [p for p in new_files if p and p != "patch.txt" and not p.endswith((".traj", ".log"))]
        if files:
            subprocess.run(["git", "add", "-N", "--", *files], cwd=repo, check=True, capture_output=True)
        return subprocess.check_output(["git", "diff", "HEAD", "--", ".", ":(exclude)patch.txt"], cwd=repo, text=True)

    @staticmethod
    def validate_patch(repo, patch):
        if not patch or not patch.lstrip().startswith("diff --git "):
            return False
        # A private index validates against HEAD regardless of staged/working changes.
        with tempfile.TemporaryDirectory(prefix="evomas_index_") as directory:
            env = os.environ | {"GIT_INDEX_FILE": str(Path(directory) / "index")}
            subprocess.run(["git", "read-tree", "HEAD"], cwd=repo, env=env, check=True, capture_output=True)
            check = subprocess.run(["git", "apply", "--cached", "--check", "-"], input=patch,
                                   cwd=repo, env=env, text=True, capture_output=True)
            return check.returncode == 0

    def run(self, spec, task, context=None):
        agent = None
        try:
            match = re.search(r"^Repository:\s*(/[^\n]+)", task, re.MULTILINE)
            repo = Path(match.group(1).strip()).resolve() if match else self.working_dir
            if repo is None or not repo.is_dir():
                raise ValueError("mini-swe-agent requires a prepared repository path")
            subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=repo, check=True, capture_output=True)
            agent = self.create_agent(spec, repo)
            self._last_agent = agent
            problem = self._extract_problem_statement(task)
            relevant = {k: v for k, v in (context or {}).items() if k != "task" and v not in (None, "", {}, [])}
            if relevant:
                problem += "\n\nContext from previous agents:\n" + json.dumps(relevant, default=str)
            result = agent.run(problem)
            status = result.get("exit_status", "Unknown")
            patch = result.get("submission", "")
            if not self.validate_patch(repo, patch):
                patch = self._get_git_diff(repo)
            valid = self.validate_patch(repo, patch)
            success = status == "Submitted" and valid
            usage = [message.get("extra", {}).get("response", {}).get("usage", {}) or {}
                     for message in agent.messages]
            model_stats = {"instance_cost": agent.cost, "api_calls": agent.n_calls,
                           "tokens_sent": sum(u.get("prompt_tokens", 0) or 0 for u in usage),
                           "tokens_received": sum(u.get("completion_tokens", 0) or 0 for u in usage)}
            return AgentResult(agent_id=spec.id, content=patch if valid else "", success=success,
                               error=None if success else f"mini-swe-agent exited with {status}; valid patch={valid}",
                               metadata={"exit_status": status, "trajectory_path": str(agent.trajectory_path),
                                         "model_stats": model_stats})
        except Exception as exc:
            return AgentResult(agent_id=spec.id, content="", success=False, error=redact(str(exc)),
                               metadata={"trajectory_path": str(agent.trajectory_path)} if agent else {})
        finally:
            if agent is not None:
                agent.save()
