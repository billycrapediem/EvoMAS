"""Shared .env configuration for pipeline roles and model clients."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


def load_model_environment():
    load_dotenv(ROOT / ".env", override=False)


def pipeline_models():
    """Exported role overrides > .env role overrides > shared MODEL_ID."""
    load_model_environment()
    shared = os.getenv("MODEL_ID")
    meta = os.getenv("META_MODEL") or shared
    judge = os.getenv("JUDGE_MODEL") or meta
    agents = (os.getenv("AGENT_MODELS") or shared or "").split()
    return meta, judge, agents


def constrain_agent_models(config_yaml, models):
    """Keep pool and generated agents within the configured model palette."""
    import yaml
    if not models:
        raise ValueError("Set MODEL_ID or AGENT_MODELS in .env, or pass --model-list")
    config = yaml.safe_load(config_yaml)
    for agent in config["agents"].values():
        if agent.get("model_id") not in models:
            agent["model_id"] = models[0]
    return yaml.safe_dump(config, sort_keys=False)


if __name__ == "__main__":
    meta, judge, agents = pipeline_models()
    if not meta or not agents:
        raise SystemExit("Set MODEL_ID in EvoMAS/.env (or META_MODEL and AGENT_MODELS overrides)")
    print("\n".join([meta, judge, *agents]))
