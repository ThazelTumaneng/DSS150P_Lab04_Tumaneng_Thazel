# src/config.py
import os
from pathlib import Path
import yaml

# Safe import in case dotenv is missing in any sub-environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = os.getenv("CONFIG_PATH", "config/settings.yml")
PIPELINE_ENV = os.getenv("PIPELINE_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))


def load_config():
    """Loads YAML configuration from settings path."""
    full_path = ROOT_DIR / CONFIG_PATH
    if not full_path.exists():
        return {}
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


# Alias in case any module calls load_settings
load_settings = load_config