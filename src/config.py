from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent

def load_config(config_path: str = "config/settings.yml") -> dict:
    with open(ROOT_DIR/config_path, "r") as f:
        return yaml.safe_load(f)

