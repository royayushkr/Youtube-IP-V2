import json
import os
from typing import Dict, List

EXTENSIONS_CONFIG_PATH = os.path.join("config", "extensions.json")

DEFAULT_EXTENSIONS: Dict[str, bool] = {
    "Home": True,
    "Channel Analysis": True,
    "Recommendations": True,
    "Ytuber": True,
    "Extension Center": True,
    "Deploy Notes": True,
}


def load_extensions() -> Dict[str, bool]:
    if not os.path.exists(EXTENSIONS_CONFIG_PATH):
        return DEFAULT_EXTENSIONS.copy()
    try:
        with open(EXTENSIONS_CONFIG_PATH, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        merged = DEFAULT_EXTENSIONS.copy()
        for k, v in data.items():
            if k in merged:
                merged[k] = bool(v)
        return merged
    except Exception:
        return DEFAULT_EXTENSIONS.copy()


def save_extensions(ext: Dict[str, bool]) -> None:
    os.makedirs(os.path.dirname(EXTENSIONS_CONFIG_PATH), exist_ok=True)
    safe = DEFAULT_EXTENSIONS.copy()
    for k in safe:
        safe[k] = bool(ext.get(k, safe[k]))
    with open(EXTENSIONS_CONFIG_PATH, "w", encoding="utf-8") as fp:
        json.dump(safe, fp, indent=2)


def get_navigation_items() -> List[str]:
    ext = load_extensions()
    ordered = [k for k in DEFAULT_EXTENSIONS.keys() if ext.get(k, False)]
    if not ordered:
        return ["Home"]
    return ordered
