import argparse
import sys

from .config import load_config
from .pipeline import run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spendcube", description="Config-driven spend cube builder")
    ap.add_argument("--config", required=True, help="company config (YAML/JSON)")
    ap.add_argument("--data-dir", default=None, help="folder with the company files (overrides data_dir in config)")
    ap.add_argument("--out", default=None, help="output CSV path (default: from config)")
    ap.add_argument("--labels-dir", default=None, help="folder holding labelled data (e.g. the original data when running new records)")
    ap.add_argument("--no-labels", action="store_true", help="ignore any labels (zero-label mode)")
    ap.add_argument("--embeddings", choices=["auto", "on", "off"], default=None)
    ap.add_argument("--llm", choices=["auto", "off"], default=None, help="optional LLM re-check of low-confidence lines")
    a = ap.parse_args(argv)
    ov = {}
    if a.embeddings:
        ov.setdefault("classification", {})["embeddings"] = a.embeddings
    if a.llm:
        ov.setdefault("classification", {}).setdefault("llm", {})["enabled"] = a.llm
    if a.labels_dir:
        ov["labels"] = {"data_dir": a.labels_dir}
    cfg = load_config(a.config, a.data_dir, ov)
    run(cfg, a.out or str(cfg.path.parent / cfg["output"]["file"]), use_labels=not a.no_labels)


if __name__ == "__main__":
    sys.exit(main())
