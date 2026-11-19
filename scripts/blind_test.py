"""One command for the blind test: runs A2, B2 and C with the frozen tool and writes the three files into submission/.

  python scripts/blind_test.py --name yourname --a2 PATH/A2 --b2 PATH/B2 --c PATH/C

A2 / B2 use the existing configs and train on the original labelled data in data/company_A and data/company_B
(if those folders are missing the tool runs with no labels). For C the config is drafted by scripts/make_config.py
from the files in the C folder (largest CSV = invoice lines, CSV whose name mentions tree/taxonom/categor = tree);
override with --c-file / --c-tree. The drafted config is copied to submission/company_C.yaml.
Each step is independent: a failure in one is printed and the others still run. No code is edited by this script."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    return subprocess.call([str(c) for c in cmd], cwd=ROOT) == 0


def pick_files(d: Path, lines=None, tree=None):
    csvs = [p for p in d.rglob("*") if p.suffix.lower() in (".csv", ".tsv", ".txt") and not p.name.startswith(".")]
    if not csvs:
        raise SystemExit(f"no data files in {d}")
    t = Path(d / tree) if tree else next((p for p in csvs if any(k in p.name.lower() for k in ("tree", "taxonom", "categor"))), None)
    rest = [p for p in csvs if p != t]
    l = Path(d / lines) if lines else max(rest, key=lambda p: p.stat().st_size)
    return l, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="your name, used in the output file names")
    ap.add_argument("--a2"), ap.add_argument("--b2"), ap.add_argument("--c")
    ap.add_argument("--c-file"), ap.add_argument("--c-tree"), ap.add_argument("--c-currency", default="USD")
    a = ap.parse_args()
    sub = ROOT / "submission"
    sub.mkdir(exist_ok=True)
    ok = {}
    if a.a2:
        lab = ["--labels-dir", "data/company_A"] if (ROOT / "data/company_A").exists() else []
        ok["A2"] = run([sys.executable, "-m", "spendcube", "--config", "configs/company_A.yaml", "--data-dir", a.a2,
                        *lab, "--out", sub / f"{a.name}__A2.csv"])
    if a.b2:
        lab = ["--labels-dir", "data/company_B"] if (ROOT / "data/company_B").exists() else []
        ok["B2"] = run([sys.executable, "-m", "spendcube", "--config", "configs/company_B.yaml", "--data-dir", a.b2,
                        *lab, "--out", sub / f"{a.name}__B2.csv"])
    if a.c:
        cd = Path(a.c).resolve()
        lines, tree = pick_files(cd, a.c_file, a.c_tree)
        print(f"C: lines file = {lines.name}, tree file = {tree.name if tree else None}")
        cfg_out = ROOT / "configs" / "company_C.yaml"
        ok["C-config"] = run([sys.executable, "scripts/make_config.py", "--data-dir", cd, "--file", lines.relative_to(cd),
                              "--taxonomy", tree.relative_to(cd), "--name", "company_C", "--currency", a.c_currency, "--out", cfg_out])
        if ok["C-config"]:
            ok["C"] = run([sys.executable, "-m", "spendcube", "--config", cfg_out, "--no-labels", "--out", sub / f"{a.name}__C.csv"])
            shutil.copy(cfg_out, sub / "company_C.yaml")
    print("\nSUMMARY:", ok)
    print("Now check submission/ then: git add submission && git commit -m 'blind test' && git push")


if __name__ == "__main__":
    main()
