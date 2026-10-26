"""Create the course submission zip: notebooks/, modules/, reports/, features/ (+ configs/, README, requirements).

Usage:  python scripts/make_submission.py [--skip-features] [--out reports/submission.zip]
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ["notebooks", "modules", "configs", "reports", "features", "docs"]
INCLUDE_FILES = ["README.md", "requirements.txt", "LICENSE", "pytest.ini"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".git", "data"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-features", action="store_true", help="do not include features/ (smaller zip)")
    ap.add_argument("--out", default="submission/ml_assignment.zip")
    args = ap.parse_args()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    n, size = 0, 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d in INCLUDE_DIRS:
            if d == "features" and args.skip_features:
                continue
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file() and not (set(p.parts) & SKIP_PARTS) and p.resolve() != out.resolve() and p.suffix != ".zip":
                    z.write(p, p.relative_to(ROOT)); n += 1; size += p.stat().st_size
        for f in INCLUDE_FILES:
            if (ROOT / f).exists():
                z.write(ROOT / f, f); n += 1
    print(f"{out}: {n} files, {size / 1e6:.1f} MB uncompressed, {out.stat().st_size / 1e6:.1f} MB zipped")


if __name__ == "__main__":
    main()
