"""Helpers shared by the notebook builders."""
import nbformat as nbf

REPO_URL = "https://github.com/nguyenchuong01/ml-pipeline-multimodal.git"

BOOTSTRAP = '''# Setup: works on Colab (clones the repo) and locally / from the submitted zip (uses the local folders)
import os, sys, subprocess
from pathlib import Path

REPO_URL = "{repo}"

def _find_root():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "modules").is_dir() and (p / "configs").is_dir():
            return p

ROOT = _find_root()
if ROOT is None:  # fresh Colab runtime
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "ml-pipeline-multimodal"], check=True)
    ROOT = Path("ml-pipeline-multimodal").resolve()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
print("Project root:", ROOT)
'''


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def bootstrap():
    return code(BOOTSTRAP.format(repo=REPO_URL))


def save(cells, path, title):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": title, "provenance": []},
    }
    nbf.write(nb, path)
    print("wrote", path)
