"""Helper for building clean notebooks from a (kind, source) cell list."""
import json
import pathlib

REPO_SLUG = "choilab-jefferson/medimage"

KERNEL_META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.13.13",
    },
}


def badge(filename):
    url = f"https://colab.research.google.com/github/{REPO_SLUG}/blob/main/{filename}"
    return (
        f'<a href="{url}" target="_parent">'
        '<img src="https://colab.research.google.com/assets/colab-badge.svg" '
        'alt="Open In Colab"/></a>'
    )


SETUP = '''\
# --- Setup: Google Colab (primary) and local checkout (testing) ---------------
REPO_URL = "https://github.com/choilab-jefferson/medimage.git"
REPO_DIR = "medimage"

import os
import pathlib
import subprocess
import sys

try:
    import google.colab  # noqa: F401

    IN_COLAB = True
except ModuleNotFoundError:
    IN_COLAB = False

if IN_COLAB:
    # On Colab the repository is not present yet, so clone it and install the
    # few packages that are not part of the default runtime.
    if not pathlib.Path(REPO_DIR).is_dir():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
    os.chdir(REPO_DIR)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "pydicom", "remotezip"],
        check=True,
    )
else:
    # Locally the notebook already lives inside the repository; walk up until
    # we find the module that the notebooks import.
    root = pathlib.Path.cwd().resolve()
    while not (root / "medimage_data.py").is_file() and root != root.parent:
        root = root.parent
    os.chdir(root)

print("Running on Colab:", IN_COLAB)
print("Working directory:", pathlib.Path.cwd().name)
'''

UNPACK = '''\
import zipfile


def unpack(archive, dest):
    """Extract a zip archive once and return the destination directory."""
    archive, dest = pathlib.Path(archive), pathlib.Path(dest)
    if not dest.is_dir():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    return dest
'''


def _existing_outputs(path):
    """Map each executed code cell's source to its outputs, in file order.

    Regenerating a chapter would otherwise discard the figures the committed
    notebooks carry. A code cell whose source is unchanged still has the same
    outputs, so those are worth keeping; one whose source changed does not, and
    is deliberately left empty rather than shown with a stale result.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        return {}
    try:
        nb = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    kept = {}
    for cell in nb.get("cells", []):
        executed = cell.get("execution_count") is not None or cell.get("outputs")
        if cell.get("cell_type") != "code" or not executed:
            continue
        key = "".join(cell["source"]).rstrip("\n")
        kept.setdefault(key, []).append(cell)
    return kept


def build(path, cells, title=None):
    """Write cells (list of ("md"|"code", source)) to `path` as a notebook.

    Outputs already stored in `path` are carried over to code cells whose
    source is byte-identical, so editing prose does not strip the figures.
    """
    carried = _existing_outputs(path)
    out = []
    reused = 0
    for n, (kind, src) in enumerate(cells):
        src = src.rstrip("\n")
        cell_id = f"cell-{n:03d}"
        if kind == "md":
            out.append(
                {"cell_type": "markdown", "id": cell_id, "metadata": {},
                 "source": src.splitlines(keepends=True)}
            )
        else:
            previous = carried[src].pop(0) if carried.get(src) else {}
            reused += bool(previous)
            out.append(
                {
                    "cell_type": "code",
                    "execution_count": previous.get("execution_count"),
                    "id": cell_id,
                    # Carried verbatim so that re-running a generator after a
                    # prose edit is a no-op for every cell it did not touch.
                    "metadata": previous.get("metadata", {}),
                    "outputs": previous.get("outputs", []),
                    "source": src.splitlines(keepends=True),
                }
            )
    nb = {"cells": out, "metadata": dict(KERNEL_META), "nbformat": 4, "nbformat_minor": 5}
    pathlib.Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    n_code = sum(c["cell_type"] == "code" for c in out)
    print(f"wrote {path}: {len(out)} cells ({n_code} code, {reused} outputs kept)")
