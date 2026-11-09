"""Assignment compliance: no forbidden ML libraries anywhere + all deliverables present."""
from __future__ import annotations

import ast
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_DIRS = {".venv", "venv", "dist", "__pycache__", ".git", ".claude", "node_modules", ".pytest_cache"}
FORBIDDEN = ("sklearn", "scikit_learn", "gensim", "scipy.cluster", "scipy.spatial.distance", "scipy.sparse",
             "nltk.cluster", "nltk.classify", "torch", "tensorflow", "keras", "transformers",
             "sentence_transformers", "spacy", "textblob", "rank_bm25", "hdbscan", "umap", "xgboost", "lightgbm")


def project_py_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for f in filenames:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def imported_modules(path):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module
            for a in node.names:
                yield f"{node.module}.{a.name}"
        elif isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) in (
                "__import__", "import_module") and node.args and isinstance(node.args[0], ast.Constant):
            yield str(node.args[0].value)


def _forbidden(mod: str) -> bool:
    return any(mod == f or mod.startswith(f + ".") for f in FORBIDDEN)


FILES = sorted(project_py_files())


def test_scan_covers_project():
    rel = {os.path.relpath(p, ROOT) for p in FILES}
    assert {"app.py", "experiments.py", "textmining/tfidf.py", "textmining/naive_bayes.py",
            "textmining/kmeans.py", "textmining/metrics.py", "ui/state.py"} <= rel


@pytest.mark.parametrize("path", FILES, ids=lambda p: os.path.relpath(p, ROOT))
def test_no_forbidden_imports(path):
    bad = sorted({m for m in imported_modules(path) if _forbidden(m)})
    assert not bad, f"{os.path.relpath(path, ROOT)} imports forbidden module(s): {bad}"


def test_sklearn_not_in_requirements():
    for name in ("requirements.txt", "requirements-dev.txt"):
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            lines = [l.split("#")[0].strip().lower() for l in open(p, encoding="utf-8")]
            assert not any(l.startswith(("scikit-learn", "sklearn", "gensim")) for l in lines), name


def test_sklearn_not_installed():
    import importlib.util
    assert importlib.util.find_spec("sklearn") is None


@pytest.mark.parametrize("rel", ["app.py", "README.md", "requirements.txt", "report/Report.pdf",
                                 "data/documents.csv", "data/metadata.csv", "data/prepare_dataset.py",
                                 "experiments.py"])
def test_required_deliverable_exists(rel):
    p = os.path.join(ROOT, rel)
    assert os.path.isfile(p) and os.path.getsize(p) > 0, rel


def test_report_is_pdf():
    with open(os.path.join(ROOT, "report", "Report.pdf"), "rb") as f:
        assert f.read(5) == b"%PDF-"


def test_requirements_list_runtime_deps():
    reqs = open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8").read().lower()
    for pkg in ("streamlit", "pandas", "numpy", "nltk", "plotly"):
        assert pkg in reqs


def test_contribution_sheet_headers():
    openpyxl = pytest.importorskip("openpyxl")
    sheets = [f for f in os.listdir(ROOT) if f.endswith("_Contribution.xlsx")]
    assert sheets, "missing *_Contribution.xlsx"
    ws = openpyxl.load_workbook(os.path.join(ROOT, sheets[0])).active
    headers = [c.value for c in ws[1]][:3]
    assert headers == ["Student Registration Number", "Name", "Percentage of contribution out of 100%"]
