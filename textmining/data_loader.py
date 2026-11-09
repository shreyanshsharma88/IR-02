"""Ingestion: CSV, JSON, or a ZIP of .txt files (folder-per-category or flat).
Returns (documents_df[doc_id, text], metadata_df[doc_id, ...])."""
from __future__ import annotations

import io
import json
import os
import zipfile

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TEXT_CANDIDATES = ["text", "content", "body", "document", "review", "abstract"]
LABEL_CANDIDATES = ["category", "label", "class", "topic"]


def load_default():
    docs = pd.read_csv(os.path.join(DATA_DIR, "documents.csv"))
    meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    return docs, meta


def split_frame(df: pd.DataFrame, text_col: str | None = None, id_col: str | None = None, label_col: str | None = None):
    df = df.copy()
    cols_l = {c.lower(): c for c in df.columns}
    text_col = text_col or next((cols_l[c] for c in TEXT_CANDIDATES if c in cols_l), None)
    if text_col is None:
        raise ValueError("No text column found; expected one of " + ", ".join(TEXT_CANDIDATES))
    if id_col is None or id_col not in df.columns:
        id_col = cols_l.get("doc_id") or cols_l.get("id")
    if id_col is None:
        df.insert(0, "doc_id", [f"D{i:05d}" for i in range(len(df))]); id_col = "doc_id"
    label_col = label_col or next((cols_l[c] for c in LABEL_CANDIDATES if c in cols_l), None)
    df = df.rename(columns={id_col: "doc_id", text_col: "text"})
    if label_col and label_col != "category":
        df = df.rename(columns={label_col: "category"})
    df["doc_id"] = df["doc_id"].astype(str)
    if df["doc_id"].duplicated().any():  # enforce unique ids
        df["doc_id"] = df["doc_id"] + "_" + df.groupby("doc_id").cumcount().astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    docs = df[["doc_id", "text"]]
    meta = df.drop(columns=["text"])
    if "title" not in meta:
        meta["title"] = df["text"].str.slice(0, 70)
    meta["n_chars"] = df["text"].str.len()
    meta["n_words"] = df["text"].str.split().str.len()
    return docs, meta


def load_uploaded(name: str, data: bytes):
    lname = name.lower()
    if lname.endswith(".csv"):
        return split_frame(pd.read_csv(io.BytesIO(data)))
    if lname.endswith(".json") or lname.endswith(".jsonl"):
        txt = data.decode("utf-8", errors="ignore")
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict) and "documents" not in obj and not any(isinstance(v, list) for v in obj.values()):
                obj = [obj]  # a single record (e.g. one-line JSONL) -> one-row frame
            df = pd.DataFrame(obj if isinstance(obj, list) else obj.get("documents", obj))
        except json.JSONDecodeError:
            df = pd.read_json(io.StringIO(txt), lines=True)
        return split_frame(df)
    if lname.endswith(".zip"):
        rows = []
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".txt") or "__MACOSX" in info.filename:
                    continue
                parts = info.filename.strip("/").split("/")
                raw = z.read(info).decode("utf-8", errors="ignore")
                lines = raw.splitlines()
                rows.append({"doc_id": "/".join(parts[-2:]).replace(".txt", ""),
                             "title": lines[0].strip() if lines else "",
                             "category": parts[-2] if len(parts) >= 2 else "unlabelled",
                             "source": name, "text": "\n".join(lines[1:]) if len(lines) > 1 else raw})
        if not rows:
            raise ValueError("ZIP contains no .txt files")
        return split_frame(pd.DataFrame(rows))
    raise ValueError("Unsupported format: use CSV, JSON or ZIP of .txt files")
