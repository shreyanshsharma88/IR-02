"""Ingestion: default BBC CSVs + uploaded CSV / JSON / JSONL / ZIP."""
from __future__ import annotations

import io
import json
import zipfile

import pandas as pd
import pytest

from textmining.data_loader import load_default, load_uploaded, split_frame


def test_load_default_shapes(dataset):
    docs, meta = dataset
    assert docs.shape == (2225, 2)
    assert list(docs.columns) == ["doc_id", "text"]
    assert len(meta) == 2225
    assert {"doc_id", "title", "category", "source", "period", "filename", "n_chars", "n_words"} <= set(meta.columns)


def test_text_and_metadata_logically_separate(dataset):
    docs, meta = dataset
    assert "text" not in meta.columns
    assert docs["doc_id"].is_unique and meta["doc_id"].is_unique
    assert (docs["doc_id"].values == meta["doc_id"].values).all()


def test_categories(dataset):
    meta = dataset[1]
    assert meta["category"].value_counts().to_dict() == {
        "sport": 511, "business": 510, "politics": 417, "tech": 401, "entertainment": 386}


def test_no_mojibake(dataset):
    docs, meta = dataset
    all_text = "".join(docs["text"]) + "".join(meta["title"])
    assert "Â" not in all_text and "â€" not in all_text
    assert "£" in all_text


def test_no_empty_documents(dataset):
    assert (dataset[0]["text"].fillna("").str.strip().str.len() > 0).all()


def test_csv_upload_with_id_and_label():
    df = pd.DataFrame({"id": [10, 11], "content": ["first doc text", "second doc"], "label": ["x", "y"]})
    docs, meta = load_uploaded("my.csv", df.to_csv(index=False).encode())
    assert list(docs.columns) == ["doc_id", "text"]
    assert docs["doc_id"].tolist() == ["10", "11"]
    assert docs["text"].tolist() == ["first doc text", "second doc"]
    assert meta["category"].tolist() == ["x", "y"]
    assert "text" not in meta.columns
    assert meta["n_words"].tolist() == [3, 2]
    assert meta["title"].tolist() == ["first doc text", "second doc"]


def test_csv_upload_without_id_or_label():
    df = pd.DataFrame({"Text": ["alpha beta", None, "gamma"]})
    docs, meta = load_uploaded("x.CSV", df.to_csv(index=False).encode())
    assert docs["doc_id"].tolist() == ["D00000", "D00001", "D00002"]
    assert docs["text"].tolist() == ["alpha beta", "", "gamma"]
    assert "category" not in meta.columns


def test_csv_duplicate_ids_made_unique():
    df = pd.DataFrame({"doc_id": ["a", "a", "b"], "text": ["t1", "t2", "t3"]})
    docs, _ = load_uploaded("d.csv", df.to_csv(index=False).encode())
    assert docs["doc_id"].is_unique
    assert docs["doc_id"].tolist() == ["a_0", "a_1", "b_0"]


def test_csv_without_text_column_raises():
    with pytest.raises(ValueError, match="No text column"):
        load_uploaded("bad.csv", b"foo,bar\n1,2\n")


def test_json_list_upload():
    data = json.dumps([{"doc_id": "a", "body": "hello world", "topic": "t1"},
                       {"doc_id": "b", "body": "bye", "topic": "t2"}]).encode()
    docs, meta = load_uploaded("x.json", data)
    assert docs["doc_id"].tolist() == ["a", "b"] and docs["text"].tolist() == ["hello world", "bye"]
    assert meta["category"].tolist() == ["t1", "t2"]


def test_json_documents_key_upload():
    data = json.dumps({"documents": [{"text": "one"}, {"text": "two"}]}).encode()
    docs, _ = load_uploaded("x.json", data)
    assert docs["text"].tolist() == ["one", "two"]


def test_jsonl_upload():
    data = b'{"text": "line one", "category": "a"}\n{"text": "line two", "category": "b"}\n'
    docs, meta = load_uploaded("x.jsonl", data)
    assert docs["text"].tolist() == ["line one", "line two"]
    assert meta["category"].tolist() == ["a", "b"]


def test_single_line_jsonl_upload():
    docs, meta = load_uploaded("one.jsonl", b'{"text": "only one record", "category": "a"}\n')
    assert docs["text"].tolist() == ["only one record"]
    assert meta["category"].tolist() == ["a"]


def _zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    return buf.getvalue()


def test_zip_folder_per_category():
    data = _zip({
        "bbc/sport/001.txt": "Team wins\n\nThe team won the match.",
        "bbc/tech/002.txt": "New phone\n\nA new phone was launched.",
        "bbc/tech/readme.md": "ignored",
        "__MACOSX/bbc/tech/._002.txt": "junk",
    })
    docs, meta = load_uploaded("bbc.zip", data)
    assert sorted(docs["doc_id"]) == ["sport/001", "tech/002"]
    m = meta.set_index("doc_id")
    assert m.loc["sport/001", "category"] == "sport"
    assert m.loc["sport/001", "title"] == "Team wins"
    assert m.loc["tech/002", "source"] == "bbc.zip"
    d = docs.set_index("doc_id")
    assert "The team won the match." in d.loc["sport/001", "text"]
    assert "Team wins" not in d.loc["sport/001", "text"]


def test_zip_flat_is_unlabelled():
    docs, meta = load_uploaded("flat.zip", _zip({"a.txt": "Title\nbody text"}))
    assert meta["category"].tolist() == ["unlabelled"]
    assert docs["text"].tolist() == ["body text"]


def test_zip_without_txt_raises():
    with pytest.raises(ValueError, match="no .txt"):
        load_uploaded("empty.zip", _zip({"a.csv": "x"}))


@pytest.mark.parametrize("name", ["data.xlsx", "data.pdf", "noext"])
def test_unsupported_format_raises(name):
    with pytest.raises(ValueError, match="Unsupported"):
        load_uploaded(name, b"whatever")


def test_split_frame_explicit_columns():
    df = pd.DataFrame({"key": ["k1"], "article": ["some text"], "desk": ["news"]})
    docs, meta = split_frame(df, text_col="article", id_col="key", label_col="desk")
    assert docs.iloc[0].tolist() == ["k1", "some text"]
    assert meta["category"].tolist() == ["news"]


def test_load_default_function_directly():
    docs, meta = load_default()
    assert len(docs) == len(meta) == 2225


def test_json_columnar_dict_upload():
    data = json.dumps({"text": ["one", "two"], "category": ["a", "b"]}).encode()
    docs, meta = load_uploaded("cols.json", data)
    assert docs["text"].tolist() == ["one", "two"] and meta["category"].tolist() == ["a", "b"]
