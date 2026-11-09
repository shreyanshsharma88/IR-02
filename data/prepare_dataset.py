"""Convert the raw BBC folder into data/documents.csv and data/metadata.csv.

Usage: python data/prepare_dataset.py [path/to/bbc]"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "raw_download", "bbc")


def main():
    docs, meta = [], []
    for category in sorted(os.listdir(RAW)):
        cdir = os.path.join(RAW, category)
        if not os.path.isdir(cdir):
            continue
        for fname in sorted(os.listdir(cdir)):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(cdir, fname)
            try:  # files are UTF-8 (e.g. '£'); latin-1 only as a fallback
                with open(path, encoding="utf-8") as f:
                    raw = f.read()
            except UnicodeDecodeError:
                with open(path, encoding="latin-1") as f:
                    raw = f.read()
            lines = [l for l in raw.splitlines()]
            title = lines[0].strip() if lines else ""
            body = "\n".join(lines[1:]).strip()
            doc_id = f"{category[:3].upper()}-{fname[:-4]}"
            docs.append({"doc_id": doc_id, "text": body})
            meta.append({
                "doc_id": doc_id, "title": title, "category": category,
                "source": "BBC News", "period": "2004-2005",
                "filename": f"{category}/{fname}",
                "n_chars": len(raw), "n_words": len(raw.split()),
            })
    with open(os.path.join(HERE, "documents.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id", "text"]); w.writeheader(); w.writerows(docs)
    with open(os.path.join(HERE, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0].keys())); w.writeheader(); w.writerows(meta)
    print(f"wrote {len(docs)} documents, {sum(m['n_chars'] for m in meta)/1e6:.2f} M chars")


if __name__ == "__main__":
    main()
