#!/usr/bin/env bash
# Usage: ./make_zip.sh [G35]  ->  dist/G35_Assignment2.zip
set -euo pipefail
GID="${1:-G35}"
cd "$(dirname "$0")"
OUT="${GID}_Assignment2"
rm -rf "dist/$OUT" "dist/$OUT.zip"
mkdir -p "dist/$OUT/data"
cp -R app.py ui textmining tests experiments.py report screenshots README.md requirements.txt requirements-dev.txt pytest.ini make_zip.sh "dist/$OUT/"
cp data/prepare_dataset.py data/documents.csv data/metadata.csv "dist/$OUT/data/"
[ -f data/raw_download/bbc-fulltext.zip ] && cp data/raw_download/bbc-fulltext.zip "dist/$OUT/data/"
cp ./*_Contribution.xlsx "dist/$OUT/"
for f in ./*.mp4 ./*.mov; do [ -e "$f" ] && cp "$f" "dist/$OUT/"; done
find "dist/$OUT" \( -name "__pycache__" -o -name ".pytest_cache" \) -type d -prune -exec rm -rf {} +
find "dist/$OUT" -name ".DS_Store" -delete
(cd dist && zip -qr "$OUT.zip" "$OUT")
echo "built dist/$OUT.zip"
