#!/usr/bin/env bash
# Download Santander train_ver2.csv from Kaggle and build the transition cache.
# Run from the repository root:  bash prep/download_santander.sh
#
# Requires the Kaggle CLI (pip install kaggle), acceptance of the competition
# rules on kaggle.com, and Kaggle credentials in ~/.kaggle/kaggle.json or the
# KAGGLE_USERNAME / KAGGLE_KEY environment variables. Credentials are never
# read from or written to this repository.
set -e
cd "$(dirname "$0")/.."
DEST="${NEGCAND_ROOT:-.}/data/santander"
mkdir -p "$DEST"
if [ ! -f "$DEST/train_ver2.csv" ]; then
  kaggle competitions download -c santander-product-recommendation -f train_ver2.csv -p "$DEST"
  (cd "$DEST" && (unzip -o train_ver2.csv.zip 2>/dev/null || true) && rm -f train_ver2.csv.zip)
fi
ls -la "$DEST"
python -m prep.santander
