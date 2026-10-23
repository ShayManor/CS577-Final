#!/usr/bin/env bash
set -euo pipefail
rsync -az csuser@100.88.24.43:~/CS577-Final/data/batches/$1/ data/batches/$1/
