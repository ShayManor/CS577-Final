#!/usr/bin/env bash
set -euo pipefail
rsync -az --delete --exclude data/ --exclude .git/ --exclude .idea/ --exclude .venv/ --exclude node_modules/ --exclude 'docs/' --exclude .superpowers/ ./ csuser@100.88.24.43:~/CS577-Final/
