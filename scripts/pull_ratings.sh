#!/usr/bin/env bash
# Snapshot the rater server's ratings (mini PC) to data/ratings/ratings.db; sqlite's backup is safe while it serves.
set -euo pipefail
mkdir -p data/ratings
ssh minipc 'python3 -c "import sqlite3; sqlite3.connect(\"/home/shay/nuance-rater/db/ratings.db\").backup(sqlite3.connect(\"/tmp/nuance-ratings.db\"))"'
scp -q minipc:/tmp/nuance-ratings.db data/ratings/ratings.db
ssh minipc 'rm /tmp/nuance-ratings.db'
python3 -c "import sqlite3; print(sqlite3.connect('data/ratings/ratings.db').execute('SELECT rater, COUNT(*) FROM ratings WHERE score IS NOT NULL GROUP BY rater').fetchall())"
