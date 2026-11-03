#!/bin/bash
# every 10 s: time, MemAvailable, Cached, SwapFree (MB), memory pressure avg10, GPU MB in use, top RSS (MB); to NFS so the
# last lines survive a frozen node
o=/scratch/scholar/manors/nuance/logs/mem/$1-$(hostname -s).log
while true; do
  m=$(awk "/MemAvailable|^Cached|SwapFree|Dirty|Writeback:/{printf \"%s=%d \", \$1, \$2/1024}" /proc/meminfo)
  p=$(awk "/some/{print \$2}" /proc/pressure/memory 2>/dev/null)
  g=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc)
  r=$(ps -u manors -o rss= --sort=-rss | head -1)
  echo "$(date +%H:%M:%S) $m psi_$p gpuMB=$g topRSS=$((r/1024))" >> $o
  sleep 10
done
