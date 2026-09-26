#!/usr/bin/env bash
# 内存诊断采样器 —— 崩溃复盘用, 只读不写任何训练状态
# 每 10s 打一行: 容器 cgroup 内存总量 / anon-file-slab 拆分 / python 进程数 / top RSS
CGROUP=/sys/fs/cgroup
MB=1048576

while true; do
  t=$(date +%H:%M:%S)
  cur=$(( $(cat $CGROUP/memory.current) / MB ))
  read -r anon file slab < <(awk '/^anon |^file |^slab /{printf "%d ", $2/1048576}' $CGROUP/memory.stat)
  slab=${slab:-0}
  # slab_reclaimable 单列出来, 它也算在 cgroup 里
  slabr=$(( $(awk '/^slab_reclaimable /{print $2}' $CGROUP/memory.stat) / MB ))
  npy=$(pgrep -cf "python" || true)
  echo "$t TOTAL=${cur}M ANON=${anon}M FILE=${file}M SLAB=${slab}M SLAB_RECLAIM=${slabr}M python_procs=${npy:-0}"
  # 按 RSS 前 12 名, 带 ppid 便于归到哪个 rank
  ps -eo rss,pid,ppid,comm --sort=-rss --no-headers 2>/dev/null | head -12 \
    | awk -v t="$t" '{printf "%s   RSS=%6.0fM pid=%-8s ppid=%-8s %s\n", t, $1/1024, $2, $3, $4}'
  echo "---"
  sleep 10
done
