#!/usr/bin/env bash
# ============================================================================
# 等 MME 跑完 → 带修好的判分参数重启评测（MME 会被 DONE 标记跳过）
# ----------------------------------------------------------------------------
# 背景：本轮评测的 --judge-base-url 与 .env 的完整端点**双重拼接**
#   （run.py:396 会自己加 /chat/completions），导致判分全失败
#   → MME 不受影响（规则判分），但 MathVista / Video-MME 会全 0。
# 策略：让 MME 跑完并留下 DONE（新逻辑：DONE 按"模型集哈希"命名），
#       然后重启 —— MME 命中 DONE 直接跳过，只有剩下两个 benchmark 重跑。
# ⚠️ 重启间隙要**先停掉 base 线的守护**，否则它可能在这 1-2 秒里抢跑 GPU。
# ============================================================================
set -uo pipefail
PROJ=/home/ml-user/workdir/project-2
LOG="$PROJ/logs/eval_full.log"

ts() { date +'%F %T'; }
echo "[$(ts)] 等 MME 完成..." >> "$PROJ/logs/eval_restart.log"
for _ in $(seq 1 720); do          # 最多等 12 小时
  grep -qa "✅ MME 完成" "$LOG" && break
  grep -qa "❌ MME" "$LOG" && { echo "[$(ts)] MME 失败, 仍然重启" >> "$PROJ/logs/eval_restart.log"; break; }
  sleep 60
done
echo "[$(ts)] MME 阶段结束, 开始重启评测" >> "$PROJ/logs/eval_restart.log"

# 1) 停 base 线守护（它在等 GPU 空闲，重启间隙可能误判）
for p in $(ps -eo pid,cmd --no-headers | grep "run_chain_bas[e]" | grep -v grep | awk '{print $1}'); do
  kill "$p" 2>/dev/null && echo "[$(ts)] 暂停 base 线守护 $p" >> "$PROJ/logs/eval_restart.log"
done
# 2) 停当前评测（run_all.sh + vlmeval run.py）
for pat in "eval/run_all.s[h]" "VLMEvalKit/run.p[y]"; do
  for p in $(ps -eo pid,cmd --no-headers | grep -E "$pat" | grep -v grep | awk '{print $1}'); do
    kill "$p" 2>/dev/null
  done
done
sleep 8
for p in $(ps -eo pid,cmd --no-headers | grep -E "VLMEvalKit/run.p[y]" | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
sleep 5

# 3) 带修好的参数重启（不再传 --judge-base-url）
cd "$PROJ"
MODELS="pretrained sft dpo sft_ep3 grpo" DATA="MME MathVista_MINI Video-MME_16frame" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 JUDGE_MODEL=gpt-5.5 \
  setsid nohup bash scripts/eval/run_all.sh > logs/eval_full2.log 2>&1 < /dev/null &
echo "[$(ts)] 评测已重启 (日志 logs/eval_full2.log)" >> "$PROJ/logs/eval_restart.log"
sleep 20
grep -qa "跳过: MME" logs/eval_full2.log && echo "[$(ts)] ✓ MME 已按预期跳过" >> "$PROJ/logs/eval_restart.log"

# 4) 恢复 base 线守护
setsid nohup bash scripts/train/run_chain_base.sh > logs/chain_base.log 2>&1 < /dev/null &
echo "[$(ts)] base 线守护已恢复" >> "$PROJ/logs/eval_restart.log"
