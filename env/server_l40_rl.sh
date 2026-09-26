#!/usr/bin/env bash
# venv-2: EasyR1 / verl (GRPO)。⚠️ 与 venv-1 严格分离 (K2)。
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/ml-user/workdir/project-2}"
cd "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export UV_HTTP_TIMEOUT=600

echo "[env-rl] 创建 venv-2 (EasyR1 + vLLM)"
uv venv --python 3.11 .venv-rl
uv pip install --python .venv-rl/bin/python \
  --index-url https://pypi.org/simple \
  -r env/requirements-server-rl.txt

echo "[env-rl] 拉取 EasyR1 并钉住 commit"
mkdir -p third_party && cd third_party
[[ -d EasyR1 ]] || git clone https://github.com/hiyouga/EasyR1.git
cd EasyR1 && git checkout "${EASYR1_COMMIT:-main}"

echo "[env-rl] 完成。⚠️ EasyR1 官方 Docker 为 hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0"
echo "          若裸装 vLLM 版本对不上, 优先改用该镜像 (风险 K3)。"
echo "          自检: bash env/check_env.py --stage rl"
