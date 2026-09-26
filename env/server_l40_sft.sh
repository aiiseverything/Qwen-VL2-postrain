#!/usr/bin/env bash
# venv-1: LLaMA-Factory (SFT + DPO)。⚠️ 不要在此 venv 里装 vLLM (K2)。
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/ml-user/workdir/project-2}"
cd "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export UV_HTTP_TIMEOUT=600

echo "[env-sft] 创建 venv-1 (LLaMA-Factory)"
uv venv --python 3.11 .venv-lf
uv pip install --python .venv-lf/bin/python \
  --index-url https://pypi.org/simple \
  -r env/requirements-server-sft.txt

echo "[env-sft] 拉取 LLaMA-Factory 并钉住 commit"
mkdir -p third_party && cd third_party
[[ -d LlamaFactory ]] || git clone https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory && git checkout "${LLAMAFACTORY_COMMIT:-main}"
uv pip install --python "$PROJECT_ROOT/.venv-lf/bin/python" -e . --no-deps

echo "[env-sft] 完成。自检: bash env/check_env.py --stage sft"
