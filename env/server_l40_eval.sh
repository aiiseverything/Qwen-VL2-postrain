#!/usr/bin/env bash
# venv-3: VLMEvalKit (评测)。⚠️ 同样独立 (K2)。
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/ml-user/workdir/project-2}"
cd "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export UV_HTTP_TIMEOUT=600

echo "[env-eval] 创建 venv-3 (VLMEvalKit)"
uv venv --python 3.11 .venv-eval
uv pip install --python .venv-eval/bin/python \
  --index-url https://pypi.org/simple \
  -r env/requirements-server-eval.txt

echo "[env-eval] 拉取 VLMEvalKit 并钉住 commit"
mkdir -p third_party && cd third_party
[[ -d VLMEvalKit ]] || git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit && git checkout "${VLMEVALKIT_COMMIT:-main}"
uv pip install --python "$PROJECT_ROOT/.venv-eval/bin/python" -e . --no-deps

echo "[env-eval] 完成。自检: bash env/check_env.py --stage eval"
