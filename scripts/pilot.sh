#!/usr/bin/env bash
# Sobe o piloto vLLM numa sessão tmux, desacoplado do terminal.
#
#   scripts/pilot.sh [nome-do-run]
#
# Acompanhar ao vivo : tmux attach -t pilot   (sair sem matar: Ctrl+B, depois D)
# Situação resumida  : scripts/pilot-status.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${1:-qwen14b-a100-$(date +%Y%m%d-%H%M)}
OUT="runs/$RUN"
GPU=${GPU:-4}
HOURS=${HOURS:-6.5}
PORT=${PORT:-8085}
SESSION=pilot

if [ -e "$OUT" ]; then
  echo "erro: $OUT já existe; o piloto recusa sobrescrever resultados. Escolha outro nome." >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "erro: já existe a sessão tmux '$SESSION'. Veja com: tmux attach -t $SESSION" >&2
  exit 1
fi

# Um vLLM órfão de um run interrompido continua segurando a GPU e a porta.
if pgrep -u "$USER" -f "vllm.entrypoints.cli.main serve.*--port $PORT" >/dev/null; then
  echo "encerrando vLLM anterior na porta $PORT"
  pkill -u "$USER" -f "vllm.entrypoints.cli.main serve.*--port $PORT" || true
  sleep 8
fi

tmux new-session -d -s "$SESSION" "
  cd '$PWD' &&
  .venv-bench/bin/python -m wrag.pilot --gpu $GPU \
    --vllm-python '$PWD/.venv-vllm/bin/python' \
    --port $PORT --hours $HOURS --output '$OUT' 2>&1 | tee 'runs/$RUN.out'
  echo
  echo '=== piloto terminou; a janela fica aberta. Ctrl+B depois D para sair. ==='
  exec bash"

echo "piloto rodando   : $OUT"
echo "  ver ao vivo    : tmux attach -t $SESSION      (sair: Ctrl+B, depois D)"
echo "  situação       : scripts/pilot-status.sh"
echo "  matar          : tmux kill-session -t $SESSION"
