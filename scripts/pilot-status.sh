#!/usr/bin/env bash
# Resumo do piloto: está vivo? em que ponto está? quanto falta?
#
#   scripts/pilot-status.sh [runs/algum-run]    (sem argumento: o run mais recente)
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=${1:-$(ls -dt runs/*/ 2>/dev/null | head -1)}
OUT=${OUT%/}
if [ ! -d "$OUT" ]; then
  echo "nenhum run encontrado em runs/"
  exit 1
fi

echo "run     : $OUT"

pid=$(pgrep -u "$USER" -f "wrag.pilot --gpu" | head -1)
if [ -n "$pid" ]; then
  echo "estado  : RODANDO (pid $pid, há $(ps -o etime= -p "$pid" | tr -d ' '))"
else
  echo "estado  : PARADO"
  [ -f "$OUT/status.json" ] && echo "          $(cat "$OUT/status.json")"
  [ -f "$OUT/error.txt" ]   && echo "          erro: $(cat "$OUT/error.txt")"
fi

echo "progresso:"
prog=$(grep -hE "perguntas concluídas|ritmo observado" "$OUT/benchmark.log" 2>/dev/null | tail -2)
if [ -n "$prog" ]; then
  echo "$prog" | sed 's/^/  /'
else
  echo "  (ainda subindo o servidor / montando índices; o progresso por pergunta vem depois)"
fi

# Durante a indexação (OpenIE, grafo) o benchmark.log fica mudo por muitos
# minutos: o laço usa tqdm, desligado por WRAG_NO_PROGRESS. Quem prova que está
# vivo nessa fase é o vLLM atendendo requisições.
echo "atividade:"
tp=$(grep "loggers.py" "$OUT/vllm.log" 2>/dev/null | tail -1 | sed 's/.*Engine 000: //' | cut -c1-110)
calls=$(grep -c "chat/completions" "$OUT/vllm.log" 2>/dev/null || echo 0)
if [ -n "$tp" ]; then
  echo "  $tp"
  echo "  $calls chamadas ao LLM até agora"
else
  echo "  (vLLM ainda não atendeu nada)"
fi

echo "log     :"
grep -v "HTTP Request" "$OUT/benchmark.log" 2>/dev/null | tail -2 | cut -c1-150 | sed 's/^/  /'

echo "gpu     :"
# Só o vLLM deste piloto; a máquina é compartilhada e o resto é de outras pessoas.
# A memória fica no EngineCore, um filho do APIServer, não no APIServer em si:
# casar por grupo de processos pega os dois, além do worker (embeddings).
vpid=$(pgrep -u "$USER" -f "vllm.entrypoints.cli.main serve" | head -1)
vpgid=$([ -n "$vpid" ] && cut -d' ' -f5 "/proc/$vpid/stat" 2>/dev/null)
wpid=$(pgrep -u "$USER" -f "wrag.pilot --gpu .* --worker" | head -1)
found=""
while IFS=, read -r gpid gmem; do
  gpid=$(echo "$gpid" | tr -d ' ')
  pgid=$(cut -d' ' -f5 "/proc/$gpid/stat" 2>/dev/null)
  if [ -n "$vpgid" ] && [ "$pgid" = "$vpgid" ]; then
    echo "  vllm     pid $gpid:$gmem"; found=1
  elif [ -n "$wpid" ] && [ "$gpid" = "$wpid" ]; then
    echo "  worker   pid $gpid:$gmem"; found=1
  fi
done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
[ -n "$found" ] || echo "  (nada seu na GPU ainda; o vLLM ainda está carregando)"

exit 0
