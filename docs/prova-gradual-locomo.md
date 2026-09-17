# Prova gradual e leitura guiada: rodada focada no LoCoMo

Esta rodada compara a melhor condição anterior (`evidence`) com uma continuação
que conserva a busca por fronteira, aquisição e empacotamento de provas. São
**duas condições pareadas**, não as nove ablações anteriores. O controle é
reexecutado com o mesmo código da nova versão para que o comparador valide o
hash do código, os dados, o modelo e as perguntas.

## Mudança proposta

O verificador de planos agora dispõe de três graus:

- `full`: o plano preserva as condições da pergunta. Pode selecionar uma
  testemunha estrutural como antes.
- `partial`: o plano é relevante, mas requer conferência de uma condição no
  texto. Se nenhum plano completo vencer, uma testemunha que caiba no contexto
  pode fornecer uma **hipótese provisória** ao leitor. Pelo menos uma das cinco
  posições fica reservada para a recuperação híbrida. A hipótese não recebe
  resposta estrutural certificada nem score de risco de uma prova completa.
- `mismatch` ou checagem indisponível: o plano não é promovido. Uma saída
  filtrada, inválida ou ambígua também não vira prova.

O prompt explicita a direção `sujeito --relação--> objeto`, para evitar o
falso veto visto em “quem apoia Calvin”. A avaliação gradual não observa
respostas nem passagens ouro. O leitor novo recebe somente fatos de
testemunhas **inteiramente contidas** nas passagens entregues. Ele deve conferir
cada relação e cada condição no texto original; os fatos extraídos são pistas,
não autoridade. Quando não há prova selecionada, usa o leitor anterior.
Esta rodada não liga `--active-operators`, pois aquela condição reduziu o F1
oficial de contagens na rodada anterior.

## Rodar no servidor

No diretório do projeto, depois de enviar o código:

```bash
GPU=4 bash scripts/run-locomo-soft-proof.sh runs/locomo-soft-proof-01
```

O padrão executa as dez conversas nas condições `evidence` e `soft-proof`, em
sequência, reutilizando uma GPU, porta e cache. Se o cache anterior estiver em
`runs/locomo-proof-ablation-01/cache`, ele é reutilizado; caso contrário o
script cria um cache sob a nova saída. Reexecute **o mesmo comando e diretório**
para retomar. `HOURS` define o prazo por tentativa (18 por padrão),
`MAX_RESUMES` o número de tentativas, e `PORT`, `CACHE_DIR`, `BENCH_PYTHON`,
`VLLM_PYTHON` e `EXISTING_SERVER=1` podem ser ajustados pelo ambiente.

Para ensaiar apenas a primeira conversa, em um diretório separado:

```bash
LOCOMO_CONVERSATION=0 GPU=4 bash scripts/run-locomo-soft-proof.sh runs/locomo-soft-proof-smoke
```

O comparador escreve `ablation.md` e `ablation.json` na raiz da saída. Além do
F1 oficial e multi-hop, registra recuperação completa de evidência, ganhos e
perdas pareados, disparo de provas completas e proporção de hipóteses
provisórias. Os JSONL preservam `classe_prova`, `prova_provisoria`, motivos da
checagem e o mapa de evidências realmente mostrado ao leitor. A meta de F1
multi-hop 0,45 continua sendo hipótese experimental; nenhum resultado da nova
versão foi medido no servidor ainda.
