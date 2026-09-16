# Pesquisa de provas: implementação e ablação LoCoMo

Esta revisão acrescenta um controlador de pesquisa em tempo de pergunta ao
WitnessRAG. Os quatro componentes são independentes e **desligados por padrão**;
o comportamento antigo continua sendo o controle da matriz.

## Componentes

1. `--active-frontier`: pesquisa a pergunta e os átomos dos planos por busca
   híbrida; mantém uma fronteira de passagens. Uma junção incompleta gera uma
   releitura dirigida pelas entidades já ligadas. Consultas de conjunto/contagem
   recebem uma rodada de busca adicional antes de encerrar, pois uma resposta
   encontrada não prova que a enumeração acabou.
2. `--active-obligations`: o Qwen confere se cada consulta compilada preserva as
   condições da pergunta original (entidades, qualificadores, tempo e operação).
   Saída inválida falha fechada. Planos rejeitados fornecem motivos ao
   replanejamento. A checagem avalia o plano, não a veracidade dos fatos.
3. `--active-context`: escolhe conjuntos **inteiros** de passagens de testemunhas
   que cabem em `top_k`; prioriza respostas distintas em perguntas de conjunto.
   O contexto restante vem do fallback híbrido.
4. `--active-operators`: usa leitura especializada em contagens e tempo. A
   resposta estrutural de `count` deixa de apresentar o tamanho do conjunto
   parcial como contagem exata e registra `limite_inferior_contagem`. O leitor
   prefere uma contagem explícita que satisfaz todos os qualificadores ou conta
   eventos/objetos distintos nas passagens, sem contar menções duplicadas.

`--verify-witnesses` é uma condição separada: exige trecho literal por átomo
antes de promover uma prova. Para `set` e `count`, a verificação avalia cada
membro, sem pedir a uma única testemunha que prove a completude da lista.

Esses controles **não** criam garantia de completude sobre o histórico textual.
O aterramento semântico, a interpretação da pergunta e a verificação LLM
continuam falíveis. Os diagnósticos registram planos, rejeições, buscas, páginas
lidas, provas selecionadas e custo de chamada por estágio.

## Executar no servidor

Depois de enviar esta revisão ao servidor, no diretório do projeto:

```bash
GPU=4 bash scripts/run-locomo-proof-ablation.sh runs/locomo-proof-ablation-01
```

O padrão executa as **dez conversas** das categorias 1 e 4, com
Qwen/Qwen2.5-14B-Instruct, BGE-M3, chunks de 2048 tokens, janelas de extração
de 512 tokens, `top_k=5`, o mesmo cache e nove condições sequenciais. Uma única
GPU e porta são reutilizadas. O script retoma conversas completas caso o prazo
de uma condição termine; para retomar após uma interrupção externa, execute o
mesmo comando com o **mesmo diretório de saída**.

Para verificar a instalação em uma conversa antes da matriz completa:

```bash
LOCOMO_CONVERSATION=0 GPU=4 bash scripts/run-locomo-proof-ablation.sh runs/locomo-proof-smoke
```

Variáveis úteis: `PORT`, `HOURS` (18 por tentativa, por padrão),
`MAX_RESUMES` (5), `CACHE_DIR`, `BENCH_PYTHON`, `VLLM_PYTHON`,
`EXISTING_SERVER=1`, `TOP_K`, `MAX_QUERY_PLANS`, `CHUNK_TOKENS` e
`IE_WINDOW_TOKENS`. Use a mesma configuração para toda a matriz. O script
verifica que as rodadas têm o mesmo código, modelo, dados, configuração base
e conjunto exato de perguntas antes de comparar. Os valores padrão reproduzem
a família B de múltiplos planos, sem verificação, no controle.

## Matriz

| Diretório | Diferença em relação ao controle |
|---|---|
| `control` | B de múltiplos planos |
| `frontier` | busca por fronteira e lacunas |
| `obligations` | checagem de condições da pergunta |
| `frontier-obligations` | ambos, para medir interação |
| `evidence` | ambos + contexto por provas completas |
| `operators` | anterior + leitura de tempo/contagem |
| `verified` | anterior + verificação textual das testemunhas |
| `no-acquisition` | `operators` sem extração dirigida |
| `one-plan` | `operators` com um único plano |

Ao final, `ablation.md` e `ablation.json` contêm F1 oficial, F1 multi-hop,
AR@5, ganhos/perdas pareados, F1 de contagens, diagnóstico numérico, taxa de
disparo e latência. Cada condição conserva seus JSONL, manifestos, logs,
relatórios e proveniência para auditoria posterior.

O diagnóstico numérico trata `2` e `two` como o mesmo inteiro quando predição
e gabarito contêm um único número reconhecível. Ele **não substitui** o F1
oficial do LoCoMo. O piloto de uma conversa serve para depuração; só a matriz
de dez conversas permite avaliar a meta multi-hop de 0,45 com o denominador
completo. Nenhum ganho da nova proposta foi medido antes de rodar no servidor.
