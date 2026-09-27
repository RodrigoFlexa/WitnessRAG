# Validação da versão experimental

- Suíte completa durante a implementação: 376 testes passaram. Depois da última
  restrição das premissas ao orçamento de fatos, os 75 testes relevantes passaram,
  incluindo 35 casos novos do portfólio.
- Simulação do comando de comparação validou as quatro variantes sem API.
- Relatório foi executado sobre as predições existentes da conv05, em diretório
  separado (`runs/multiplan-report-smoke`); os resultados originais não foram regravados.
- Sintaxe PowerShell, compilação Python e `git diff --check` passaram.

Checagem real com gpt-4o-mini: sete chamadas, nenhuma extração nem chamada de
leitor do LoCoMo. `results.json`: cadeia de dois fatos e lista de dois itens,
ambas com contratos válidos e testemunhas entregues. `contrast-results.json`:
trabalhar/liderar passou; possuir/tocar distinguiu a semântica mas recusou também
o item correto por citar a pergunta em q0. O parser não aceitou essa citação.
`contrast-repaired.json`: após esclarecer o prompt com um exemplo de citação da
fonte, somente o instrumento efetivamente tocado foi aceito.

Esses casos não são uma avaliação de F1/BLEU. A rodada completa permanece por
executar com `python scripts/run-multiplan-comparison.py`.
