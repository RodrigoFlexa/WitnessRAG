# Artigo AAMAS 2027: organização

Compile `main.tex` (pdfLaTeX + BibTeX). No Overleaf, defina `main.tex` como documento principal.

## Arquivos

| arquivo | conteúdo |
|---|---|
| `main.tex` | classe, pacotes, macros, siglas, título, autores, ordem das seções |
| `sections/abstract.tex` | resumo (~200 palavras) |
| `sections/introduction.tex` | 1. Introduction |
| `sections/related_work.tex` | 2. Related Work (3 subseções) |
| `sections/problem_formulation.tex` | 3. Problem Formulation |
| `sections/method.tex` | 4. WitnessRAG (7 subseções) |
| `sections/experimental_setup.tex` | 5. Experimental Setup |
| `sections/results.tex` | 6. Results and Analysis |
| `sections/discussion.tex` | 7. Discussion and Limitations |
| `sections/conclusion.tex` | 8. Conclusion |
| `sections/acknowledgments.tex` | agradecimentos (somem no modo anônimo) |
| `sections/appendix.tex` | material suplementar (fora das 8 páginas) |
| `references.bib` | referências |

Os arquivos do template (`aamas.cls`, `ACM-Reference-Format.bst`, `by.pdf`,
`AAMAS_2027_sample.tex`...) ficam como vieram.

## Regras da chamada (main track)

- Até **8 páginas** de texto; referências em páginas extras sem limite.
- Revisão **duplo-cega**: manter `\documentclass[sigconf,anonymous]{aamas}`
  até o camera-ready. Autocitação só na terceira pessoa (como está com o
  artigo do EMAS).
- Suplementar: um zip de até 25 MB; os revisores não são obrigados a ler, então
  nada essencial pode ficar só lá.
- Se ferramentas de IA generativa foram usadas para criar hipóteses ou
  metodologia, a chamada pede para informar o prompt, a ferramenta e a versão.
- Prazos: resumo em 1/10/2026, artigo em 8/10/2026 (AoE).
- Área sugerida: *Generative and Agentic AI (GAAI)*.

## Estilo (o que os artigos do AAMAS fazem)

- Seções numeradas em inglês: Introduction, Related Work, formulação do
  problema, método, Experimental Setup, resultados, discussão, Conclusion.
- Introdução: contexto, problema, lacuna, ideia, visão geral do método, o que
  nos diferencia, resultados principais, contribuições em lista, parágrafo final
  com a organização do texto.
- Related Work em subseções; cada uma termina posicionando o nosso trabalho.
- Primeira pessoa do plural ("we propose"), presente para o método, passado
  para os experimentos. Parágrafos de 4 a 6 frases.
- Tabelas com `booktabs`, legenda acima; figuras com legenda abaixo e
  `\Description{...}` (exigido pela ACM para acessibilidade).

## Convenções do texto

- Nome do método: `\sys` (troca num lugar só, em `main.tex`).
- Operações: `\register`, `\search`, `\plan`, `\prove`, `\verify`, `\respond`.
- Relações e variáveis: `\rel{camped\_at}(Melanie, \var{where})`.
- Siglas: `\ac{LLM}` (primeira vez por extenso), `\acp{LLM}` (plural).
- Pendências: `\todo{...}` aparece em vermelho. Antes de submeter:
  `grep -n "\\todo" sections/*.tex` tem de voltar vazio.
