# WitnessRAG benchmark suite

The suite fixes the generator to `Qwen/Qwen2.5-14B-Instruct`, the reader budget
to five passages, seed 42, and physical GPU 1 by default.  Every run writes raw
JSONL predictions, a final `report.json`/`report.md`, and a `live_report.json`
after every completed question.

## Ablation contract

The LoCoMo pilot has four cumulative, paired arms over the same conversation,
questions, model, embeddings, and shared cache:

| arm | graded obligations | temporal metadata/ranking | one complementary tail swap |
|---|---:|---:|---:|
| `base` | no | no | no |
| `soft` | yes | no | no |
| `temporal` | yes | yes | no |
| `full` | yes | yes | yes |

The context selector is answer-blind.  It uses only the question, compiled plan,
source conditions, passage text, session date, and sequence.  It protects the
first four passages and never changes a context containing a complete delivered
witness.  It adds no LLM call.

Run conversation 0 first:

```bash
GPU=1 bash scripts/run-witness-suite.sh locomo runs/witness-suite-locomo-conv00
```

Only after a paired gain is established, run all conversations by setting
`LOCOMO_CONVERSATION=all`.  The comparison is written to
`ablation_report.json` and `ablation_report.md`.

LoCoMo includes categories 1–4: multi-hop, temporal, open-domain and single-hop.
F1 follows the pinned official evaluator.  BLEU-1 is an added deterministic
unigram diagnostic because the upstream LoCoMo evaluator does not define BLEU.

## HotpotQA memory sizes

The three conditions preserve all selected questions and their supports, then
add seeded distractors until the index contains exactly 56,000, 224,000 or
448,000 passages.  The realized count is recorded in `data_selection.json`.

```bash
GPU=1 HOTPOT_QUESTIONS=1000 bash scripts/run-witness-suite.sh hotpotqa runs/witness-suite-hotpot
```

This creates independent 56k, 224k and 448k reports.  F1 is the primary metric.

## RULER 128k

RULER examples are indexed independently so information cannot cross synthetic
contexts.  Put `retrieval.jsonl`, `mt.jsonl`, `agg.jsonl`, and `qa.jsonl` in one
directory (JSON is also accepted):

```bash
GPU=1 RULER_DIR=/data/ruler-128k bash scripts/run-witness-suite.sh ruler runs/witness-suite-ruler
```

The scalable default is `witnessrag-lite`: hybrid retrieval plus the zero-LLM context policy.  The
isolated runner also accepts `--engine witnessrag` for the much more expensive
OpenIE/join condition.  Reports contain F1 and EM overall and per task.

## NarrativeQA

The adapter accepts common JSON/JSONL fields (`context`, `document.text`,
`question`, `answers`).  Each story/question context is isolated and chunked.

```bash
GPU=1 NARRATIVEQA_FILE=/data/narrativeqa-test.jsonl \
  bash scripts/run-witness-suite.sh narrativeqa runs/witness-suite-narrativeqa
```

F1 is primary.  Raw answers, retrieved chunks, latency, and per-item diagnostics
remain in `predictions.jsonl`.
