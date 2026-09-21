"""Métricas do avaliador oficial do LoCoMo.

Porte fiel de `task_eval/evaluation.py` no commit fixado em `wrag.locomo`:
normalização que remove "and" além dos artigos, stemming de Porter e, na
categoria 1, F1 por sub-resposta separada por vírgula. O EM oficial compara
*conjuntos* de tokens, então a ordem dos itens não conta.

Estas funções não substituem `wrag.eval.metrics`: o harness usa o mesmo leitor e
as mesmas métricas em todos os datasets, e essa coluna existe para comparar com
tabelas publicadas do LoCoMo. As duas são calculadas sobre a mesma predição.

O stemmer é o do NLTK, como no original; uma reimplementação local mudaria os
números sem aviso. Sem NLTK instalado, a coluna oficial é omitida, não estimada.
"""

from __future__ import annotations

import string
from functools import lru_cache
from typing import Sequence

import regex

from wrag.locomo import REVISION as LOCOMO_REVISION

# Answer categories from the pinned LoCoMo evaluator.  BLEU-1 below is an
# additional requested diagnostic; upstream officially defines F1, not BLEU.
CATEGORY_BY_TYPE = {"multi-hop": 1, "temporal": 2, "open-domain": 3, "single-hop": 4}
CATEGORY_NAME = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
SOURCE = (f"https://github.com/snap-research/locomo/blob/{LOCOMO_REVISION}"
          "/task_eval/evaluation.py")


class StemmerUnavailable(RuntimeError):
    """NLTK ausente: o avaliador oficial não pode ser reproduzido fielmente."""


@lru_cache(maxsize=1)
def _stemmer():
    try:
        from nltk.stem import PorterStemmer
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise StemmerUnavailable(
            "o avaliador oficial do LoCoMo usa nltk.stem.PorterStemmer; "
            "instale `nltk` para calcular essa coluna") from exc
    return PorterStemmer()


def available() -> bool:
    try:
        _stemmer()
    except StemmerUnavailable:
        return False
    return True


def normalize_answer(s: str) -> str:
    """Normalização oficial: remove vírgulas, pontuação, artigos e "and"."""
    s = (s or "").replace(",", "")

    def remove_articles(text: str) -> str:
        return regex.sub(r"\b(a|an|the|and)\b", " ", text)

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return " ".join(remove_articles(remove_punc(s.lower())).split())


def _stem_tokens(text: str) -> list[str]:
    stem = _stemmer().stem
    return [stem(w) for w in normalize_answer(text).split()]


def f1_score(prediction: str, ground_truth: str) -> float:
    """F1 por token com stemming. Usado nas categorias 2, 3 e 4."""
    from collections import Counter

    pred = _stem_tokens(prediction)
    gold = _stem_tokens(ground_truth)
    if not pred or not gold:
        return 0.0
    num_same = sum((Counter(pred) & Counter(gold)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(gold)
    return 2 * precision * recall / (precision + recall)


def multi_answer_f1(prediction: str, ground_truth: str) -> float:
    """Categoria 1: média, sobre cada sub-resposta ouro, do melhor F1 predito.

    Só a vírgula separa itens, e a média é sobre o ouro: itens previstos a mais
    não são penalizados. É assimétrico de propósito, e é o que o oficial faz.
    """
    predictions = [p.strip() for p in (prediction or "").split(",")]
    golds = [g.strip() for g in (ground_truth or "").split(",")]
    scores = [max(f1_score(p, g) for p in predictions) for g in golds]
    return sum(scores) / len(scores)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """EM oficial: igualdade entre *conjuntos* de tokens normalizados."""
    pred = set(normalize_answer(prediction).split())
    gold = set(normalize_answer(ground_truth).split())
    return float(pred == gold)


def bleu1_score(prediction: str, ground_truth: str) -> float:
    """Unigram BLEU with clipped precision and the standard brevity penalty.

    This is dependency-free and deterministic.  For category 1 we mirror the
    official multi-answer F1 convention: each gold comma-separated item takes
    its best matching predicted item and extra predictions are not penalized.
    """
    import math
    from collections import Counter
    pred = _stem_tokens(prediction)
    gold = _stem_tokens(ground_truth)
    if not pred or not gold:
        return 0.0
    overlap = sum((Counter(pred) & Counter(gold)).values())
    precision = overlap / len(pred)
    if precision == 0:
        return 0.0
    brevity = 1.0 if len(pred) >= len(gold) else math.exp(1.0 - len(gold) / len(pred))
    return brevity * precision


def multi_answer_bleu1(prediction: str, ground_truth: str) -> float:
    predictions = [p.strip() for p in (prediction or "").split(",")]
    golds = [g.strip() for g in (ground_truth or "").split(",")]
    return sum(max(bleu1_score(p, g) for p in predictions) for g in golds) / len(golds)


def question_score(prediction: str, answer: str, category: int) -> float:
    """Despacho por categoria, como em `eval_question_answering`."""
    if category == 3:
        answer = answer.split(";")[0].strip()
    if category == 1:
        return multi_answer_f1(prediction, answer)
    if category in (2, 3, 4):
        return f1_score(prediction, answer)
    if category == 5:
        text = (prediction or "").lower()
        return float("no information available" in text or "not mentioned" in text)
    raise ValueError(f"categoria LoCoMo desconhecida: {category!r}")


def evidence_recall(retrieved_dialog_ids: Sequence[str], evidence: Sequence[str]) -> float:
    """Recall oficial: fração das falas de evidência presentes no contexto.

    O oficial mede sobre identificadores de fala (ou de sessão). Aqui o contexto
    é um bloco de falas, então o chamador precisa expandir as passagens
    recuperadas nas falas que elas contêm. É recall por evidência, não a
    exigência de recuperar todas — essa é `all_recall` do harness.
    """
    if not evidence:
        return float("nan")
    retrieved = set(retrieved_dialog_ids)
    return sum(1 for ev in evidence if ev in retrieved) / len(evidence)


def score_record(record: dict) -> dict[str, float]:
    """Métricas oficiais de um registro do harness (`<dataset>/<metodo>.jsonl`)."""
    category = record.get("categoria_locomo") or CATEGORY_BY_TYPE.get(record.get("tipo", ""))
    if category is None:
        raise ValueError(f"registro sem categoria oficial: {record.get('qid')!r}")
    gold = record["respostas_ouro"][0]
    prediction = record.get("resposta", "")
    bleu_gold = gold.split(";")[0].strip() if category == 3 else gold
    bleu = bleu1_score(prediction, bleu_gold)
    return {"f1_locomo": question_score(prediction, gold, category),
            "bleu1_locomo": bleu,
            "em_locomo": exact_match_score(prediction, gold),
            "categoria_locomo": category}


# ---------------------------------------------------------------------------
# Agregação de várias conversas
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float("nan")


def aggregate_runs(run_dirs: Sequence["Path"]) -> dict:
    """Junta as rodadas por conversa numa tabela só, por método.

    A média é **micro**: todas as perguntas entram no mesmo denominador, sem
    ponderar conversas iguais. Uma conversa com 260 perguntas pesa mais que uma
    com 105, que é o que se quer quando a unidade de interesse é a pergunta. O
    detalhamento por conversa vai junto, porque a variância entre conversas é
    grande e uma média só esconde isso.

    Cada conversa é um corpus próprio: estas rodadas não são comparáveis com um
    experimento que juntasse as dez num índice único.
    """
    import json
    from pathlib import Path

    per_method: dict[str, list[dict]] = {}
    for run_dir in run_dirs:
        for path in sorted(Path(run_dir).glob("locomo/*.jsonl")):
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for row in rows:
                row["_conversa"] = row["qid"].split(":")[1] if ":" in row["qid"] else str(run_dir)
            per_method.setdefault(path.stem, []).extend(rows)

    scored = available()
    out: dict = {"conversas": len(list(run_dirs)), "metodos": {}, "oficial_disponivel": scored}
    for method, rows in per_method.items():
        for row in rows:
            if scored and ("f1_locomo" not in row or "bleu1_locomo" not in row):
                row.update(score_record(row))

        def block(subset: list[dict]) -> dict:
            values = {"n": len(subset),
                      "f1": _mean([r["f1"] for r in subset]),
                      "em": _mean([r["em"] for r in subset]),
                      "recall@5": _mean([r["recall@5"] for r in subset]),
                      "all_recall@5": _mean([r["all_recall@5"] for r in subset])}
            if scored:
                values["f1_locomo"] = _mean([r["f1_locomo"] for r in subset])
                values["bleu1_locomo"] = _mean([r["bleu1_locomo"] for r in subset])
                values["em_locomo"] = _mean([r["em_locomo"] for r in subset])
            fired = ([r for r in subset if isinstance(r.get("diagnosticos"), dict)]
                     if method.startswith("witnessrag") or method == "relational" else [])
            if fired:
                values["taxa_de_disparo"] = _mean(
                    [float("fallback" not in r["diagnosticos"]) for r in fired])
                values["taxa_de_intervencao"] = _mean(
                    [float(bool(r["diagnosticos"].get("contexto_alterado_pelo_witness")))
                     for r in fired])
            return values

        entry = block(rows)
        entry["por_categoria"] = {
            name: block([r for r in rows if r.get("tipo") == name])
            for name in ("single-hop", "multi-hop", "temporal", "open-domain")
            if any(r.get("tipo") == name for r in rows)
        }
        entry["por_conversa"] = {
            conversation: block([r for r in rows if r["_conversa"] == conversation])
            for conversation in sorted({r["_conversa"] for r in rows})
        }
        out["metodos"][method] = entry
    return out
