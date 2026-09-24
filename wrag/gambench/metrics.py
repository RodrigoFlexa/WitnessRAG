"""
Métricas do protocolo do GAM, portadas literalmente do código de avaliação
(research/eval/*_test.py, commit em protocol.GAM_COMMIT; licença MIT,
Copyright (c) 2025 VectorSpaceLab).

* HotpotQA e NarrativeQA: `qa_f1_score` do LongBench, o máximo sobre as
  respostas de referência. Resposta vazia vale 0.
* RULER: `evaluate_answer`, correto (1) somente se TODAS as saídas esperadas
  aparecem na resposta, por uma de três regras: substring em minúsculas,
  substring depois de trocar pontuação por espaço, ou todas as palavras de mais
  de dois caracteres presentes.

Duas métricas extras ficam ao lado, só para diagnóstico e nunca na tabela: a
pontuação oficial da RULER (string_match_all / string_match_part, com crédito
parcial) e o exact match normalizado.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Sequence


# -- HotpotQA / NarrativeQA (idêntico nos dois scripts do GAM) ---------------

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: Sequence[str], ground_truth: Sequence[str]) -> float:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def gam_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    """`_calculate_f1` com a guarda do chamador (`if pred_answer else 0.0`)."""
    if not prediction:
        return 0.0
    best = 0.0
    for gold in gold_answers:
        best = max(best, qa_f1_score(prediction, gold))
    return float(best)


def exact_match(prediction: str, gold_answers: Sequence[str]) -> float:
    """Diagnóstico: EM depois da mesma normalização do F1."""
    pred = normalize_answer(prediction or "")
    return float(any(pred == normalize_answer(g) for g in gold_answers)) if pred else 0.0


# -- RULER (eval/ruler_test.py) ----------------------------------------------

def ruler_normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ruler_correct(model_response: str, ground_truth_outputs: Sequence[str]) -> bool:
    """`evaluate_answer` do GAM, linha por linha."""
    if not ground_truth_outputs:
        return False
    if not model_response:
        return False
    model_response_lower = model_response.lower()
    model_response_normalized = ruler_normalize_text(model_response)
    unique_answers = list(set(ground_truth_outputs))
    for answer in unique_answers:
        answer_str = str(answer).strip()
        if not answer_str:
            continue
        answer_lower = answer_str.lower()
        if answer_lower in model_response_lower:
            continue
        answer_normalized = ruler_normalize_text(answer_str)
        if answer_normalized in model_response_normalized:
            continue
        answer_words = [w for w in answer_normalized.split() if len(w) > 2]
        if answer_words:
            if all(word in model_response_normalized for word in answer_words):
                continue
        return False
    return True


def ruler_string_match_all(prediction: str, references: Sequence[str]) -> float:
    """Diagnóstico: métrica oficial da RULER para NIAH/VT/CWE/FWE (0-1)."""
    refs = [str(r) for r in references]
    if not refs:
        return 0.0
    pred = (prediction or "").lower()
    return sum(1.0 if r.lower() in pred else 0.0 for r in refs) / len(refs)


def ruler_string_match_part(prediction: str, references: Sequence[str]) -> float:
    """Diagnóstico: métrica oficial da RULER para QA (0-1)."""
    pred = (prediction or "").lower()
    return max((1.0 if str(r).lower() in pred else 0.0 for r in references), default=0.0)


def ruler_official(task: str, prediction: str, references: Sequence[str]) -> float:
    if task.startswith("qa"):
        return ruler_string_match_part(prediction, references)
    return ruler_string_match_all(prediction, references)
