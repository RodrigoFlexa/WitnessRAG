"""
Carregamento dos formatos MuSiQue, 2WikiMultiHopQA, HotpotQA e sample.

O padrão mantém o corpus completo disponível ao subamostrar perguntas.
subset_corpus é uma condição de piloto explícita e altera o universo de busca.
Evidências e decomposições anotadas auxiliam diagnósticos; sua tradução
heurística não estabelece consultas formalmente corretas. Compartilhar os
arquivos de um artigo não torna esta implementação uma reprodução do artigo.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from wrag import config as C
from wrag.util import get_logger, normalize, read_json, sha

log = get_logger("wrag.data")

DATASETS = ("musique", "2wikimultihopqa", "hotpotqa", "sample", "locomo")


@dataclass(frozen=True)
class Passage:
    pid: str
    title: str
    text: str

    @property
    def full(self) -> str:
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass
class Question:
    qid: str
    question: str
    answers: list[str]
    gold_pids: list[str] = field(default_factory=list)
    dataset: str = ""
    qtype: str = ""
    evidences: list[tuple[str, str, str]] = field(default_factory=list)
    decomposition: list[dict[str, Any]] = field(default_factory=list)
    missing_supports: int = 0

    @property
    def n_hops(self) -> int:
        if self.decomposition:
            return len(self.decomposition)
        if self.evidences:
            return len(self.evidences)
        return max(1, len(self.gold_pids))

    @property
    def hop_source(self) -> str:
        return "decomposition" if self.decomposition else "evidence_count" if self.evidences else "support_count_proxy"


@dataclass
class Corpus:
    name: str
    passages: list[Passage]
    questions: list[Question]

    def __post_init__(self) -> None:
        self._by_pid = {p.pid: p for p in self.passages}

    def get(self, pid: str) -> Passage:
        return self._by_pid[pid]

    def texts(self) -> list[str]:
        return [p.full for p in self.passages]

    @property
    def pids(self) -> list[str]:
        return [p.pid for p in self.passages]

    def stats(self) -> dict[str, Any]:
        hops = [q.n_hops for q in self.questions]
        return {
            "dataset": self.name,
            "n_passages": len(self.passages),
            "n_questions": len(self.questions),
            "hops_medio": round(sum(hops) / max(1, len(hops)), 2),
            "com_evidencia_anotada": sum(1 for q in self.questions if q.evidences),
            "com_decomposicao": sum(1 for q in self.questions if q.decomposition),
            "question_ids": [q.qid for q in self.questions],
            "corpus_hash": sha([(p.pid, p.title, p.text) for p in self.passages]),
            "questions_hash": sha([(q.qid, q.question, q.answers, q.gold_pids,
                                     q.evidences, q.decomposition) for q in self.questions]),
        }


# ---------------------------------------------------------------------------
# Casamento passagem <-> anotação
# ---------------------------------------------------------------------------
# Os três datasets referenciam passagens de formas diferentes (por índice, por
# título, por título+texto). Um índice único resolve os três, e o casamento por
# título é o fallback porque HotpotQA e 2Wiki só dão o título.

class _PassageIndex:
    def __init__(self) -> None:
        self.by_key: dict[str, str] = {}
        self.by_title: dict[str, set[str]] = {}
        self.passages: list[Passage] = []

    def add(self, title: str, text: str) -> str:
        key = sha(title.strip(), text.strip())
        if key in self.by_key:
            return self.by_key[key]
        pid = "p-" + key[:24]
        passage = Passage(pid=pid, title=title.strip(), text=text.strip())
        self.passages.append(passage)
        self.by_key[key] = pid
        self.by_title.setdefault(normalize(title), set()).add(pid)
        return pid

    def find(self, title: str, text: str | None = None) -> str | None:
        if text:
            key = sha(title.strip(), text.strip())
            if key in self.by_key:
                return self.by_key[key]
            return None  # texto distinto não pode ser substituído pelo primeiro título
        candidates = self.by_title.get(normalize(title), set())
        return next(iter(candidates)) if len(candidates) == 1 else None


def _dataset_paths(name: str, data_dir: Path) -> tuple[Path, Path]:
    return data_dir / f"{name}.json", data_dir / f"{name}_corpus.json"


def load_dataset(
    name: str,
    n_questions: int | None = None,
    seed: int = C.SEED,
    data_dir: Path | None = None,
    subset_corpus: bool = False,
) -> Corpus:
    data_dir = data_dir or C.DATA_DIR
    qpath, cpath = _dataset_paths(name, data_dir)
    if not qpath.exists():
        if name == "locomo":
            raise FileNotFoundError(f"{qpath} não existe. Use `python -m wrag.pilot --dataset locomo` "
                                    "ou prepare com `python -m wrag.locomo --output <pasta>`.")
        raise FileNotFoundError(
            f"{qpath} não existe. Rode `python -m wrag.cli prepare-data` para copiar os "
            f"subconjuntos oficiais do repositório do HippoRAG."
        )

    raw_questions = read_json(qpath, [])
    raw_corpus = read_json(cpath, [])

    index = _PassageIndex()
    for item in raw_corpus:
        index.add(item.get("title", ""), item.get("text", ""))
    if not raw_corpus:
        for item in raw_questions:
            for title, text in _iter_candidates(name, item):
                index.add(title, text)

    questions = [q for q in (_parse_question(name, item, index) for item in raw_questions) if q]
    if len({q.qid for q in questions}) != len(questions):
        raise ValueError("IDs de perguntas duplicados no dataset")
    if any(q.missing_supports for q in questions):
        raise ValueError("passagens de apoio não encontradas ou ambíguas no corpus; corrija os dados")

    if n_questions and n_questions < len(questions):
        rng = random.Random(seed)
        questions = sorted(rng.sample(questions, n_questions), key=lambda q: q.qid)

    passages = index.passages
    if subset_corpus and n_questions:
        keep = _candidate_pids(name, raw_questions, questions, index)
        if keep:
            passages = [p for p in index.passages if p.pid in keep]
            log.info("%s: corpus reduzido a %d passagens candidatas das %d perguntas sorteadas",
                     name, len(passages), len(questions))

    corpus = Corpus(name=name, passages=passages, questions=questions)
    missing = sum(1 for q in questions if not q.gold_pids)
    if missing:
        log.warning("%s: %d perguntas sem passagem de apoio casada (recall será subestimado nelas)",
                    name, missing)
    log.info("%s carregado: %s", name, {k: v for k, v in corpus.stats().items() if k != "question_ids"})
    return corpus


def _candidate_pids(name: str, raw_questions: list[dict], kept: list[Question],
                    index: _PassageIndex) -> set[str]:
    """Passagens candidatas (apoio + distratores) das perguntas mantidas."""
    keep_ids = {q.qid for q in kept}
    pids: set[str] = set()
    for item in raw_questions:
        qid = str(item.get("id") or item.get("_id") or "")
        if qid not in keep_ids:
            continue
        for title, text in _iter_candidates(name, item):
            pid = index.find(title, text)
            if pid:
                pids.add(pid)
    for q in kept:
        pids.update(q.gold_pids)
    return pids


def _iter_candidates(name: str, item: dict) -> Iterable[tuple[str, str]]:
    if name == "musique" or "paragraphs" in item:
        for par in item.get("paragraphs", []):
            yield par.get("title", ""), par.get("paragraph_text", par.get("text", ""))
    if "context" in item:
        for entry in item.get("context", []):
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                title, sentences = entry
                text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
                yield title, text


def _parse_question(name: str, item: dict, index: _PassageIndex) -> Question | None:
    qid = str(item.get("id") or item.get("_id") or "")
    text = (item.get("question") or "").strip()
    if not qid or not text:
        return None

    answer = item.get("answer")
    answers = [answer] if isinstance(answer, str) else list(answer or [])
    answers += [a for a in item.get("answer_aliases", []) if a]
    answers = [a for a in answers if isinstance(a, str) and a.strip()]

    gold: list[str] = []
    evidences: list[tuple[str, str, str]] = []
    decomposition: list[dict[str, Any]] = []
    missing_supports = 0

    if "paragraphs" in item:  # MuSiQue e sample
        for par in item["paragraphs"]:
            if par.get("is_supporting"):
                pid = index.find(par.get("title", ""), par.get("paragraph_text", par.get("text", "")))
                if pid:
                    gold.append(pid)
                else:
                    missing_supports += 1
        decomposition = list(item.get("question_decomposition", []) or [])

    if "supporting_facts" in item:  # 2Wiki e HotpotQA
        context = {c[0]: (" ".join(c[1]) if isinstance(c[1], list) else str(c[1]))
                   for c in item.get("context", []) if isinstance(c, (list, tuple)) and len(c) == 2}
        for fact in item["supporting_facts"]:
            title = fact[0] if isinstance(fact, (list, tuple)) else str(fact)
            pid = index.find(title, context.get(title))
            if pid and pid not in gold:
                gold.append(pid)
            if not pid:
                missing_supports += 1

    for ev in item.get("evidences", []) or []:
        if isinstance(ev, (list, tuple)) and len(ev) == 3:
            evidences.append((str(ev[0]), str(ev[1]), str(ev[2])))

    return Question(
        qid=qid, question=text, answers=answers, gold_pids=gold, dataset=name,
        qtype=str(item.get("type") or item.get("level") or ""),
        evidences=evidences, decomposition=decomposition, missing_supports=missing_supports,
    )
