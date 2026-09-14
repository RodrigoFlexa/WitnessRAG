"""Adaptação textual do LoCoMo: uma conversa, categorias oficiais 1 e 4.

Prepara dados sem modelo/GPU: python -m wrag.locomo --output runs/locomo-data
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request

from wrag.util import write_json

REVISION = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
URL = f"https://raw.githubusercontent.com/snap-research/locomo/{REVISION}/data/locomo10.json"
CATEGORIES = {1: "multi-hop", 4: "single-hop"}


def convert(raw, conversation_index=0, turns_per_passage=8, n_questions=None, seed=42):
    if not isinstance(raw, list) or not 0 <= conversation_index < len(raw):
        raise ValueError("índice de conversa inválido (começa em zero)")
    if turns_per_passage < 1 or (n_questions is not None and n_questions < 1):
        raise ValueError("tamanho de passagem e número de perguntas devem ser positivos")
    sample = raw[conversation_index]
    sample_id = str(sample["sample_id"])
    conversation = sample["conversation"]
    sessions = sorted((k for k in conversation if re.fullmatch(r"session_\d+", k)),
                      key=lambda k: int(k.split("_")[1]))
    passages, turn_to_passage = [], {}
    # Segmentação determinada apenas pelo diálogo, jamais pela evidência ouro.
    for session in sessions:
        turns = conversation[session]
        for start in range(0, len(turns), turns_per_passage):
            block = turns[start:start + turns_per_passage]
            lines = []
            title = f"LoCoMo {sample_id} {session} block {start // turns_per_passage + 1}"
            date = str(conversation.get(f"{session}_date_time", ""))
            if date:
                lines.append(f"Session date: {date}")
            for turn in block:
                dia_id = turn["dia_id"]
                if dia_id in turn_to_passage:
                    raise ValueError(f"dia_id duplicado: {dia_id}")
                turn_to_passage[dia_id] = len(passages)
                lines.append(f"[{dia_id}] {turn['speaker']}: {turn['text']}")
                if turn.get("blip_caption"):
                    lines.append(f"[{dia_id}] Image caption (automatic): {turn['blip_caption']}")
            passages.append({"title": title, "text": "\n".join(lines)})
    if not passages:
        raise ValueError("conversa vazia")
    questions, mapping = [], {}
    for qi, qa in enumerate(sample["qa"]):
        category = qa["category"]
        if category not in CATEGORIES:
            continue
        evidence = []
        for ref in qa.get("evidence", []):
            # A primeira conversa contém a anotação 'D8:6; D9:17'.
            refs = [s.strip() for s in re.split(r"[;,]", ref) if s.strip()]
            for dia_id in refs:
                if dia_id not in turn_to_passage:
                    raise ValueError(f"QA {qi}: evidência desconhecida {dia_id!r}")
                if dia_id not in evidence:
                    evidence.append(dia_id)
        if not evidence:
            raise ValueError(f"QA {qi}: sem evidência; recall não pode ser avaliado")
        answer = qa.get("answer")
        if answer is None or isinstance(answer, (dict, list)):
            raise ValueError(f"QA {qi}: resposta inválida")
        qid = f"locomo:{sample_id}:qa{qi}"
        supports = sorted({turn_to_passage[e] for e in evidence})
        questions.append({"id": qid, "question": qa["question"], "answer": str(answer),
                          "type": CATEGORIES[category], "paragraphs": [
                              {**passages[i], "is_supporting": True} for i in supports]})
        mapping[qid] = {"category": category, "evidence_dialog_ids": evidence,
                        "support_titles": [passages[i]["title"] for i in supports]}
    if n_questions and n_questions < len(questions):
        questions = sorted(random.Random(seed).sample(questions, n_questions), key=lambda q: q["id"])
    if not questions:
        raise ValueError("nenhuma pergunta single-hop/multi-hop")
    metadata = {"dataset": "locomo", "conversation_index": conversation_index,
                "sample_id": sample_id, "categories": CATEGORIES,
                "questions": len(questions), "questions_by_type": dict(Counter(q["type"] for q in questions)),
                "sessions": len(sessions), "turns": len(turn_to_passage),
                "selected_passages": len(passages), "turns_per_passage": turns_per_passage,
                "corpus_scope": "locomo_full_selected_conversation", "seed": seed,
                "text_policy": "speaker + dialog id + date + text + released BLIP captions; no summaries/personas/QA",
                "metric_policy": "harness EM/token F1; recall at passage-block level; not official LoCoMo scoring",
                "question_mapping": {q["id"]: mapping[q["id"]] for q in questions}}
    return questions, passages, metadata


def prepare(output, source_file=None, conversation_index=0, turns_per_passage=8,
            n_questions=None, seed=42, max_passages=1500):
    output = Path(output)
    snapshot = output / "source-data" / "locomo10.json"
    if source_file:
        payload = Path(source_file).read_bytes()
        source = str(Path(source_file).resolve())
    elif snapshot.exists():
        payload = snapshot.read_bytes()
        source = "saved snapshot (origin not revalidated)"
    else:
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
        source = URL
    questions, passages, metadata = convert(json.loads(payload), conversation_index,
        turns_per_passage, n_questions, seed)
    if len(passages) > max_passages:
        raise ValueError(f"conversa tem {len(passages)} passagens; aumente --max-passages")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(payload)
    metadata.update(source=source, source_sha256=hashlib.sha256(payload).hexdigest(),
                    upstream_revision=REVISION if source == URL else None)
    write_json(output / "data" / "locomo.json", questions)
    write_json(output / "data" / "locomo_corpus.json", passages)
    write_json(output / "data_selection.json", metadata)
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-file", type=Path)
    p.add_argument("--conversation-index", type=int, default=0)
    p.add_argument("--turns-per-passage", type=int, default=8)
    args = p.parse_args()
    meta = prepare(args.output, args.source_file, args.conversation_index, args.turns_per_passage)
    print(json.dumps({k: v for k, v in meta.items() if k != "question_mapping"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
