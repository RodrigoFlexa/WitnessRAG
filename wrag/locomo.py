"""Adaptação textual do LoCoMo: uma conversa, categorias oficiais 1--4.

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
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta

from wrag.util import write_json

REVISION = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
URL = f"https://raw.githubusercontent.com/snap-research/locomo/{REVISION}/data/locomo10.json"
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}

# The released snapshot contains four malformed evidence references. Keep the
# repairs narrow and auditable: any other unknown id remains a hard error.
EVIDENCE_REPAIRS = {
    ("conv-42", 58, "D10:19"): "D20:15",
    ("conv-42", 88, "D"): "D1:16",
    ("conv-43", 18, "D:11:26"): "D11:26",
    ("conv-47", 38, "D4:36"): "D13:3",
    ("conv-50", 69, "D30:05"): "D30:5",
}


def _temporal_annotation(dia_id: str, text: str, session_date: str) -> str:
    """Normalize relative time using the turn's session date, without QA labels."""
    match = re.search(r"\b(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\b", session_date)
    if not match:
        return ""
    try:
        anchor = datetime.strptime(" ".join(match.groups()), "%d %B %Y").date()
    except ValueError:
        return ""
    fmt = lambda value: f"{value.day} {value.strftime('%B %Y')}"
    low, values = text.casefold(), []
    def add(source, value):
        if source in low:
            values.append(f'"{source}"={value}')
    add("today", fmt(anchor)); add("yesterday", fmt(anchor - timedelta(days=1)))
    add("tomorrow", fmt(anchor + timedelta(days=1)))
    add("two days ago", fmt(anchor - timedelta(days=2)))
    add("last year", str(anchor.year - 1)); add("next year", str(anchor.year + 1))
    previous_month = anchor.replace(day=1) - timedelta(days=1)
    next_month = (anchor.replace(day=28) + timedelta(days=4)).replace(day=1)
    add("last month", previous_month.strftime("%B %Y")); add("next month", next_month.strftime("%B %Y"))
    this_monday = anchor - timedelta(days=anchor.weekday())
    prior_week_start, prior_week_end = this_monday - timedelta(days=7), this_monday - timedelta(days=1)
    weekend_end = anchor - timedelta(days=(anchor.weekday() - 6) % 7 or 7)
    weekend_start = weekend_end - timedelta(days=1)
    add("last week", f"the week before {fmt(anchor)} [start={prior_week_start.isoformat()}, end={prior_week_end.isoformat()}]")
    add("last weekend", f"the weekend before {fmt(anchor)} [start={weekend_start.isoformat()}, end={weekend_end.isoformat()}]")
    add("past weekend", f"the weekend before {fmt(anchor)} [start={weekend_start.isoformat()}, end={weekend_end.isoformat()}]")
    weekdays = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    for number, weekday in enumerate(weekdays):
        aliases = {weekday, weekday[:3], weekday[:4]}
        if any(re.search(rf"\blast\s+{re.escape(alias)}\b", low) for alias in aliases):
            delta = (anchor.weekday() - number) % 7 or 7
            point = anchor - timedelta(days=delta)
            values.append(f'"last {weekday}"={fmt(point)} [start={point.isoformat()}, end={point.isoformat()}]')
    duration = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten) years? ago\b", low)
    if duration:
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        years = int(duration.group(1)) if duration.group(1).isdigit() else words[duration.group(1)]
        values.append(f'"{duration.group(0)}"={anchor.year - years}')
    return (f"[{dia_id} temporal] reference_time={anchor.isoformat()}; " +
            "; ".join(dict.fromkeys(values))) if values else ""


def convert(raw, conversation_index=0, turns_per_passage=8, n_questions=None, seed=42,
            chunk_tokens=None, tokenizer_name="Qwen/Qwen2.5-14B-Instruct",
            tokenizer_revision=None, token_counter=None, temporal_annotations=False):
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

    def turn_lines(turn, session_date=""):
        dia_id = turn["dia_id"]
        anchor = f" date={session_date}" if temporal_annotations and session_date else ""
        lines = [f"[{dia_id}{anchor}] {turn['speaker']}: {turn['text']}"]
        annotation = _temporal_annotation(dia_id, str(turn["text"]), session_date) if temporal_annotations else ""
        if annotation:
            lines.append(annotation)
        if turn.get("blip_caption"):
            lines.append(f"[{dia_id}] Image caption (automatic): {turn['blip_caption']}")
        return lines

    if chunk_tokens:
        if chunk_tokens < 1:
            raise ValueError("tamanho do chunk em tokens deve ser positivo")
        if token_counter is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, revision=tokenizer_revision or None)
            token_counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))

        current_lines, current_ids = [], []

        def flush():
            if not current_ids:
                return
            index = len(passages)
            first_date = next((line.partition(":")[2].strip() for line in current_lines
                               if line.startswith("Session date:")), "")
            passages.append({"title": f"LoCoMo {sample_id} history chunk {index + 1}",
                             "text": "\n".join(current_lines), "session_time": first_date,
                             "sequence": index, "source_ids": list(current_ids)})
            for dia_id in current_ids:
                turn_to_passage[dia_id] = index
            current_lines.clear()
            current_ids.clear()

        for session in sessions:
            date = str(conversation.get(f"{session}_date_time", ""))
            for turn_index, turn in enumerate(conversation[session]):
                dia_id = turn["dia_id"]
                if dia_id in turn_to_passage or dia_id in current_ids:
                    raise ValueError(f"dia_id duplicado: {dia_id}")
                lines = ([f"Session date: {date}"] if date and turn_index == 0 else []) + turn_lines(turn, date)
                if current_ids and token_counter("\n".join(current_lines + lines)) > chunk_tokens:
                    flush()
                    lines = ([f"Session date: {date}"] if date else []) + turn_lines(turn, date)
                current_lines.extend(lines)
                current_ids.append(dia_id)
        flush()
    else:
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
                    lines.extend(turn_lines(turn, date))
                passages.append({"title": title, "text": "\n".join(lines),
                                 "session_time": date, "sequence": len(passages),
                                 "source_ids": [str(turn["dia_id"]) for turn in block]})
    if not passages:
        raise ValueError("conversa vazia")
    questions, mapping, evidence_repairs = [], {}, []
    for qi, qa in enumerate(sample["qa"]):
        category = qa["category"]
        if category not in CATEGORIES:
            continue
        evidence = []
        for ref in qa.get("evidence", []):
            # A release alterna entre ponto e vírgula, vírgula e espaços para
            # separar IDs (por exemplo, 'D9:1 D4:4 D4:6'). Separar apenas em
            # delimitadores preserva os reparos pontuais de IDs malformados.
            refs = [s for s in re.split(r"[;,\s]+", ref.strip()) if s]
            for dia_id in refs:
                repaired = EVIDENCE_REPAIRS.get((sample_id, qi, dia_id))
                if repaired is not None:
                    evidence_repairs.append({"qa_index": qi, "original": dia_id,
                                             "replacement": repaired})
                    dia_id = repaired
                if dia_id not in turn_to_passage:
                    raise ValueError(f"QA {qi}: evidência desconhecida {dia_id!r}")
                if dia_id not in evidence:
                    evidence.append(dia_id)
        # The pinned release contains a small number of category-3 inference
        # questions without annotated dialogue evidence (for example QA 30 in
        # conv00). They remain in answer evaluation, while retrieval recall is
        # undefined. Categories 1, 2 and 4 require evidence; accepting an empty
        # annotation there would silently corrupt their retrieval denominator.
        if not evidence and category != 3:
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
                "chunk_tokens": chunk_tokens, "tokenizer_name": tokenizer_name if chunk_tokens else None,
                "temporal_annotations": temporal_annotations,
                "tokenizer_revision": tokenizer_revision if chunk_tokens else None,
                "corpus_scope": "locomo_full_selected_conversation", "seed": seed,
                "text_policy": "speaker + dialog id + date + text + released BLIP captions; no summaries/personas/QA",
                "metric_policy": "official LoCoMo F1/EM plus diagnostic BLEU-1 for categories 1-4; recall at passage-block level",
                "evidence_repairs": evidence_repairs,
                "question_mapping": {q["id"]: mapping[q["id"]] for q in questions}}
    return questions, passages, metadata


def _fetch_with_retry(url, attempts=5, timeout=60):
    """GET com backoff: o handshake TLS para raw.githubusercontent.com
    ocasionalmente sofre reset de conexão (WinError 10054); um único urlopen
    sem retry derruba a rodada inteira por um soluço passageiro de rede."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
            if attempt + 1 == attempts:
                break
            time.sleep(min(30.0, 2.0 ** attempt))
    raise last_exc


def prepare(output, source_file=None, conversation_index=0, turns_per_passage=8,
            n_questions=None, seed=42, max_passages=1500, chunk_tokens=None,
            tokenizer_name="Qwen/Qwen2.5-14B-Instruct", tokenizer_revision=None,
            temporal_annotations=False):
    output = Path(output)
    snapshot = output / "source-data" / "locomo10.json"
    if source_file:
        payload = Path(source_file).read_bytes()
        source = str(Path(source_file).resolve())
    elif snapshot.exists():
        payload = snapshot.read_bytes()
        source = "saved snapshot (origin not revalidated)"
    else:
        payload = _fetch_with_retry(URL)
        source = URL
    questions, passages, metadata = convert(json.loads(payload), conversation_index,
        turns_per_passage, n_questions, seed, chunk_tokens, tokenizer_name, tokenizer_revision,
        temporal_annotations=temporal_annotations)
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
    p.add_argument("--chunk-tokens", type=int)
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-14B-Instruct")
    args = p.parse_args()
    meta = prepare(args.output, args.source_file, args.conversation_index, args.turns_per_passage,
                   chunk_tokens=args.chunk_tokens, tokenizer_name=args.tokenizer)
    print(json.dumps({k: v for k, v in meta.items() if k != "question_mapping"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
