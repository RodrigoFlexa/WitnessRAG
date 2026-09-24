"""
Dados do protocolo do GAM: download verificado, preparação e carregamento.

`prepare` baixa os arquivos exatos (repositório e commit fixados em
protocol.py, SHA-256 conferido) e grava cada benchmark no formato que o
código de avaliação do GAM lê:

* HotpotQA: os três JSON do MemAgent, sem alteração.
* RULER: cada parquet vira `<tarefa>.jsonl` com `split_input` do GAM
  (download_data/download_ruler.py), que separa exemplo, instrução, contexto
  e pergunta; as demais colunas (index, outputs, length) são mantidas.
* NarrativeQA: a ordem do split de teste é a dos oito shards em sequência;
  `random.seed(42)` e `random.shuffle` sobre as 10.557 perguntas, e ficam as
  300 primeiras (eval_narrativeqa.sh: --seed 42 --end-idx 300). Só os livros
  e roteiros dessas perguntas são gravados, uma vez cada.

Os carregadores devolvem `Sample` com o `_id`, o `index` e a pergunta
exatamente como o GAM os monta.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from wrag.gambench import protocol as P
from wrag.util import get_logger, write_json

log = get_logger("wrag.gambench.data")


@dataclass
class Sample:
    sid: str                 # o `_id` do GAM
    benchmark: str
    split: str               # 56k/224k/448k, a tarefa do RULER ou "test"
    index: int               # o `index` do GAM
    question: str            # pergunta usada na busca e no prompt
    answers: list[str]
    context: str
    context_key: str         # contextos iguais (NarrativeQA) compartilham a chave
    example: str = ""        # RULER: exemplo do prompt (VT e CWE)
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: Path, remote: P.RemoteFile) -> bool:
    return path.exists() and path.stat().st_size == remote.size and _sha256(path) == remote.sha256


def _verify_arg() -> Any:
    for name in ("WRAG_HF_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return True


def fetch(remote: P.RemoteFile, dest_dir: Path, source_dir: Path | None = None) -> Path:
    """Garante `dest_dir/<nome>` idêntico ao arquivo fixado; baixa se preciso.

    Com `source_dir`, copia de lá (para servidores sem acesso ao Hugging Face)
    e confere do mesmo jeito.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / remote.name
    if _verified(target, remote):
        return target
    partial = target.with_suffix(target.suffix + ".part")
    if source_dir is not None:
        candidates = [source_dir / remote.name, source_dir / remote.path]
        source = next((c for c in candidates if c.exists()), None)
        if source is None:
            raise FileNotFoundError(f"{remote.name} não está em {source_dir}")
        shutil.copyfile(source, partial)
    else:
        import httpx

        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        url = f"{endpoint}/datasets/{remote.repo}/resolve/{remote.revision}/{remote.path}"
        headers = {}
        token = os.environ.get("HF_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        print(f"baixando {remote.repo}/{remote.path} ({remote.size / 2**20:.0f} MB)", flush=True)
        started, last = time.time(), 0.0
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, read=300.0),
                          verify=_verify_arg()) as client:
            with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                done = 0
                with partial.open("wb") as fh:
                    for block in response.iter_bytes(1 << 20):
                        fh.write(block)
                        done += len(block)
                        if time.time() - last > 15:
                            last = time.time()
                            print(f"  {done / 2**20:.0f}/{remote.size / 2**20:.0f} MB "
                                  f"({time.time() - started:.0f}s)", flush=True)
    if partial.stat().st_size != remote.size or _sha256(partial) != remote.sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{remote.name}: tamanho ou SHA-256 diferente do fixado "
                           f"({remote.repo}@{remote.revision[:12]})")
    partial.replace(target)
    return target


# ---------------------------------------------------------------------------
# RULER: split_input do GAM
# ---------------------------------------------------------------------------

_NIAH_NUMBER = ("A special magic number is hidden within the following text. "
                "Make sure to memorize it. I will quiz you about the number afterwards.",
                "What is the special magic number")
_NIAH_UUID = ("A special magic uuid is hidden within the following text. "
              "Make sure to memorize it. I will quiz you about the uuid afterwards.",
              "What is the special magic uuid")
_NIAH_NUMBERS = ("Some special magic numbers are hidden within the following text. "
                 "Make sure to memorize it. I will quiz you about the numbers afterwards.",
                 "What are all the special magic numbers")
# No original cada tarefa NIAH tem um bloco próprio; os blocos são idênticos a
# menos da instrução e do início da pergunta, que ficam nesta tabela.
_NIAH = {"niah_single_1": _NIAH_NUMBER, "niah_single_2": _NIAH_NUMBER,
         "niah_multikey_1": _NIAH_NUMBER, "niah_multikey_2": _NIAH_NUMBER,
         "niah_single_3": _NIAH_UUID, "niah_multikey_3": _NIAH_UUID,
         "niah_multiquery": _NIAH_NUMBERS, "niah_multivalue": _NIAH_NUMBERS}
_CWE_INSTR = ("Below is a numbered list of words. In these words, some appear more often "
              "than others. Memorize the ones that appear most often.")
_FWE_INSTR = ("Read the following coded text and track the frequency of each coded word. "
              "Find the three most frequently appeared coded words.")
_QA_INSTR = ("Answer the question based on the given documents. Only give me the answer and "
             "do not output any other words.")
_VT_INSTR = "Memorize and track the chain(s) of variable assignment hidden in the following text."


def split_input(text: Any, dataset: str) -> tuple[str, str, str, str]:
    """(example, instruction, context, question), como no GAM."""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return "", "", "", ""
    example = instruction = context = question = ""

    if dataset == "cwe":
        instr_positions = []
        start = 0
        while True:
            idx = text.find(_CWE_INSTR, start)
            if idx == -1:
                break
            instr_positions.append(idx)
            start = idx + len(_CWE_INSTR)
        if instr_positions:
            instr_start = instr_positions[-1]
            instr_end = instr_start + len(_CWE_INSTR)
            instruction = _CWE_INSTR
            if instr_start > 0:
                example = text[:instr_start].strip()
            q_pos = text.find("Question:", instr_end)
            if q_pos != -1:
                context = text[instr_end:q_pos].strip()
                question = text[q_pos:].strip()
            else:
                context = text[instr_end:].strip()
        else:
            q_pos = text.find("Question:")
            if q_pos != -1:
                context = text[:q_pos].strip()
                question = text[q_pos:].strip()
            else:
                context = text
        return example, instruction, context, question

    if dataset == "fwe":
        q_pos = text.rfind("Question:")
        if q_pos != -1:
            question = text[q_pos:].strip()
        else:
            q_pos = -1
        instr_start = text.find(_FWE_INSTR)
        if instr_start != -1:
            instr_end = instr_start + len(_FWE_INSTR)
            instruction = _FWE_INSTR
            end_for_context = q_pos if q_pos != -1 else len(text)
            context = text[instr_end:end_for_context].strip()
        else:
            context = text if q_pos == -1 else text[:q_pos].strip()
        return example, instruction, context, question

    if dataset in _NIAH:
        magic_instr, q_key = _NIAH[dataset]
        instr_start = text.find(magic_instr)
        if instr_start != -1:
            instr_end = instr_start + len(magic_instr)
            instruction = magic_instr
            q_pos = text.rfind(q_key)
            if q_pos != -1 and q_pos > instr_end:
                context = text[instr_end:q_pos].strip()
                question = text[q_pos:].strip()
            else:
                context = text[instr_end:].strip()
        else:
            context = text
        return example, instruction, context, question

    if dataset in ("qa_1", "qa_2"):
        instr_start = text.find(_QA_INSTR)
        if instr_start != -1:
            instr_end = instr_start + len(_QA_INSTR)
            instruction = _QA_INSTR
        else:
            instr_end = 0
        q_pos = text.rfind("Question:")
        if q_pos != -1:
            context = text[instr_end:q_pos].strip()
            question = text[q_pos:].strip()
        else:
            context = text[instr_end:].strip()
        return example, instruction, context, question

    if dataset == "vt":
        first = text.find(_VT_INSTR)
        second = text.find(_VT_INSTR, first + len(_VT_INSTR)) if first != -1 else -1
        if first != -1 and second != -1:
            example = text[first:second].strip()
            instruction = _VT_INSTR
            rest = text[second:].strip()
            instr_start2 = rest.find(_VT_INSTR)
            instr_end2 = instr_start2 + len(_VT_INSTR) if instr_start2 != -1 else 0
            q_pos = rest.rfind("Question:")
            if q_pos != -1 and q_pos > instr_end2:
                context = rest[instr_end2:q_pos].strip()
                question = rest[q_pos:].strip()
            else:
                context = rest[instr_end2:].strip()
        else:
            instr_start = text.find(_VT_INSTR)
            if instr_start != -1:
                before = text[:instr_start].strip()
                if before:
                    example = before
                instr_end = instr_start + len(_VT_INSTR)
                instruction = _VT_INSTR
            else:
                instr_end = 0
            q_pos = text.rfind("Question:")
            if q_pos != -1 and q_pos > instr_end:
                context = text[instr_end:q_pos].strip()
                question = text[q_pos:].strip()
            else:
                context = text[instr_end:].strip()
        return example, instruction, context, question

    return example, instruction, text, question


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} não é serializável")


def convert_ruler_parquet(parquet: Path, task: str, output: Path) -> int:
    """`process_dataset` do GAM: uma linha JSON por amostra, colunas extras mantidas."""
    import pyarrow.parquet as pq

    table = pq.ParquetFile(parquet)
    columns = table.schema_arrow.names
    text_col = next((c for c in ("input", "text", "content", "prompt") if c in columns), None)
    if text_col is None:
        raise ValueError(f"{parquet}: coluna de texto ausente em {columns}")
    partial = output.with_suffix(".jsonl.part")
    count = 0
    with partial.open("w", encoding="utf-8") as fh:
        for batch in table.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                example, instruction, context, question = split_input(row[text_col], task)
                sample = {"example": example, "instruction": instruction,
                          "context": context, "question": question}
                for key in columns:
                    if key != text_col:
                        sample[key] = row[key]
                fh.write(json.dumps(sample, ensure_ascii=False, default=_jsonable) + "\n")
                count += 1
    partial.replace(output)
    return count


# ---------------------------------------------------------------------------
# preparação
# ---------------------------------------------------------------------------

def prepare(data_dir: Path, benchmarks: Iterable[str] = P.BENCHMARKS,
            source_dir: Path | None = None, keep_raw: bool = True) -> dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "gam_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({"protocolo": P.GAM_PAPER, "gam_repo": P.GAM_REPO,
                     "gam_commit": P.GAM_COMMIT})
    for benchmark in benchmarks:
        if benchmark == "hotpotqa":
            files = {}
            for name in P.HOTPOT_SPLITS.values():
                path = fetch(P.HOTPOT_FILES[name], data_dir / "hotpotqa", source_dir)
                files[name] = {"sha256": P.HOTPOT_FILES[name].sha256,
                               "amostras": len(_read_json_list(path))}
            manifest["hotpotqa"] = {"repo": P.HOTPOT_REPO, "revisao": P.HOTPOT_REVISION,
                                    "arquivos": files}
        elif benchmark == "ruler":
            raw_dir = data_dir / "ruler" / "raw"
            counts = {}
            for task in P.RULER_TASKS:
                output = data_dir / "ruler" / f"{task}.jsonl"
                raw = fetch(P.RULER_FILES[task], raw_dir, source_dir)
                marker = output.with_suffix(".sha256")
                if not (output.exists() and marker.exists()
                        and marker.read_text().strip() == P.RULER_FILES[task].sha256):
                    print(f"convertendo RULER {task} (split_input do GAM)", flush=True)
                    convert_ruler_parquet(raw, task, output)
                    marker.write_text(P.RULER_FILES[task].sha256 + "\n")
                counts[task] = _count_lines(output)
                if not keep_raw:
                    raw.unlink(missing_ok=True)
            manifest["ruler"] = {"repo": P.RULER_REPO, "revisao": P.RULER_REVISION,
                                 "amostras": counts}
        elif benchmark == "narrativeqa":
            raw_dir = data_dir / "narrativeqa" / "raw"
            shards = [fetch(remote, raw_dir, source_dir) for remote in P.NARRATIVEQA_FILES]
            info = _prepare_narrativeqa(shards, data_dir / "narrativeqa")
            if not keep_raw:
                for shard in shards:
                    shard.unlink(missing_ok=True)
            manifest["narrativeqa"] = {"repo": P.NARRATIVEQA_REPO,
                                       "revisao": P.NARRATIVEQA_REVISION, **info}
        else:
            raise ValueError(f"benchmark desconhecido: {benchmark}")
        write_json(manifest_path, manifest)
    return manifest


def _count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path}: esperava uma lista JSON")
    return data


def narrativeqa_selection(total: int, seed: int = P.NARRATIVEQA_SEED,
                          n: int = P.NARRATIVEQA_SAMPLES) -> list[int]:
    """Índices originais das perguntas escolhidas, na ordem de avaliação.

    `random.shuffle` de uma lista depende só do comprimento e do estado do
    gerador, então embaralhar os índices dá a mesma permutação que o GAM
    aplica à lista de amostras.
    """
    order = list(range(total))
    random.seed(seed)
    random.shuffle(order)
    return order[:n]


def _prepare_narrativeqa(shards: list[Path], out_dir: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    questions_path = out_dir / "questions.jsonl"
    documents_path = out_dir / "documents.jsonl"
    stamp = hashlib.sha256("".join(r.sha256 for r in P.NARRATIVEQA_FILES).encode()).hexdigest()
    marker = out_dir / "selection.sha256"
    if questions_path.exists() and documents_path.exists() and marker.exists() \
            and marker.read_text().strip() == stamp:
        return json.loads((out_dir / "selection.json").read_text())

    # Primeira passada: só metadados (a coluna do documento repete o livro
    # inteiro em cada pergunta; lê-la toda de uma vez passa de 4 GB).
    rows: list[dict[str, Any]] = []
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=64,
                                                        columns=["document", "question", "answers"]):
            for row in batch.to_pylist():
                document = row.get("document") or {}
                question = row.get("question") or {}
                answers = [a.get("text", "") for a in (row.get("answers") or [])
                           if isinstance(a, dict) and a.get("text")]
                rows.append({"index": len(rows), "document_id": document.get("id", f"doc-{len(rows)}"),
                             "question": question.get("text", "") if isinstance(question, dict) else "",
                             "answers": answers})
    if len(rows) != P.NARRATIVEQA_TEST_SIZE:
        raise RuntimeError(f"NarrativeQA test tem {len(rows)} perguntas; esperado "
                           f"{P.NARRATIVEQA_TEST_SIZE}")
    chosen = narrativeqa_selection(len(rows))
    if tuple(chosen[:5]) != P.NARRATIVEQA_FIRST_INDICES:
        raise RuntimeError("a seleção do NarrativeQA não reproduz a do GAM")
    wanted = {rows[i]["document_id"] for i in chosen}

    texts: dict[str, dict[str, Any]] = {}
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=32, columns=["document"]):
            for row in batch.to_pylist():
                document = row.get("document") or {}
                doc_id = document.get("id")
                if doc_id in wanted and doc_id not in texts:
                    summary = document.get("summary") or {}
                    texts[doc_id] = {"document_id": doc_id, "kind": document.get("kind", ""),
                                     "title": summary.get("title", "") if isinstance(summary, dict) else "",
                                     "text": document.get("text", "") or ""}
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "questions.jsonl.part").open("w", encoding="utf-8") as fh:
        for order, index in enumerate(chosen):
            fh.write(json.dumps({"order": order, **rows[index]}, ensure_ascii=False) + "\n")
    with (out_dir / "documents.jsonl.part").open("w", encoding="utf-8") as fh:
        for doc_id in sorted(texts):
            fh.write(json.dumps(texts[doc_id], ensure_ascii=False) + "\n")
    (out_dir / "questions.jsonl.part").replace(questions_path)
    (out_dir / "documents.jsonl.part").replace(documents_path)
    info = {"perguntas_teste": len(rows), "selecionadas": len(chosen),
            "documentos": len(texts), "semente": P.NARRATIVEQA_SEED,
            "primeiros_indices": chosen[:5]}
    write_json(out_dir / "selection.json", info)
    marker.write_text(stamp + "\n")
    return info


# ---------------------------------------------------------------------------
# carregadores (formato do GAM)
# ---------------------------------------------------------------------------

def splits_of(benchmark: str) -> tuple[str, ...]:
    if benchmark == "hotpotqa":
        return tuple(P.HOTPOT_SPLITS)
    if benchmark == "ruler":
        return P.RULER_TASKS
    if benchmark == "narrativeqa":
        return ("test",)
    raise ValueError(f"benchmark desconhecido: {benchmark}")


def load(data_dir: Path, benchmark: str, split: str) -> list[Sample]:
    if benchmark == "hotpotqa":
        return load_hotpotqa(data_dir / "hotpotqa" / P.HOTPOT_SPLITS[split], split)
    if benchmark == "ruler":
        return load_ruler(data_dir / "ruler" / f"{split}.jsonl", split)
    if benchmark == "narrativeqa":
        return load_narrativeqa(data_dir / "narrativeqa")
    raise ValueError(f"benchmark desconhecido: {benchmark}")


def load_hotpotqa(path: Path, split: str) -> list[Sample]:
    """`load_hotpotqa` do GAM: _id = hotpotqa-<index>, pergunta = input."""
    samples = []
    for idx, item in enumerate(_read_json_list(path)):
        index = item.get("index", idx)
        sid = f"hotpotqa-{index}"
        samples.append(Sample(sid=sid, benchmark="hotpotqa", split=split, index=int(index),
                              question=item.get("input", ""), answers=list(item.get("answers", [])),
                              context=item.get("context", "") or "", context_key=sid,
                              meta={"num_docs": item.get("num_docs")}))
    _check_unique(samples, path)
    return samples


def load_ruler(path: Path, task: str) -> list[Sample]:
    """`load_ruler_jsonl` do GAM: _id = <tarefa>-<linha>; a busca usa só a
    pergunta, o exemplo entra apenas no prompt de resposta."""
    samples = []
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if not line.strip():
                continue
            item = json.loads(line)
            sid = f"{task}-{idx}"
            samples.append(Sample(sid=sid, benchmark="ruler", split=task, index=idx,
                                  question=(item.get("question") or "").strip(),
                                  answers=[str(x) for x in item.get("outputs") or []],
                                  context=item.get("context") or "", context_key=sid,
                                  example=item.get("example") or "",
                                  meta={"length": item.get("length"),
                                        "ruler_index": item.get("index")}))
    _check_unique(samples, path)
    return samples


def load_narrativeqa(directory: Path) -> list[Sample]:
    """As 300 perguntas na ordem do GAM; _id = narrativeqa-<doc>-<índice original>."""
    texts: dict[str, str] = {}
    with (directory / "documents.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                texts[row["document_id"]] = row.get("text", "")
    samples = []
    with (directory / "questions.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            doc = row["document_id"]
            samples.append(Sample(sid=f"narrativeqa-{doc}-{row['index']}", benchmark="narrativeqa",
                                  split="test", index=int(row["index"]), question=row["question"],
                                  answers=list(row["answers"]), context=texts.get(doc, ""),
                                  context_key=f"narrativeqa-doc-{doc}",
                                  meta={"document_id": doc, "order": row.get("order")}))
    _check_unique(samples, directory)
    return samples


def _check_unique(samples: list[Sample], source: Any) -> None:
    seen: set[str] = set()
    for sample in samples:
        if sample.sid in seen:
            raise ValueError(f"{source}: _id repetido {sample.sid}")
        seen.add(sample.sid)


def iter_selected(samples: list[Sample], start: int = 0, end: int | None = None) -> Iterator[Sample]:
    """`--start-idx/--end-idx` do GAM (end exclusivo)."""
    yield from samples[start: len(samples) if end is None else min(end, len(samples))]
