"""Download immutable Parquet data, validate it, and reproduce official chunks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import protocol as P
from .vendor.templates import get_template


@dataclass(frozen=True)
class Sample:
    split: str
    source: str
    row_index: int
    context: str
    questions: list[str]
    answers: list[list[str]]
    metadata: dict

    @property
    def key(self) -> str:
        return f"{self.split}:{self.row_index}:{P.digest(self.context)[:16]}"

    def qid(self, index: int) -> str:
        # Dataset QA IDs can repeat across rows; row identity is always retained.
        return f"{self.key}:q{index}"

    def official_id(self, index: int) -> str | None:
        ids = self.metadata.get("qa_pair_ids") or []
        return str(ids[index]) if index < len(ids) else None

    def evaluation_answers(self, index: int):
        # ConversationCreator passes the entire answers array when the context
        # has one question. Preserve that shape in exports and judge prompts.
        return self.answers if len(self.questions) == 1 else self.answers[index]

    def query(self, index: int) -> str:
        # All current official RAG query templates consume only {question}.
        # Answers, keypoints, qtypes and temporal gold annotations stay private.
        return get_template(self.source, "query", "WitnessRAG_rag").format(
            question=self.questions[index])


def dataset_root(cache: Path) -> Path:
    return cache / "data" / P.DATASET_REVISION


def verify_data(cache: Path) -> dict:
    root = dataset_root(cache)
    verified = {}
    for name, expected in P.DATA_FILES.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run the prepare command first.")
        actual = {"size": path.stat().st_size, "sha256": P.file_digest(path)}
        if actual != expected:
            raise ValueError(f"Dataset checksum mismatch: {path}")
        verified[name] = actual
    return verified


def tokenizer_identity(cache: Path) -> dict:
    configure_nltk(cache)
    import nltk
    import tiktoken
    root = Path(str(nltk.data.find("tokenizers/punkt_tab/english")))
    return {"nltk_version": nltk.__version__, "tiktoken_version": tiktoken.__version__,
            "encoding": tiktoken.encoding_for_model("gpt-4o-mini").name,
            "english_punkt": {p.name: P.file_digest(p) for p in sorted(root.glob("*.txt"))},
            "english_collocations": P.file_digest(root / "collocations.tab"),
            "english_orthography": P.file_digest(root / "ortho_context.tab")}


def configure_nltk(cache: Path) -> None:
    import nltk
    location = str((cache / "nltk_data").resolve())
    if location not in nltk.data.path:
        nltk.data.path.insert(0, location)
    try:
        nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError as exc:
        raise RuntimeError("NLTK English tokenizer missing. Run prepare first.") from exc


def prepare(cache: Path) -> dict:
    from huggingface_hub import hf_hub_download
    import nltk
    for name in P.DATA_FILES:
        hf_hub_download(P.DATASET, name, repo_type="dataset", revision=P.DATASET_REVISION,
                        local_dir=dataset_root(cache))
    nltk_dir = cache / "nltk_data"
    nltk_dir.mkdir(parents=True, exist_ok=True)
    # NLTK >=3.9 uses the tab parameters, not the old pickle package. Avoid a
    # network refresh on every resumed run when the local resource exists.
    if str(nltk_dir.resolve()) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_dir.resolve()))
    try:
        nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError:
        import time
        for attempt in range(3):
            if nltk.download("punkt_tab", download_dir=str(nltk_dir), quiet=True):
                break
            time.sleep(attempt + 1)
        else:
            raise RuntimeError("Could not download the official NLTK English sentence tokenizer")
    files = verify_data(cache)
    samples = list(iter_samples(cache))
    counts = {}
    for sample in samples:
        row = counts.setdefault(sample.source, {"split": sample.split, "contexts": 0, "questions": 0})
        row["contexts"] += 1
        row["questions"] += len(sample.questions)
    manifest = {"dataset": P.DATASET, "revision": P.DATASET_REVISION,
                "upstream_commit": P.UPSTREAM_COMMIT, "files": files,
                "tokenizer": tokenizer_identity(cache), "sources": counts,
                "contexts": len(samples), "questions": sum(len(s.questions) for s in samples)}
    (cache / "prepared.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def iter_samples(cache: Path, splits: tuple[str, ...] = P.SPLITS,
                 sources: tuple[str, ...] = (), suite: str = "all") -> Iterator[Sample]:
    import pyarrow.parquet as pq
    if suite not in ("all", "paper"):
        raise ValueError(f"Unknown suite: {suite}")
    for split in splits:
        if split not in P.SPLITS:
            raise ValueError(f"Unknown split: {split}")
        path = dataset_root(cache) / "data" / f"{split}-00000-of-00001.parquet"
        rows = pq.read_table(path).to_pylist()
        for index, row in enumerate(rows):
            meta = row["metadata"]
            source = meta["source"]
            if sources and source not in sources:
                continue
            if suite == "paper" and source not in P.PAPER_SOURCES:
                continue
            questions, answers = row["questions"], row["answers"]
            if source not in P.CONFIGS:
                raise ValueError(f"Unrecognized dataset source: {source}")
            if len(questions) != len(answers) or not questions:
                raise ValueError(f"Mismatched or empty QA fields: {split} row {index}")
            if not isinstance(row["context"], str) or len(row["context"]) <= 2000:
                raise ValueError(f"Invalid context: {split} row {index}")
            if not all(isinstance(q, str) and q for q in questions):
                raise ValueError(f"Invalid questions: {split} row {index}")
            if not all(isinstance(a, list) and a and all(isinstance(x, str) for x in a)
                       for a in answers):
                raise ValueError(f"Invalid answers: {split} row {index}")
            ids = meta.get("qa_pair_ids") or []
            if ids and len(ids) != len(questions):
                raise ValueError(f"Mismatched official IDs: {split} row {index}")
            yield Sample(split, source, index, row["context"], questions, answers, meta)


def chunk_text(text: str, chunk_size: int, *, sentences=None, encoding=None) -> list[str]:
    """Exact release algorithm, including its oversized-sentence behavior.

    No overlap, no hard cutting long sentences, no raw-character truncation.
    Token counts exclude joining spaces, just like the official helper.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if encoding is None:
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    if sentences is None:
        import nltk
        sentences = nltk.sent_tokenize(text)
    chunks, current, count = [], [], 0
    for sentence in sentences:
        length = len(encoding.encode(sentence, allowed_special={"<|endoftext|>"}))
        if count + length > chunk_size:
            chunks.append(" ".join(current))
            current, count = [sentence], length
        else:
            current.append(sentence)
            count += length
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunks_for(sample: Sample, cache: Path, settings: dict) -> list[str]:
    configure_nltk(cache)
    import tiktoken
    key = P.digest([sample.context, settings["chunk_size"], tokenizer_identity(cache), tiktoken.__version__])
    path = cache / "chunks" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    chunks = chunk_text(sample.context, settings["chunk_size"])
    path.parent.mkdir(parents=True, exist_ok=True)
    # Concurrent preparations write equivalent payloads; replacement is atomic.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return chunks


def memorize_message(source: str, chunk: str, index: int) -> str:
    template = get_template(source, "memorize", "WitnessRAG_rag")
    # Release uses wall-clock time here; it is NOT event recency. Use stable
    # ingestion order so caches/resumes don't change the experimental memory.
    return template.format(context=chunk, time_stamp=f"[ingestion step {index}; not an event date]")
