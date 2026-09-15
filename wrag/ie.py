"""
Extração aberta (OpenIE) compartilhada por todos os métodos com grafo.

Esta é a variável que a proposta manda controlar. GraphRAG, HippoRAG, HippoRAG 2
e WITNESS-RAG consomem exatamente a mesma base de fatos `F`, extraída uma vez por
corpus e cacheada em disco. Sem isso, uma diferença de 3 pontos de recall entre
dois métodos poderia ser inteiramente diferença de extrator, e a comparação que
mais interessa — "o WITNESS-RAG supera um executor relacional convencional sobre
os MESMOS fatos extraídos?" — ficaria impossível de fazer.

Passagens que o filtro de conteúdo do Azure recusa entram no `Ledger` como
`stage="index"` e simplesmente não contribuem fatos. Elas continuam no corpus e
continuam recuperáveis pelo RAG denso — o que é o comportamento honesto: o
prejuízo do bloqueio recai sobre os métodos com grafo, e o relatório mostra
quantas passagens foram perdidas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

from wrag import config as C
from wrag import prompts
from wrag.data import Corpus
from wrag.llm import LLM, GenParams
from wrag.llm.filters import LEDGER
from wrag.util import get_logger, canonical_symbol as normalize, read_json, sha, write_json

log = get_logger("wrag.ie")


@dataclass
class Fact:
    """Um fato com proveniência.

    `pid` é a passagem que o sustenta — a "fonte" e o "trecho" do
    f = (predicado, argumentos, fonte, trecho, tempo, status) da proposta.
    `acquired=True` marca fatos criados em tempo de consulta pela aquisição
    adaptativa, para que o custo deles seja contabilizado separadamente.
    """

    fid: str
    subject: str
    relation: str
    object: str
    pid: str
    confidence: float = 0.9
    acquired: bool = False
    time: str = ""        # escopo temporal declarado no texto; "" quando não há
    subj_id: int = -1     # id canônico de entidade, preenchido na resolução
    obj_id: int = -1
    rel_id: int = -1      # id canônico de relação

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, self.object)

    def verbalize(self) -> str:
        # O tempo entra na verbalização porque é o que distingue dois fatos com a
        # mesma tripla em sessões diferentes; a tripla em si continua com três
        # elementos, e as métricas de cobertura seguem comparando triplas.
        base = f"{self.subject} {self.relation} {self.object}"
        return f"{base} ({self.time})" if self.time else base

    def to_dict(self) -> dict[str, Any]:
        out = {"fid": self.fid, "s": self.subject, "r": self.relation, "o": self.object,
               "pid": self.pid, "conf": self.confidence, "acq": self.acquired}
        if self.time:
            out["t"] = self.time
        return out


@dataclass
class ExtractionResult:
    facts: list[Fact] = field(default_factory=list)
    entities_by_passage: dict[str, list[str]] = field(default_factory=dict)
    blocked_pids: list[str] = field(default_factory=list)
    empty_pids: list[str] = field(default_factory=list)
    extraction_windows: int = 0
    blocked_windows: int = 0

    def stats(self) -> dict[str, Any]:
        """Contagens da extração, incluindo a fragmentação do vocabulário.

        `relacoes_por_fato` perto de 1 e `relacoes_unicas` alto significam que o
        predicado virou uma frase diferente a cada fato. Um vocabulário assim não
        sustenta junção: cada átomo alcança um ou dois fatos e a variável
        compartilhada nunca casa. É a forma mais barata de ver um prompt de
        extração degenerar, e custa uma passada sobre os fatos já extraídos.
        """
        counts: dict[str, int] = {}
        for fact in self.facts:
            key = normalize(fact.relation)
            counts[key] = counts.get(key, 0) + 1
        entities = {normalize(f.subject) for f in self.facts} | {normalize(f.object) for f in self.facts}
        total = len(self.facts) or 1
        return {
            "n_fatos": len(self.facts),
            "n_relacoes_distintas": len(counts),
            "n_entidades_distintas": len(entities),
            "n_objetos_distintos": len({normalize(f.object) for f in self.facts}),
            "relacoes_por_fato": round(len(counts) / total, 3),
            "relacoes_unicas": sum(1 for c in counts.values() if c == 1),
            "palavras_por_relacao": round(sum(len(normalize(f.relation).split())
                                              for f in self.facts) / total, 2),
            "passagens_bloqueadas": len(self.blocked_pids),
            "passagens_sem_fato": len(self.empty_pids),
            "janelas_extracao": self.extraction_windows,
            "janelas_bloqueadas": self.blocked_windows,
        }


def _cache_path(corpus: Corpus, llm: LLM, cfg: C.IEConfig) -> Path:
    # O modo diálogo só entra na chave quando está ligado: acrescentar um campo
    # novo à chave invalidaria as extrações já pagas de todos os corpora, e uma
    # extração cara não pode ser descartada por uma opção que não foi usada.
    config = asdict(cfg)
    templates = [prompts.NER_SYSTEM, prompts.NER_TEMPLATE,
                 prompts.OPENIE_SYSTEM, prompts.OPENIE_TEMPLATE]
    if cfg.dialogue_mode:
        templates += [prompts.OPENIE_DIALOGUE_SYSTEM, prompts.OPENIE_DIALOGUE_TEMPLATE]
    else:
        config.pop("dialogue_mode", None)
    if not cfg.window_tokens:
        for name in ("window_tokens", "window_overlap_tokens", "window_tokenizer",
                     "window_tokenizer_revision"):
            config.pop(name, None)
    key = sha({
        "dataset": corpus.name,
        "passages": [(p.pid, p.title, p.text) for p in corpus.passages],
        "backend": llm.name,
        "deployment": getattr(llm, "deployment", ""),
        "provider_identity": llm.cache_identity() if hasattr(llm, "cache_identity") else "",
        "config": config,
        "prompts": templates,
        "prompt_version": 3,
    })
    return C.CACHE_DIR / "openie" / f"{corpus.name}-{key[:16]}.json"


def extract_corpus(
    corpus: Corpus,
    llm: LLM,
    cfg: C.IEConfig | None = None,
    use_cache: bool = True,
) -> ExtractionResult:
    """Extrai a base de fatos do corpus inteiro. Cacheada por corpus + backend."""
    cfg = cfg or C.IEConfig()
    path = _cache_path(corpus, llm, cfg)
    if use_cache:
        cached = read_json(path)
        if cached:
            result = ExtractionResult(
                facts=[Fact(fid=r["fid"], subject=r["s"], relation=r["r"], object=r["o"],
                            pid=r["pid"], confidence=r.get("conf", 0.9), time=r.get("t", ""))
                       for r in cached["facts"]],
                entities_by_passage=cached.get("entities", {}),
                blocked_pids=cached.get("blocked", []),
                empty_pids=cached.get("empty", []),
                extraction_windows=cached.get("extraction_windows", 0),
                blocked_windows=cached.get("blocked_windows", 0),
            )
            for pid in result.blocked_pids:
                LEDGER.add("index", corpus.name, "shared-ie", pid, "cache: bloqueado na extração")
            log.info("OpenIE em cache: %s", result.stats())
            return result

    passages = corpus.passages
    params = GenParams(temperature=cfg.temperature, max_tokens=cfg.max_tokens, json_mode=True)

    # As mesmas janelas alimentam NER e OpenIE. Assim, um chunk de 2.048 tokens
    # não sofre nem o teto global de 60 entidades nem o de 40 triplas.
    units = [(p, wi, text) for p in passages
             for wi, text in enumerate(_extraction_windows(p.full, cfg))]

    # -- passo 1: NER
    entities_by_passage: dict[str, list[str]] = {}
    entities_by_unit: dict[tuple[str, int], list[str]] = {}
    blocked_units: set[tuple[str, int]] = set()
    blocked: set[str] = set()
    if cfg.two_step:
        ner_prompts = [prompts.NER_TEMPLATE.format(text=text) for _p, _wi, text in units]
        ner_results = llm.chat_many(ner_prompts, system=prompts.NER_SYSTEM, params=params,
                                    stage="index.ner", desc="NER das passagens")
        for (passage, window_index, _text), result in zip(units, ner_results):
            if result.filtered:
                blocked_units.add((passage.pid, window_index))
                LEDGER.add("index", corpus.name, "shared-ie", passage.pid,
                           f"NER bloqueado; janela={window_index}")
                continue
            data = result.json()
            data = data if isinstance(data, dict) else {}
            found = [
                str(e).strip() for e in (data.get("named_entities") or []) if str(e).strip()
            ][:60]
            entities_by_unit[(passage.pid, window_index)] = found
            known = entities_by_passage.setdefault(passage.pid, [])
            seen = {normalize(e) for e in known}
            for entity in found:
                key = normalize(entity)
                if key not in seen:
                    known.append(entity)
                    seen.add(key)

    # -- passo 2: OpenIE condicionado às entidades. O leitor e o recuperador
    # continuam vendo as passagens originais; somente a indexação pode quebrar
    # uma passagem longa em janelas. Toda tripla volta ao pid da passagem pai.
    targets = [(p, wi, text) for p, wi, text in units if (p.pid, wi) not in blocked_units]
    template, system = ((prompts.OPENIE_DIALOGUE_TEMPLATE, prompts.OPENIE_DIALOGUE_SYSTEM)
                        if cfg.dialogue_mode else (prompts.OPENIE_TEMPLATE, prompts.OPENIE_SYSTEM))
    ie_prompts = [
        template.format(
            text=text,
            entities=prompts.jdump(entities_by_unit.get((p.pid, wi), [])),
            max_triples=cfg.max_triples_per_passage,
        )
        for p, wi, text in targets
    ]
    ie_results = llm.chat_many(ie_prompts, system=system, params=params,
                               stage="index.openie", desc="OpenIE das passagens")

    facts: list[Fact] = []
    facts_by_parent: dict[str, int] = {p.pid: 0 for p in passages}
    windows_by_parent: dict[str, int] = {}
    blocked_by_parent: dict[str, int] = {}
    for pid, _wi in blocked_units:
        blocked_by_parent[pid] = blocked_by_parent.get(pid, 0) + 1
    for passage, _wi, _text in units:
        windows_by_parent[passage.pid] = windows_by_parent.get(passage.pid, 0) + 1
    for (passage, window_index, _text), result in zip(targets, ie_results):
        if result.filtered:
            blocked_by_parent[passage.pid] = blocked_by_parent.get(passage.pid, 0) + 1
            LEDGER.add("index", corpus.name, "shared-ie", passage.pid,
                       f"OpenIE bloqueado; janela={window_index}")
            continue
        triples = _parse_triples(result.json(), cfg.max_triples_per_passage)
        facts_by_parent[passage.pid] += len(triples)
        facts.extend(_facts_from_triples(triples, passage.pid))

    # Uma janela bloqueada não apaga os fatos das demais. A passagem só entra em
    # blocked_pids quando nenhuma janela pôde ser lida.
    for pid, count in blocked_by_parent.items():
        if count == windows_by_parent.get(pid, 0):
            blocked.add(pid)
    empty = [p.pid for p in passages if p.pid not in blocked and facts_by_parent[p.pid] == 0]

    facts = _dedupe(facts)
    out = ExtractionResult(facts=facts, entities_by_passage=entities_by_passage,
                           blocked_pids=sorted(blocked), empty_pids=empty,
                           extraction_windows=len(units),
                           blocked_windows=sum(blocked_by_parent.values()))
    log.info("OpenIE concluída: %s", out.stats())
    if use_cache:
        write_json(path, {
            "facts": [f.to_dict() for f in facts],
            "entities": entities_by_passage,
            "blocked": out.blocked_pids,
            "empty": empty,
            "extraction_windows": out.extraction_windows,
            "blocked_windows": out.blocked_windows,
        })
    return out


def _extraction_windows(text: str, cfg: C.IEConfig) -> list[str]:
    """Token windows for indexing only; output provenance stays on the parent."""
    if not cfg.window_tokens:
        return [text]
    if cfg.window_tokens < 1 or not 0 <= cfg.window_overlap_tokens < cfg.window_tokens:
        raise ValueError("janela OpenIE exige 0 <= overlap < window_tokens")
    if not cfg.window_tokenizer:
        raise ValueError("window_tokenizer é obrigatório quando window_tokens > 0")
    tokenizer = _load_window_tokenizer(cfg.window_tokenizer, cfg.window_tokenizer_revision)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= cfg.window_tokens:
        return [text]
    step = cfg.window_tokens - cfg.window_overlap_tokens
    return [tokenizer.decode(tokens[start:start + cfg.window_tokens], skip_special_tokens=True)
            for start in range(0, len(tokens), step)
            if tokens[start:start + cfg.window_tokens]]


@lru_cache(maxsize=4)
def _load_window_tokenizer(name: str, revision: str = ""):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, revision=revision or None)


def _parse_triples(data: Any, limit: int) -> list[tuple[str, str, str, str]]:
    """Triplas com um quarto elemento opcional: o tempo, vazio quando não há.

    O modo diálogo pede quatro elementos; o prompt geral pede três. Aceitar os
    dois formatos aqui evita perder uma extração inteira por causa do formato.
    """
    if not isinstance(data, dict):
        return []
    raw = data.get("triples") or data.get("fact") or []
    out: list[tuple[str, str, str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            item = [item.get("subject"), item.get("relation"), item.get("object"),
                    item.get("time")]
        if not isinstance(item, (list, tuple)) or not 3 <= len(item) <= 4:
            continue
        s, r, o = (str(x or "").strip() for x in item[:3])
        when = str(item[3] or "").strip()[:80] if len(item) == 4 else ""
        # Um fato sem sujeito, relação ou objeto não é um fato incompleto: é
        # ruído de extração que envenenaria a junção mais tarde.
        if not s or not r or not o or len(s) > 200 or len(o) > 200:
            continue
        out.append((s, r, o, when))
        if len(out) >= limit:
            break
    return out


def _facts_from_triples(triples: Iterable[Sequence[str]], pid: str,
                        acquired: bool = False, confidence: float | None = None) -> list[Fact]:
    facts = []
    for triple in triples:
        s, r, o = triple[0], triple[1], triple[2]
        when = triple[3] if len(triple) > 3 else ""
        fid = sha(normalize(s), normalize(r), normalize(o), pid)[:16]
        facts.append(Fact(fid=fid, subject=s, relation=r, object=o, pid=pid,
                          confidence=confidence if confidence is not None else 0.9,
                          acquired=acquired, time=when))
    return facts


def _dedupe(facts: Sequence[Fact]) -> list[Fact]:
    """Uma ocorrência por fonte. Duplicação entre fontes não calibra confiança."""
    by_key: dict[tuple[str, str, str], list[Fact]] = {}
    for f in facts:
        by_key.setdefault((normalize(f.subject), normalize(f.relation), normalize(f.object)), []).append(f)

    out: list[Fact] = []
    for group in by_key.values():
        by_pid: dict[str, Fact] = {}
        for f in group:
            by_pid.setdefault(f.pid, f)
        for f in by_pid.values():
            out.append(f)
    return out


def extract_targeted(
    llm: LLM,
    relation: str,
    passages: Sequence[tuple[str, str, str]],
    anchor: str | None = None,
    dataset: str = "",
    method: str = "witnessrag",
    question_id: str = "",
) -> list[Fact]:
    """Extração dirigida a uma relação, usada pela aquisição adaptativa.

    `passages` são triplas (pid, título, texto). Devolve fatos novos marcados
    `acquired=True`, cujo custo o relatório contabiliza no estágio de consulta e
    não no de indexação — que é exatamente a distinção que a proposta cobra ao
    perguntar se "a economia de consulta é anulada pelo custo de extração".
    """
    if not passages:
        return []
    anchor_line = f'The subject or object must be "{anchor}".' if anchor else ""
    batch = [
        prompts.TARGETED_IE_TEMPLATE.format(relation=relation, anchor_line=anchor_line,
                                            text=f"{title}\n{text}")
        for _pid, title, text in passages
    ]
    results = llm.chat_many(batch, system=prompts.TARGETED_IE_SYSTEM,
                            params=GenParams(temperature=0.0, max_tokens=800, json_mode=True),
                            stage="witness.acquire", desc="")
    facts: list[Fact] = []
    for (pid, _title, _text), result in zip(passages, results):
        if result.filtered:
            LEDGER.add("acquire", dataset, method, question_id or pid,
                       f"extração dirigida bloqueada; passagem={pid}")
            continue
        triples = _parse_triples(result.json(), 10)
        # Confiança menor que a da extração geral: o prompt dirigido induz o
        # modelo a encontrar a relação pedida, e essa pressão produz falsos
        # positivos que a extração livre não produziria.
        facts.extend(_facts_from_triples(triples, pid, acquired=True, confidence=0.80))
    return facts
