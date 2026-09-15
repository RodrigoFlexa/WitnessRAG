"""
Consultas conjuntivas positivas com uma variável de resposta.

Single-hop usa um átomo; cadeias ligam variáveis intermediárias; interseções
exigem a mesma atribuição às variáveis compartilhadas. Contagens, negação e
comparações não são executadas como provas neste fragmento.
O compilador LLM é falível. Os compiladores por decomposição e evidências usam
anotações privilegiadas e são heurísticos; não são consultas ouro garantidas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from wrag import prompts
from wrag.data import Question
from wrag.llm import LLM, GenParams
from wrag.llm.filters import LEDGER
from wrag.util import get_logger, normalize

log = get_logger("wrag.witness.query")

VAR_RE = re.compile(r"^\?[A-Za-z_]\w*$")

_SET_NOUNS = frozenset(
    "activities items things books films movies songs instruments hobbies interests "
    "places countries cities events plans goals projects jobs pets languages games "
    "purchases paintings artworks subjects topics foods restaurants schools universities "
    "awards people friends relatives".split()
)
_SET_VERBS = frozenset(
    "buy bought purchase purchased paint painted read visit visited attend attended "
    "join joined do done make made create created play played learn learned learnt "
    "mention mentioned discuss discussed".split()
)


def is_var(term: str) -> bool:
    return bool(VAR_RE.match((term or "").strip()))


def var_name(term: str) -> str:
    return term.strip().lstrip("?")


def looks_like_answer_set(question: str) -> bool:
    """Detecta pedidos enumerativos pela pergunta, sem usar rótulos ouro.

    O pós-processamento é deliberadamente conservador: cobre cabeças plurais
    inequívocas e construções como "what has X painted?". Casos ambíguos ficam
    a cargo do compilador, em vez de converter toda pergunta com *what* em lista.
    """
    words = re.findall(r"[a-z]+", (question or "").lower())
    if not words:
        return False
    if words[0] in {"what", "which"} and any(w in _SET_NOUNS for w in words[1:4]):
        return True
    if words[0] == "what" and len(words) > 3 and words[1] in {"has", "have"}:
        return any(w in _SET_VERBS for w in words[2:])
    return False


@dataclass
class Atom:
    relation: str
    subject: str
    object: str

    @property
    def subject_is_var(self) -> bool:
        return is_var(self.subject)

    @property
    def object_is_var(self) -> bool:
        return is_var(self.object)

    @property
    def n_constants(self) -> int:
        return int(not self.subject_is_var) + int(not self.object_is_var)

    def verbalize(self) -> str:
        """Forma textual para casar contra a verbalização de um fato. Variáveis
        somem: o que resta é a parte da qual o índice denso pode se aproximar."""
        parts = []
        if not self.subject_is_var:
            parts.append(self.subject)
        parts.append(self.relation)
        if not self.object_is_var:
            parts.append(self.object)
        return " ".join(parts)

    def variables(self) -> list[str]:
        return [var_name(t) for t in (self.subject, self.object) if is_var(t)]

    def to_dict(self) -> dict[str, str]:
        return {"relation": self.relation, "subject": self.subject, "object": self.object}


@dataclass
class ConjunctiveQuery:
    answer_var: str = "x"
    atoms: list[Atom] = field(default_factory=list)
    expected_type: str = "other"
    aggregation: str = "none"
    fallback: str = ""
    source: str = "llm"          # llm | decomposition | evidences
    filtered: bool = False       # a compilação foi bloqueada pelo filtro
    validation_error: str = ""
    repairs: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.atoms)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    def variables(self) -> list[str]:
        seen: list[str] = []
        for atom in self.atoms:
            for v in atom.variables():
                if v not in seen:
                    seen.append(v)
        return seen

    def constants(self) -> list[str]:
        out: list[str] = []
        for atom in self.atoms:
            for term in (atom.subject, atom.object):
                if not is_var(term) and term not in out:
                    out.append(term)
        return out

    def shape(self) -> str:
        """Classificação estrutural, usada para quebrar os resultados por tipo.

        A separação entre `chain` e `intersection` é a que mais importa
        analiticamente: são os dois regimes em que a proposta prevê comportamento
        diferente do PPR.
        """
        if self.n_atoms <= 1:
            return "single-hop"
        neighbors: dict[str, set[str]] = {}
        edges = set()
        for atom in self.atoms:
            left, right = atom.subject, atom.object
            neighbors.setdefault(left, set()).add(right)
            neighbors.setdefault(right, set()).add(left)
            edges.add(tuple(sorted((left, right))))
        visited, todo = set(), [next(iter(neighbors))]
        while todo:
            node = todo.pop()
            if node not in visited:
                visited.add(node)
                todo.extend(neighbors[node] - visited)
        if len(visited) != len(neighbors):
            return "disconnected"
        if len(edges) >= len(neighbors):
            return "cyclic"
        counts: dict[str, int] = {}
        for atom in self.atoms:
            for v in set(atom.variables()):
                counts[v] = counts.get(v, 0) + 1
        shared = [v for v, c in counts.items() if c > 1]
        if not shared:
            return "disconnected"
        if any(v == self.answer_var and counts[v] > 1 for v in shared):
            return "intersection"
        if any(len(adj) > 2 for adj in neighbors.values()):
            return "branching"
        return "chain"

    def to_dict(self) -> dict[str, Any]:
        return {"answer_var": self.answer_var, "atoms": [a.to_dict() for a in self.atoms],
                "expected_type": self.expected_type, "aggregation": self.aggregation,
                "shape": self.shape(), "source": self.source, "fallback": self.fallback,
                "validation_error": self.validation_error,
                "repairs": list(self.repairs),
                # O vocabulário vem do grafo extraído, não de anotação do dataset.
                "uses_annotations": self.source not in ("llm", "llm-vocabulario")}


# ---------------------------------------------------------------------------
# Compilação
# ---------------------------------------------------------------------------

def compile_with_llm(llm: LLM, question: Question, max_atoms: int = 4,
                     temperature: float = 0.0, dataset: str = "", method: str = "witnessrag",
                     vocabulary: str = "") -> ConjunctiveQuery:
    result = llm.chat(
        prompts.COMPILE_TEMPLATE.format(question=question.question, max_atoms=max_atoms,
                                        vocabulary=vocabulary),
        system=prompts.COMPILE_SYSTEM,
        params=GenParams(temperature=temperature, max_tokens=900, json_mode=True),
        stage="witness.compile",
    )
    if result.filtered:
        LEDGER.add("compile", dataset, method, question.qid, "compilação da consulta bloqueada")
        return ConjunctiveQuery(fallback=question.question, filtered=True)

    data = result.json()
    if not isinstance(data, dict):
        log.debug("compilação não devolveu JSON para %s", question.qid)
        return ConjunctiveQuery(fallback=question.question)

    atoms = _parse_atoms(data.get("atoms"), max_atoms)
    if isinstance(data.get("atoms"), list) and len(data["atoms"]) > max_atoms:
        return ConjunctiveQuery(fallback=question.question, validation_error="limite_de_atomos")
    if not isinstance(data.get("atoms"), list) or len(atoms) != len(data["atoms"]):
        return ConjunctiveQuery(fallback=question.question, validation_error="atomo_invalido")
    answer = var_name(str(data.get("answer_var") or "x"))
    query = ConjunctiveQuery(
        answer_var=answer,
        atoms=atoms,
        expected_type=str(data.get("expected_type") or "other"),
        aggregation=str(data.get("aggregation") or "none"),
        fallback=str(data.get("fallback") or question.question),
        source="llm-vocabulario" if vocabulary else "llm",
    )
    return _repair(query, question)


def _parse_atoms(raw: Any, max_atoms: int) -> list[Atom]:
    atoms: list[Atom] = []
    if not isinstance(raw, (list, tuple)):
        return atoms
    for item in raw or []:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            item = {"subject": item[0], "relation": item[1], "object": item[2]}
        if not isinstance(item, dict):
            continue
        relation = str(item.get("relation") or "").strip()
        subject = str(item.get("subject") or "").strip()
        obj = str(item.get("object") or "").strip()
        if not relation or not subject or not obj:
            continue
        atoms.append(Atom(relation=relation, subject=subject, object=obj))
        if len(atoms) >= max_atoms:
            break
    return atoms


def _repair(query: ConjunctiveQuery, question: Question) -> ConjunctiveQuery:
    """Conserta o erro de compilação mais comum: a variável de resposta não
    aparece em átomo nenhum. Sem esse reparo a junção terminaria sem candidato a
    resposta e a pergunta cairia no fallback denso por um motivo puramente
    sintático — e o experimento contaria como falha do método o que foi
    falha de formatação."""
    query.aggregation = query.aggregation.strip().lower()
    if query.aggregation in {"list", "all", "enumerate"}:
        query.aggregation = "set"
    if query.aggregation == "none" and looks_like_answer_set(question.question):
        query.aggregation = "set"
    if not query.atoms:
        return query
    answer = query.answer_var
    present = {v for atom in query.atoms for v in atom.variables()}
    if answer in present:
        return query
    if len(present) == 1:
        query.answer_var = next(iter(present))
        query.repairs.append("variavel_de_resposta_unica")
        return query
    query.validation_error = "variavel_de_resposta_ausente"
    query.atoms = []
    return query


def from_decomposition(question: Question) -> ConjunctiveQuery:
    """Cadeia a partir da decomposição anotada do MuSiQue.

    Cada sub-pergunta vira um átomo. As referências "#1", "#2" do MuSiQue apontam
    para a resposta de um passo anterior, e viram exatamente a variável daquele
    passo — que é como a cadeia se liga.
    """
    steps = question.decomposition
    if not steps:
        return ConjunctiveQuery(fallback=question.question, source="decomposition")

    atoms: list[Atom] = []
    for i, step in enumerate(steps):
        text = str(step.get("question") or "")
        subject, relation = _split_subquestion(text)
        references = re.findall(r"#(\d+)", text)
        if len(references) > 1 or any(int(r) > i or int(r) < 1 for r in references):
            return ConjunctiveQuery(source="decomposition-heuristic", fallback=question.question,
                                    validation_error="decomposicao_nao_suportada")
        ref = re.search(r"#(\d+)", text)
        if ref:
            subject = f"?s{int(ref.group(1)) - 1}"
        atoms.append(Atom(relation=relation or "related to",
                          subject=subject or f"?s{i - 1}" if i else subject,
                          object=f"?s{i}"))
    query = ConjunctiveQuery(answer_var=f"s{len(steps) - 1}", atoms=atoms,
                             fallback=question.question, source="decomposition-heuristic")
    return query


def _split_subquestion(text: str) -> tuple[str, str]:
    """Separa "Quem dirigiu X?" em (X, "dirigiu"). Heurística, e assumidamente
    imperfeita: este diagnóstico usa anotações privilegiadas e não estabelece
    um teto de compilação."""
    text = text.strip().rstrip("?")
    text = re.sub(r"^(who|what|when|where|which|whose)\b", "", text, flags=re.I).strip()
    match = re.search(r"#\d+|[A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*)*", text)
    entity = match.group(0) if match else ""
    relation = text.replace(entity, " ").strip() if entity else text
    relation = re.sub(r"^(who|what|when|where|which|whose|the)\b", "", relation, flags=re.I).strip()
    return entity, relation or text


def from_evidences(question: Question) -> ConjunctiveQuery:
    """Consulta a partir das triplas de evidência anotadas do 2Wiki.

    Esta é a condição "grafo e consulta corretos por construção" do lado da
    consulta: as triplas de evidência SÃO a testemunha de referência, então
    convertê-las em átomos dá a consulta que, sobre a interpretação de
    referência, tem exatamente aquela testemunha.
    """
    evidences = question.evidences
    if not evidences:
        return ConjunctiveQuery(fallback=question.question, source="evidences")

    answers = {normalize(a) for a in question.answers}
    atoms: list[Atom] = []
    var_of: dict[str, str] = {}
    counter = 0

    def term(value: str) -> str:
        nonlocal counter
        key = normalize(value)
        if key in answers:
            return "?x"
        if key in var_of:
            return var_of[key]
        return value

    # Um objeto que reaparece como sujeito de outra evidência é a entidade
    # intermediária da cadeia; vira variável para que a junção tenha que
    # descobri-la em vez de recebê-la pronta.
    subjects = {normalize(s) for s, _r, _o in evidences}
    for _s, _r, obj in evidences:
        key = normalize(obj)
        if key in subjects and key not in answers and key not in var_of:
            var_of[key] = f"?y{counter}"
            counter += 1

    for s, r, o in evidences:
        atoms.append(Atom(relation=r, subject=term(s), object=term(o)))

    query = ConjunctiveQuery(answer_var="x", atoms=atoms, fallback=question.question,
                             source="evidences")
    return _repair(query, question)


def compile_query(llm: LLM, question: Question, mode: str = "llm",
                  max_atoms: int = 4, temperature: float = 0.0,
                  dataset: str = "", method: str = "witnessrag",
                  vocabulary: str = "") -> ConjunctiveQuery:
    if mode in {"oracle", "annotated"}:
        if question.evidences:
            query = from_evidences(question)
        elif question.decomposition:
            query = from_decomposition(question)
        else:
            query = ConjunctiveQuery(fallback=question.question, source="oracle-indisponivel")
        return query  # nunca misturar silenciosamente anotação e compilação por LLM
    return compile_with_llm(llm, question, max_atoms=max_atoms, temperature=temperature,
                            dataset=dataset, method=method, vocabulary=vocabulary)
