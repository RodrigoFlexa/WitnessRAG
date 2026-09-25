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
    # Optional time slot (design v3): a variable such as "?t" makes the date of
    # the matched fact a value of the query, so "when" questions have a proof
    # whose answer is a date. Empty for the legacy two-argument atoms.
    time: str = ""

    @property
    def subject_is_var(self) -> bool:
        return is_var(self.subject)

    @property
    def object_is_var(self) -> bool:
        return is_var(self.object)

    @property
    def time_is_var(self) -> bool:
        return is_var(self.time)

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
        names = [var_name(t) for t in (self.subject, self.object) if is_var(t)]
        if self.time_is_var and var_name(self.time) not in names:
            names.append(var_name(self.time))
        return names

    def entity_variables(self) -> list[str]:
        return [var_name(t) for t in (self.subject, self.object) if is_var(t)]

    def to_dict(self) -> dict[str, str]:
        out = {"relation": self.relation, "subject": self.subject, "object": self.object}
        if self.time:
            out["time"] = self.time
        return out


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
    # Requirements checked against literal source passages after graph joining.
    # They are not graph predicates and cannot make an incomplete join complete.
    conditions: list[str] = field(default_factory=list)
    # Unary type atoms (design v4): variable name -> type phrase copied from the
    # question ("martial art", "musical instrument"). Checked by ranking and by
    # the verifier, never by a similarity threshold (see search.type_affinity).
    types: dict[str, str] = field(default_factory=dict)

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
            for v in set(atom.entity_variables()):
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
        result = {"answer_var": self.answer_var, "atoms": [a.to_dict() for a in self.atoms],
                "expected_type": self.expected_type, "aggregation": self.aggregation,
                "shape": self.shape(), "source": self.source, "fallback": self.fallback,
                "validation_error": self.validation_error,
                "repairs": list(self.repairs),
                # O vocabulário vem do grafo extraído, não de anotação do dataset.
                "uses_annotations": not self.source.startswith("llm")}
        if self.conditions:
            result["source_conditions"] = list(self.conditions)
        if self.types:
            result["types"] = dict(self.types)
        return result


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

    return _query_from_data(data, question, max_atoms,
                            "llm-vocabulario" if vocabulary else "llm")


def compile_plans_with_llm(llm: LLM, question: Question, max_atoms: int = 4,
                           max_plans: int = 3, temperature: float = 0.0,
                           dataset: str = "", method: str = "witnessrag",
                           vocabulary: str = "", feedback: str = "",
                           source_conditions: bool = False) -> list[ConjunctiveQuery]:
    """Compila interpretações alternativas em uma chamada de LLM.

    Os planos são hipóteses ordenadas, não programas confiáveis. O retriever os
    valida contra o índice e mantém no diagnóstico tudo que foi tentado.
    """
    result = llm.chat(
        (prompts.COMPILE_PLANS_REPAIR_TEMPLATE if source_conditions else
         prompts.COMPILE_PLANS_TEMPLATE).format(
            question=question.question, max_atoms=max_atoms,
            max_plans=max(1, max_plans), vocabulary=vocabulary,
            planning_feedback=feedback),
        system=prompts.COMPILE_SYSTEM,
        params=GenParams(temperature=temperature, max_tokens=2200, json_mode=True),
        stage="witness.replan" if feedback else "witness.compile",
    )
    if result.filtered:
        LEDGER.add("compile", dataset, method, question.qid,
                   "compilação dos planos bloqueada")
        return [ConjunctiveQuery(fallback=question.question, filtered=True,
                                 source="llm-plan")]
    data = result.json()
    raw_plans = data.get("plans") if isinstance(data, dict) else None
    if not isinstance(raw_plans, list):
        return [ConjunctiveQuery(fallback=question.question,
                                 validation_error="planos_invalidos", source="llm-plan")]
    if feedback:
        source = "llm-replan-vocabulario" if vocabulary else "llm-replan"
    else:
        source = "llm-plan-vocabulario" if vocabulary else "llm-plan"
    plans, seen = [], set()
    for raw in raw_plans[:max(1, max_plans)]:
        if not isinstance(raw, dict):
            continue
        plan = _query_from_data(raw, question, max_atoms, source)
        signature = tuple((normalize(a.subject), normalize(a.relation), normalize(a.object))
                          for a in plan.atoms) + ((plan.answer_var, plan.aggregation),
                                                  tuple(normalize(c) for c in plan.conditions))
        if signature not in seen:
            seen.add(signature)
            plans.append(plan)
    return plans or [ConjunctiveQuery(fallback=question.question,
                                      validation_error="planos_vazios", source=source)]


def _query_from_data(data: dict[str, Any], question: Question, max_atoms: int,
                     source: str) -> ConjunctiveQuery:
    atoms = _parse_atoms(data.get("atoms"), max_atoms)
    if isinstance(data.get("atoms"), list) and len(data["atoms"]) > max_atoms:
        return ConjunctiveQuery(fallback=question.question, validation_error="limite_de_atomos",
                                source=source)
    if not isinstance(data.get("atoms"), list) or len(atoms) != len(data["atoms"]):
        return ConjunctiveQuery(fallback=question.question, validation_error="atomo_invalido",
                                source=source)
    answer = var_name(str(data.get("answer_var") or "x"))
    query = ConjunctiveQuery(
        answer_var=answer,
        atoms=atoms,
        expected_type=str(data.get("expected_type") or "other"),
        aggregation=str(data.get("aggregation") or "none"),
        fallback=str(data.get("fallback") or question.question),
        source=source,
    )
    raw_conditions = data.get("source_conditions", [])
    if isinstance(raw_conditions, list) and len(raw_conditions) <= 6 and all(
            isinstance(item, str) and 0 < len(item.strip()) <= 160
            for item in raw_conditions):
        query.conditions = list(dict.fromkeys(item.strip() for item in raw_conditions))
    elif raw_conditions:
        query.validation_error = "condicoes_textuais_invalidas"
        query.atoms = []
        return query
    # JSON already has separate subject/object fields. A model occasionally
    # repeats Prolog-like arguments inside ``relation`` (for example,
    # ``create(Melanie, ?x)``). Treating that whole string as a predicate makes
    # grounding meaningless. Parenthetical natural qualifiers such as
    # "go to beach (2023)" remain valid because whitespace precedes "(".
    if any("?" in atom.relation or re.search(r"\w\(", atom.relation)
           for atom in query.atoms):
        query.validation_error = "relacao_contem_argumentos"
        query.atoms = []
        return query
    if any(not is_var(term) and normalize(term) in {"true", "false", "yes"}
           for atom in query.atoms for term in (atom.subject, atom.object)):
        query.validation_error = "constante_booleana_artificial"
        query.atoms = []
        return query
    query = _repair(query, question)
    _attach_types(query, data)
    return query


_GENERIC_TYPES = frozenset("thing things item items something stuff entity entities "
                           "other answer value one ones".split())


def _attach_types(query: ConjunctiveQuery, data: dict[str, Any]) -> None:
    """Read the optional unary type atoms. Invalid entries are dropped with a
    repair note; a type never invalidates the plan, because it only narrows it."""
    raw = data.get("types")
    if raw is None and isinstance(data.get("answer_type"), str):
        raw = {query.answer_var: data["answer_type"]}
    if not raw or not query.atoms:
        return
    if not isinstance(raw, dict):
        query.repairs.append("tipos_invalidos")
        return
    variables = {v for atom in query.atoms for v in atom.entity_variables()}
    for name, phrase in raw.items():
        var = var_name(str(name))
        text = re.sub(r"\s+", " ", str(phrase or "")).strip().strip(".")
        if not text or var not in variables or len(text) > 60:
            if text:
                query.repairs.append(f"tipo_descartado:{var}")
            continue
        if normalize(text) in _GENERIC_TYPES:
            continue
        query.types[var] = text


_AUXILIARIES = frozenset("do does did has have had is are was were will would can could "
                        "should might may".split())
_TYPE_HEAD = re.compile(r"^\s*(?:what|which)\s+(?:(?:kinds?|types?|sorts?)\s+of\s+)?"
                        r"((?:[a-z][a-z'\-]*\s+){0,3}?[a-z][a-z'\-]*)\s+"
                        r"(?:" + "|".join(sorted(_AUXILIARIES)) + r")\b", re.I)


def _singular(word: str) -> str:
    low = word.lower()
    if low.endswith("ies") and len(low) > 4:
        return word[:-3] + "y"
    if low.endswith(("sses", "shes", "ches", "xes")):
        return word[:-2]
    if low.endswith("s") and not low.endswith(("ss", "us", "is")) and len(low) > 3:
        return word[:-1]
    return word


def question_type_phrase(question: str) -> str:
    """The kind of thing a wh-question asks for, read from its words only:
    "What martial arts has John done?" -> "martial art"; "What kind of books
    does X have?" -> "book"; "What did X buy?" -> "" (no kind named). Generic
    heads ("things", "items") give "". Never reads a benchmark label."""
    match = _TYPE_HEAD.match(question or "")
    if not match:
        return ""
    words = match.group(1).split()
    if not words:
        return ""
    if any(w.lower() in _AUXILIARIES for w in words):
        return ""
    words[-1] = _singular(words[-1])
    phrase = " ".join(words)
    if normalize(phrase) in _GENERIC_TYPES or normalize(words[-1]) in _GENERIC_TYPES:
        return ""
    return phrase


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
        # Only a variable is accepted in the time slot; dates belong to the
        # plan's period, never to an atom constant.
        time = str(item.get("time") or "").strip()
        atoms.append(Atom(relation=relation, subject=subject, object=obj,
                          time=time if is_var(time) else ""))
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

    # Erro de direção observado em perguntas inglesas do tipo
    # "What has Alice bought?": alguns compiladores emitem buy(?x, Alice),
    # embora a gramática torne Alice o agente e ?x o objeto. O reparo só vale
    # para um único átomo e para o padrão explícito has/have; não tenta adivinhar
    # a direção de relações arbitrárias.
    words = re.findall(r"[a-z]+", question.question.lower())
    if (len(query.atoms) == 1 and len(words) > 3 and words[0] == "what"
            and words[1] in {"has", "have"}):
        atom = query.atoms[0]
        if (atom.subject_is_var and var_name(atom.subject) == query.answer_var
                and not atom.object_is_var):
            atom.subject, atom.object = atom.object, atom.subject
            query.repairs.append("direcao_what_has")

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


def compile_query_plans(llm: LLM, question: Question, mode: str = "llm",
                        max_atoms: int = 4, max_plans: int = 3,
                        temperature: float = 0.0, dataset: str = "",
                        method: str = "witnessrag", vocabulary: str = "",
                        feedback: str = "", source_conditions: bool = False) -> list[ConjunctiveQuery]:
    if mode != "llm":
        return [compile_query(llm, question, mode=mode, max_atoms=max_atoms,
                              temperature=temperature, dataset=dataset,
                              method=method, vocabulary=vocabulary)]
    return compile_plans_with_llm(llm, question, max_atoms=max_atoms,
                                  max_plans=max_plans, temperature=temperature,
                                  dataset=dataset, method=method, vocabulary=vocabulary,
                                  feedback=feedback,
                                  source_conditions=source_conditions)
