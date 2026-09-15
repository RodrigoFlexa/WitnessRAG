"""
Todos os prompts do benchmark, em um arquivo só.

Duas convenções que o resto do código depende:

1. Toda entrada variável vem depois de `### INPUT`. O backend `stub` corta por
   esse marcador para achar a passagem ou a pergunta; um prompt que não o
   respeite roda no Azure e falha no teste offline.
2. Toda saída é JSON. `response_format` nem sempre está disponível no gateway,
   então o parser é tolerante (`parse_json_loose`) e o prompt insiste no formato.

Os prompts de NER e OpenIE seguem de perto o HippoRAG (extração em dois passos:
primeiro entidades, depois triplas condicionadas às entidades), porque o mesmo
extrator alimenta os quatro métodos com grafo. Trocar o extrator entre métodos
mudaria a variável errada.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Extração (compartilhada por GraphRAG, HippoRAG, HippoRAG 2 e WITNESS-RAG)
# ---------------------------------------------------------------------------

NER_SYSTEM = (
    "You are an information extraction system. You extract named entities from text. "
    "You always answer with a single JSON object and nothing else."
)

NER_TEMPLATE = """Extract every named entity from the passage below.

Include people, organizations, locations, works (films, books, albums), events,
dates and other proper nouns. Keep the surface form as it appears in the text.

Answer with JSON exactly in this shape:
{{"named_entities": ["entity one", "entity two"]}}

### INPUT
{text}"""


OPENIE_SYSTEM = (
    "You are an open information extraction system. You convert passages into "
    "subject-relation-object triples. You always answer with a single JSON object and nothing else."
)

OPENIE_TEMPLATE = """Convert the passage below into open knowledge-graph triples.

Rules:
- Use the named entities listed as subjects or objects whenever they apply, but you
  may also introduce other noun phrases (concepts, dates, values) when the passage
  states a relation about them.
- The relation must be a short phrase taken from or faithful to the passage.
- Every triple must be supported by the passage on its own. Do not infer.
- Keep dates and numbers as they are written.
- At most {max_triples} triples.

Answer with JSON exactly in this shape:
{{"triples": [["subject", "relation", "object"]]}}

Named entities found in this passage: {entities}

### INPUT
{text}"""


# Extração de diálogo. Num corpus conversacional a maior parte das asserções
# está em primeira pessoa ("I moved from Sweden", "my kids love dinosaurs"): sem
# resolver o falante, o sujeito do fato é um pronome e nenhuma junção fecha.
# A data da sessão entra como escopo temporal do fato, não como entidade solta.
OPENIE_DIALOGUE_SYSTEM = (
    "You are an open information extraction system for conversation transcripts. "
    "You convert dialogue turns into subject-relation-object triples with the speaker "
    "resolved. You always answer with a single JSON object and nothing else."
)

OPENIE_DIALOGUE_TEMPLATE = """Convert the conversation block below into open knowledge-graph triples.

Each line is "[dialog id] Speaker: text". The block starts with the session date.

Rules:
- Resolve the speaker: "I", "me", "my", "mine" refer to the speaker of THAT line.
  Write the speaker's name as the subject, never a pronoun.
- Resolve "you"/"your" to the other participant of the conversation.
- Resolve "he", "she", "they", "it" and possessives to the entity named earlier in
  this block. If the referent is not in this block, keep the noun phrase as written
  and do not guess a name.
- Name relatives and belongings through their owner: "my son Theo" is
  ("Theo", "child of", "<speaker>"), not ("my son", ...).
- Scope unnamed relatives and possessions to their owner so generic entities do
  not collide across speakers: "my grandmother" becomes "<speaker>'s grandmother"
  and "my necklace symbolizes strength" has subject "<speaker>'s necklace".
- The relation is SHORT: one to four words, a predicate and nothing else. Write it
  in a reusable canonical form: base/present-tense verb plus any REQUIRED
  preposition ("read", "paint", "work at", "child of"). Reuse exactly the same
  relation for paraphrases in this block. It never contains the object, a whole
  sentence, time, an adverb, or the name of the person being addressed.
- For reported speech, the fact is about the CONTENT, not about the listener.
  "Mel: running is a great way to destress" is ("Mel", "destresses by", "running"),
  never ("Mel", "said running is a great way to destress", "<listener>").
- The object is a thing, person, place, date or value taken from the text. Never
  "true", "false" or "yes": if a statement has no object, write the relation so
  that it has one, or leave the statement out.
- Every triple must be supported by the block on its own. Do not infer.
- Add the time as a fourth element when the triple describes something that
  happened, changed or was stated at a moment: use the session date for what the
  speaker reports in that session, or the explicit date/period the text gives
  ("last year", "in October"). Use "" when the fact is not tied to a time.
- Keep dates, numbers and titles as they are written.
- At most {max_triples} triples.

Answer with JSON exactly in this shape:
{{"triples": [["subject", "relation", "object", "time"]]}}

Example. For the block

  Session date: 8 May, 2023
  [D1:1] Mel: I finally finished Charlotte's Web with my son Theo last week!
  [D1:2] Caroline: Nice! I painted a sunset yesterday.

the triples are
{{"triples": [["Mel", "read", "Charlotte's Web", "last week"],
             ["Theo", "child of", "Mel", ""],
             ["Caroline", "paint", "a sunset", "7 May, 2023"]]}}

Named entities found in this block: {entities}

### INPUT
{text}"""


# Extração dirigida, usada pela aquisição adaptativa do WITNESS-RAG. A diferença
# em relação ao OpenIE geral é o alvo: aqui já sabemos qual buraco da testemunha
# queremos fechar, então pedimos exatamente aquela relação.
TARGETED_IE_SYSTEM = (
    "You are a targeted extraction system. Given a passage and one specific relation "
    "of interest, you extract only the triples that instantiate that relation. "
    "You always answer with a single JSON object and nothing else."
)

TARGETED_IE_TEMPLATE = """Extract only triples that express the relation of interest.

Relation of interest: "{relation}"
{anchor_line}
If the passage does not state that relation, answer with an empty list. Do not infer
and do not substitute a different relation.

Answer with JSON exactly in this shape:
{{"triples": [["subject", "relation", "object"]]}}

### INPUT
{text}"""


# ---------------------------------------------------------------------------
# HippoRAG: NER da consulta / HippoRAG 2: recognition memory
# ---------------------------------------------------------------------------

QUERY_NER_SYSTEM = NER_SYSTEM

QUERY_NER_TEMPLATE = """Extract the named entities mentioned in the question below.
These are the entities a reader would look up to start answering it.

Answer with JSON exactly in this shape:
{{"named_entities": ["entity one"]}}

### INPUT
{question}"""


TRIPLE_FILTER_SYSTEM = (
    "You judge whether knowledge-graph facts are relevant to a question. "
    "You always answer with a single JSON object and nothing else."
)

TRIPLE_FILTER_TEMPLATE = """Below is a question and a numbered list of candidate facts.

Keep only the facts that a person would actually use while answering the question,
including facts needed for an intermediate step. Drop facts about unrelated entities.
Keep at most {max_kept} facts, ordered by usefulness.

Answer with JSON exactly in this shape:
{{"fact": [["subject", "relation", "object"]]}}

### INPUT
PERGUNTA: {question}

{triples}"""


# ---------------------------------------------------------------------------
# GraphRAG: relatório de comunidade
# ---------------------------------------------------------------------------

COMMUNITY_SYSTEM = (
    "You are an analyst writing a short report about a cluster of related entities. "
    "You always answer with a single JSON object and nothing else."
)

COMMUNITY_TEMPLATE = """Write a short report about the community of entities below.

The report is used to answer questions later, so state concrete facts and names,
not generalities. Two to four sentences.

Answer with JSON exactly in this shape:
{{"title": "short title", "summary": "the report"}}

### INPUT
Entities: {entities}

Relationships:
{relationships}"""


# ---------------------------------------------------------------------------
# WITNESS-RAG: compilação da pergunta em consulta conjuntiva
# ---------------------------------------------------------------------------

COMPILE_SYSTEM = (
    "You translate natural-language questions into conjunctive graph queries. "
    "You always answer with a single JSON object and nothing else."
)

COMPILE_TEMPLATE = """Translate the question into a conjunctive query over a knowledge graph.

A conjunctive query is a set of atoms that must ALL hold at the same time, sharing
variables. Variables start with "?". The answer variable is "?x". Intermediate
entities you do not know are variables too ("?y", "?z").

Guidance:
- The literal variable "?x" MUST occur in at least one atom. It is the value the
  question asks for. Use "?y" and "?z" only for intermediate entities.
- A single-hop question becomes one atom: relation(constant, ?x).
- A chain question becomes atoms linked by an intermediate variable:
  relation1(constant, ?y) AND relation2(?y, ?x).
- An intersection question repeats the SAME variable in two atoms:
  relation1(?x, constantA) AND relation2(?x, constantB).
- Write relations as short natural-language phrases ("director", "date of death",
  "employer", "located in"). Do not invent a schema.
- Constants must be entity names copied from the question.
- Prefer the smallest query that expresses the question. Do not create separate
  atoms for adjectives, time phrases, reasons, feelings or event context. A
  possessive common noun is a scoped constant ("Caroline's necklace"), not an
  unknown intermediate entity, unless the question actually asks who owns it.
- Qualifiers such as "recently", "after the accident" and "during the workshop"
  stay inside the main relation phrase; never encode them as artificial atoms
  like recent(?x, true), after(?x, accident) or during(?x, workshop). Never use
  "true", "false" or "yes" as an argument.
- Every atom must be connected through a shared variable or constant. Never emit
  an unrelated atom merely because its words occur in the question.
- At most {max_atoms} atoms. If the question needs comparison, counting or
  negation, still emit the atoms that fetch the facts to be compared, and set
  "aggregation" to describe what is done with them.
- Use aggregation "set" when the question asks for all matching people or items.
  A witness then proves one member; the final answer is the union of all proven
  members. Use "none" when exactly one value is requested.
{vocabulary}
Answer with JSON exactly in this shape:
{{"answer_var": "x",
  "atoms": [{{"relation": "...", "subject": "...", "object": "?x"}}],
  "expected_type": "person|place|date|organization|work|number|other",
  "aggregation": "none|set|max|min|compare|count",
  "fallback": "a keyword query to use if the graph search fails"}}

Examples:
Question: "When did Lothair II's mother die?"
{{"answer_var": "x", "atoms": [{{"relation": "mother", "subject": "Lothair II", "object": "?y"}}, {{"relation": "date of death", "subject": "?y", "object": "?x"}}], "expected_type": "date", "aggregation": "none", "fallback": "Lothair II mother date of death"}}

Question: "Which Stanford professor works on Alzheimer's?"
{{"answer_var": "x", "atoms": [{{"relation": "professor at", "subject": "?x", "object": "Stanford University"}}, {{"relation": "researches", "subject": "?x", "object": "Alzheimer's"}}], "expected_type": "person", "aggregation": "none", "fallback": "Stanford professor Alzheimer's research"}}

Question: "What activities does Melanie partake in?"
{{"answer_var": "x", "atoms": [{{"relation": "participate in", "subject": "Melanie", "object": "?x"}}], "expected_type": "other", "aggregation": "set", "fallback": "Melanie activities participate"}}

Question: "What does Caroline's necklace symbolize?"
{{"answer_var": "x", "atoms": [{{"relation": "symbolize", "subject": "Caroline's necklace", "object": "?x"}}], "expected_type": "other", "aggregation": "none", "fallback": "Caroline necklace symbolize"}}

### INPUT
{question}"""


COMPILE_PLANS_TEMPLATE = """Generate up to {max_plans} distinct candidate query plans for the question.

Each plan is a positive conjunctive query. Variables start with "?" and the
literal answer variable "?x" must occur in at least one atom. Each atom has a
short relation, subject and object. Use at most {max_atoms} atoms per plan.

Order plans from most faithful to least preferred:
1. A minimal direct plan when the question can be expressed by one fact. Keep
   qualifiers such as "recently", "after the accident" or "during the workshop"
   inside that relation phrase.
2. A chain plan only when an unnamed intermediate entity must genuinely be found.
3. An intersection plan only when the same answer must independently satisfy
   two relations.

Plans must be meaningfully different. Never pad a plan with artificial atoms such
as recent(?x, true), after(?x, event), type_of(?x, requested_type), or an atom
unconnected to the answer path. Never use true, false or yes as an argument.
Preserve relation direction. For "What has Alice bought?", use buy(Alice, ?x).
Use aggregation "set" for all matching members, "count" for a count, and "none"
for one value. max/min/compare are allowed descriptions but are not executable.
{vocabulary}
Return JSON exactly in this shape:
{{"plans": [
  {{"answer_var": "x",
    "atoms": [{{"relation": "...", "subject": "...", "object": "?x"}}],
    "expected_type": "person|place|date|organization|work|number|other",
    "aggregation": "none|set|max|min|compare|count",
    "fallback": "keyword query"}}
]}}

### INPUT
{question}"""


# ---------------------------------------------------------------------------
# Leitura final (idêntica para os cinco sistemas)
# ---------------------------------------------------------------------------

QA_SYSTEM = (
    "You answer questions using only the passages provided. You always answer with a "
    "single JSON object and nothing else."
)

QA_TEMPLATE = """Answer the question using only the passages below.

Give the shortest answer that is complete: a name, a date, a number or a short noun
phrase. Do not write a sentence. If the passages do not contain the answer, answer
with "insufficient information".

Answer with JSON exactly in this shape:
{{"answer": "..."}}

### INPUT
{passages}

PERGUNTA: {question}"""


# Variante ciente de conjunto. A diferença é só a regra de completude: quando a
# pergunta pede um conjunto, a resposta curta de um item está ERRADA por omissão,
# e o leitor precisa poder dizer isso. O resto do prompt é idêntico, e a variante
# vale para todos os métodos da rodada — a comparação entre métodos não muda.
QA_SET_TEMPLATE = """Answer the question using only the passages below.

Give the shortest answer that is complete: a name, a date, a number or a short noun
phrase. Do not write a sentence.

Some questions ask for a SET: everything a person did, made, visited, read, owns,
was given, or took part in, possibly mentioned on different days. For those, list
EVERY item the passages support, separated by commas, and nothing else. Answering
with one item is complete only when the passages support exactly one. Never add an
item the passages do not state, and never repeat the same item twice.

If the passages do not contain the answer, answer with "insufficient information".

Answer with JSON exactly in this shape:
{{"answer": "..."}}

### INPUT
{passages}

PERGUNTA: {question}"""


def format_vocabulary(relations: Sequence[str], entities: Sequence[str]) -> str:
    """Bloco de vocabulário para a compilação. Vazio quando a ablação está desligada.

    São sugestões extraídas do grafo, não um esquema fechado: o compilador pode
    emitir outra relação, e o texto diz isso. Fechar o vocabulário transformaria
    um erro de extração em impossibilidade de compilar.
    """
    if not relations and not entities:
        return ""
    parts = ["\nThe graph was built from the corpus and already contains these relations."
             " Prefer one of them, with its exact spelling, whenever it expresses what the"
             " question asks. Invent a new relation only when none of them fits:"]
    if relations:
        parts.append("  " + "; ".join(relations))
    if entities:
        parts.append("Entity names that exist in the graph, useful as constants:")
        parts.append("  " + "; ".join(entities))
    return "\n".join(parts) + "\n"


def format_passages(passages: Sequence[tuple[str, str]], max_chars: int | None = None) -> str:
    """Formata (título, texto) para o prompt de leitura. Mesma formatação para
    todos os métodos: a diferença entre eles tem que ser o que foi recuperado,
    não como foi apresentado."""
    blocks = []
    for i, (title, text) in enumerate(passages, 1):
        body = text if max_chars is None or len(text) <= max_chars else text[: max_chars - 3] + "..."
        blocks.append(f"[{i}] {title}\n{body}")
    return "\n\n".join(blocks) if blocks else "(nenhuma passagem recuperada)"


def format_triples(triples: Sequence[Sequence[str]]) -> str:
    return "\n".join(
        f"{i}. ({t[0]} | {t[1]} | {t[2]})" for i, t in enumerate(triples, 1)
    )


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
