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
  "Ravi: running is a great way to destress" is ("Ravi", "destresses by", "running"),
  never ("Ravi", "said running is a great way to destress", "<listener>").
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
  [D1:1] Ravi: I finally finished The Secret Garden with my son Niko last week!
  [D1:2] Lea: Nice! I painted a sunset yesterday.

the triples are
{{"triples": [["Ravi", "read", "The Secret Garden", "last week"],
             ["Niko", "child of", "Ravi", ""],
             ["Lea", "paint", "a sunset", "7 May, 2023"]]}}

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
  "employer", "located in"). The relation field contains only that predicate
  phrase. Put arguments only in subject/object: use relation="play",
  subject="Nira", object="?x"; never relation="play(Nira, ?x)".
  Do not invent a schema.
- Constants must be entity names copied from the question.
- Prefer the smallest query that expresses the question. Do not create separate
  atoms for adjectives, time phrases, reasons, feelings or event context. A
  possessive common noun is a scoped constant ("Asha's necklace"), not an
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

Answer with JSON exactly in this shape:
{{"answer_var": "x",
  "atoms": [{{"relation": "...", "subject": "...", "object": "?x"}}],
  "expected_type": "person|place|date|organization|work|number|other",
  "aggregation": "none|set|max|min|compare|count",
  "fallback": "a keyword query to use if the graph search fails"}}

Synthetic examples follow. Learn their structure; do not copy their names or
predicates into the answer.

Question: "Which instrument does Nira play?"
{{"answer_var":"x","atoms":[{{"relation":"play","subject":"Nira","object":"?x"}}],"expected_type":"other","aggregation":"none","fallback":"Nira instrument play"}}

Question: "In which city is the laboratory led by Omar located?"
{{"answer_var":"x","atoms":[{{"relation":"lead","subject":"Omar","object":"?y"}},{{"relation":"located in","subject":"?y","object":"?x"}}],"expected_type":"place","aggregation":"none","fallback":"Omar laboratory city"}}

Question: "Which researcher works at Northstar Institute and studies coral bleaching?"
{{"answer_var":"x","atoms":[{{"relation":"work at","subject":"?x","object":"Northstar Institute"}},{{"relation":"study","subject":"?x","object":"coral bleaching"}}],"expected_type":"person","aggregation":"set","fallback":"Northstar researcher coral bleaching"}}

Question: "What exhibition did Jun visit after the conference?"
{{"answer_var":"x","atoms":[{{"relation":"visit after conference","subject":"Jun","object":"?x"}}],"expected_type":"other","aggregation":"none","fallback":"Jun exhibition after conference"}}

Question: "Which dishes did Mateo cook for the festival?"
{{"answer_var":"x","atoms":[{{"relation":"cook for festival","subject":"Mateo","object":"?x"}}],"expected_type":"other","aggregation":"set","fallback":"Mateo festival dishes"}}

Question: "How many apprentices joined the observatory?"
{{"answer_var":"x","atoms":[{{"relation":"join","subject":"?x","object":"the observatory"}}],"expected_type":"number","aggregation":"count","fallback":"observatory apprentices joined"}}

Return one JSON object only. Do not include explanations, markdown or examples.
{vocabulary}

### INPUT
{question}"""


COMPILE_PLANS_TEMPLATE = """Generate up to {max_plans} distinct candidate query plans for the question.

Each plan is a positive conjunctive query. Variables start with "?" and the
literal answer variable "?x" must occur in at least one atom. Each atom has a
short relation, subject and object. The `relation` field contains ONLY the
predicate phrase: write `{{"relation": "create", "subject": "Asha",
"object": "?x"}}`, never `{{"relation": "create(Asha, ?x)", ...}}`.
Use at most {max_atoms} atoms per plan.

Order plans from most faithful to least preferred:
1. A minimal direct plan only when every requirement can be expressed by one fact. Keep
   qualifiers such as "recently", "after the accident" or "during the workshop"
   inside that relation phrase. Never remove a person, time, place, reason or
   other condition merely because a broader atom is easier to match.
2. A chain plan only when an unnamed intermediate entity must genuinely be found.
3. An intersection plan only when the same answer must independently satisfy
   two relations.

Plans must be meaningfully different. Never pad a plan with artificial atoms such
as recent(?x, true), after(?x, event), type_of(?x, requested_type), or an atom
unconnected to the answer path. Never use true, false or yes as an argument.
Preserve relation direction. For "What has Alice bought?", use buy(Alice, ?x).
Use aggregation "set" for all matching members, "count" for a count, and "none"
for one value. max/min/compare are allowed descriptions but are not executable.

Return JSON exactly in this shape:
{{"plans": [
  {{"answer_var": "x",
    "atoms": [{{"relation": "...", "subject": "...", "object": "?x"}}],
    "expected_type": "person|place|date|organization|work|number|other",
    "aggregation": "none|set|max|min|compare|count",
    "fallback": "keyword query"}}
]}}

Synthetic examples follow. Learn their STRUCTURE; do not copy their entity names
or predicates into the answer. Return fewer than {max_plans} plans when additional
plans would merely be padding. When the question has a genuine intermediate entity,
multiple independent conditions, or an ambiguous predicate, provide at least two
faithful alternatives when possible.

Example 1 — a direct fact needs one plan, not an invented chain.
Question: "Which instrument does Nira play?"
{{"plans":[{{"answer_var":"x","atoms":[{{"relation":"play","subject":"Nira","object":"?x"}}],"expected_type":"other","aggregation":"none","fallback":"Nira instrument play"}}]}}

Example 2 — a real chain uses an unknown intermediate entity.
Question: "In which city is the laboratory led by Omar located?"
{{"plans":[{{"answer_var":"x","atoms":[{{"relation":"lead","subject":"Omar","object":"?y"}},{{"relation":"located in","subject":"?y","object":"?x"}}],"expected_type":"place","aggregation":"none","fallback":"Omar laboratory city"}}]}}

Example 3 — an intersection repeats the answer variable because both conditions
must describe the same answer.
Question: "Which researcher works at Northstar Institute and studies coral bleaching?"
{{"plans":[{{"answer_var":"x","atoms":[{{"relation":"work at","subject":"?x","object":"Northstar Institute"}},{{"relation":"study","subject":"?x","object":"coral bleaching"}}],"expected_type":"person","aggregation":"set","fallback":"Northstar researcher coral bleaching"}}]}}

Example 4 — event context stays in the main predicate. It does not become
`after(?x, conference)` or a boolean atom, and it must not be dropped.
Question: "What exhibition did Jun visit after the conference?"
{{"plans":[{{"answer_var":"x","atoms":[{{"relation":"visit after conference","subject":"Jun","object":"?x"}}],"expected_type":"other","aggregation":"none","fallback":"Jun exhibition after conference"}}]}}

Example 5 — plural enumeration uses set; one witness may prove one member.
Question: "Which dishes did Mateo cook for the festival?"
{{"plans":[{{"answer_var":"x","atoms":[{{"relation":"cook for festival","subject":"Mateo","object":"?x"}}],"expected_type":"other","aggregation":"set","fallback":"Mateo festival dishes"}}]}}

Example 6 — counting retrieves the members to count; the number is not an
invented graph constant.
Question: "How many apprentices joined the observatory?"
{{"plans":[{{"answer_var":"x","atoms":[{{"relation":"join","subject":"?x","object":"the observatory"}}],"expected_type":"number","aggregation":"count","fallback":"observatory apprentices joined"}}]}}

Remember: every response is one JSON object with only the `plans` field and the
plan fields shown above. Do not include explanations, markdown or the examples.
{planning_feedback}
{vocabulary}

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


QA_COUNT_TEMPLATE = """Answer the question using only the original dialogue excerpts below.
Identify the exact person, event, time and other conditions in the question.
Count distinct matching events or items, not mentions, speakers or passages.
Repeated discussion of one event counts once. An explicit total can be used only
when it matches all conditions. A list of some events does not establish that
there were no others. If the excerpts do not establish the answer, return
"insufficient information". Return only a short JSON answer, without reasoning:
{{"answer": "..."}}

### INPUT
{passages}

PERGUNTA: {question}"""


QA_PROOF_TEMPLATE = """Answer the question using only the source passages below.

The evidence map lists possible graph joins extracted from those passages. It
is a navigation aid, not an answer key or a certificate. For each candidate,
check the cited passage text, the direction of every relation, and EVERY
condition in the question. Ignore any candidate that fails a condition or is
not explicitly supported by its cited passage. The map can omit correct answers:
also inspect the passages themselves. Never infer an exact count from a partial
list of graph candidates.

Give the shortest complete answer: a name, date, number or short noun phrase.
For questions asking for a set, list every distinct supported item, separated
by commas. If the passages do not contain the answer, say "insufficient
information". Return JSON exactly: {{"answer": "..."}}

EVIDENCE MAP ({proof_status}; conditions to verify: {pending}):
{proof_hints}

SOURCE PASSAGES:
{passages}

QUESTION: {question}"""


QA_OPERATOR_TEMPLATE = """Answer the question using only the passages below.

First identify every condition in the question, including the person, event,
source, date and any words such as after, before, from, or during. Use only
statements that satisfy all those conditions.

For "how many", prefer an explicit count stated in a passage when it matches
all conditions. Otherwise count DISTINCT people, objects or events; repeated
mentions of the same event are one event. Do not count the number of retrieved
passages, graph facts or candidate answers. If evidence is incomplete, answer
"insufficient information" rather than treating a partial list as complete.

For dates, resolve relative expressions such as "yesterday" using the date of
the same session, and respect the order of events. For questions asking for a
set, include every distinct item supported by the passages, separated by commas.

Return the shortest complete answer, with no explanation, as JSON:
{{"answer": "..."}}

### INPUT
{passages}

PERGUNTA: {question}"""


# Versioned plan/controller interface. The graph executes only short directed
# predicates; additional requirements must survive as explicit source checks.
COMPILE_PLANS_REPAIR_TEMPLATE = COMPILE_PLANS_TEMPLATE.replace(
    'Keep\n   qualifiers such as "recently", "after the accident" or "during the workshop"\n   inside that relation phrase. Never remove a person, time, place, reason or\n   other condition merely because a broader atom is easier to match.',
    'Keep the graph predicate short. Put time, place, reason, type and other\n   qualifiers that cannot be executed by one graph atom in source_conditions.\n   Never remove a requirement from the plan.')
COMPILE_PLANS_REPAIR_TEMPLATE = COMPILE_PLANS_REPAIR_TEMPLATE.replace(
    'only the `plans` field and the\nplan fields shown above',
    'the `plans` field and plan fields shown above, plus source_conditions')
COMPILE_PLANS_REPAIR_TEMPLATE = COMPILE_PLANS_REPAIR_TEMPLATE.replace(
    '"expected_type":"number","aggregation":"count"',
    '"expected_type":"person","aggregation":"count"')
_REPAIR_EXTENSION = """

For this experiment, each plan may also include `source_conditions`: a list of
requirements from the QUESTION that must be verified with verbatim quotes in
source passages after the graph join. Use a short graph relation that exists in
the suggested vocabulary when an entire time/place/type/recipient qualifier
would make the graph predicate unmatchable. Keep every removed qualifier in
`source_conditions`; never silently drop a requirement. Example structure:
{{"plans":[{{"answer_var":"x","atoms":[{{"relation":"goals",
"subject":"John","object":"?x"}}],"source_conditions":
["goal concerns John's basketball career"],"aggregation":"set",
"expected_type":"other","fallback":"..."}}]}}
The graph match only proposes a candidate; every source condition requires a
literal supporting quote. A source condition must be a concrete fact to check,
not a generic label such as "relevance" or "answer the question". For count,
`source_conditions` should specify the concrete membership criterion for one
event/item; deduplication and collection completeness are handled separately.
Do not claim exhaustiveness of a set or count from one matched member.
For a count query, expected_type describes one bound member (for example a
person or event), never the final numeric total.
"""
COMPILE_PLANS_REPAIR_TEMPLATE = COMPILE_PLANS_REPAIR_TEMPLATE.replace(
    "\n### INPUT\n{question}", _REPAIR_EXTENSION + "\n### INPUT\n{question}")


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
