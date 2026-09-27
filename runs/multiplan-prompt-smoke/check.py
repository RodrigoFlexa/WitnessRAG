"""Two synthetic questions; at most six API calls, no indexing/reader calls."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from test_proof_controller import retriever, POOL
import test_proof_controller as toy
from wrag.data import Question
from wrag.llm import get_llm

# Keep the synthetic source turns, use English graph predicates like LoCoMo.
relations = {"trabalha em": "works at", "localizada em": "located in", "acampou em": "camped at",
             "pinta": "paints", "le": "reads"}
toy.DIALOGUE = [(*row[:-1], (row[-1][0], relations[row[-1][1]], row[-1][2], row[-1][3])) for row in toy.DIALOGUE]
llm = get_llm("openai", deployment="gpt-4o-mini", max_tokens=2400, use_cache=True)
outputs = []
for qid, text in [("synthetic-chain", "Where is the company Ana works at located?"),
                  ("synthetic-list", "Where has Bruno camped?")]:
    question = Question(qid, text, [], dataset="toy")
    r = retriever(llm, multiplan_portfolio=True, portfolio_max_plans=2, proof_cycles=1,
                  fact_delivery="facts", fact_budget=8, fact_fill="question", fact_time="both",
                  typed_variables=True, item_set_proofs=True, witness_delivery="mixed",
                  relation_alternatives=True)
    result = r._retrieve_proof(question, 5, *POOL)
    d = result.diagnostics
    outputs.append({"question": text, "diagnostics": d})
    print(json.dumps({"question": text, "contract_valid": bool(d["contrato"]),
                      "plans": d["planejamento"]["planos_distintos"],
                      "proofs_delivered": d["n_testemunhas_no_contexto"],
                      "calls": d["planejamento"]["chamadas"]}), flush=True)
Path(__file__).with_name("results.json").write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
