"""Two verifier calls: work != lead and own != play. No gold benchmark data."""
import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wrag.llm import get_llm, GenParams
from wrag.witness.portfolio import SYSTEM, CHECK_PROMPT, parse_contract, render_candidates, verify_coverage

llm = get_llm("openai", deployment="gpt-4o-mini", max_tokens=1600, use_cache=True)
cases = [
    ("Which city contains the laboratory Omar leads?", "Omar leads", "Omar leads that laboratory", [
        {"answer": "Recife", "facts": [
            {"triple": ("Omar", "works at", "Lab A"), "excerpt": "[D1:1] Omar: I work at Lab A as a technician. I do not lead it."},
            {"triple": ("Lab A", "located in", "Recife"), "excerpt": "[D1:2] Omar: Lab A is in Recife."}]},
        {"answer": "Salvador", "facts": [
            {"triple": ("Omar", "leads", "Lab B"), "excerpt": "[D1:3] Omar: I lead Lab B."},
            {"triple": ("Lab B", "located in", "Salvador"), "excerpt": "[D1:4] Omar: Lab B is in Salvador."}]}]),
    ("What instrument does Mei play?", "Mei play", "Mei actually plays the instrument", [
        {"answer": "guitar", "facts": [{"triple": ("Mei", "owns", "guitar"),
            "excerpt": "[D1:1] Mei: I own a guitar but I cannot play it."}]},
        {"answer": "violin", "facts": [{"triple": ("Mei", "plays", "violin"),
            "excerpt": "[D1:2] Mei: I play violin every week."}]}]),
]
rows = []
args = argparse.ArgumentParser()
args.add_argument("--last-only", action="store_true")
last_only = args.parse_args().last_only
if last_only:
    cases = cases[-1:]
for question, quote, condition, candidates in cases:
    contract = parse_contract({"operation": "lookup", "answer_type": "other", "ambiguous": False,
        "requirements": [{"id": "r1", "quote": quote, "condition": condition}]}, question)
    result = llm.chat(CHECK_PROMPT.format(question=question, contract=json.dumps(contract.to_dict()),
        plan="direct route; q0 and r1 required", candidates=render_candidates(candidates)),
        system=SYSTEM, params=GenParams(temperature=0, max_tokens=1600, json_mode=True), stage="witness.confirm")
    verdict = verify_coverage(result.json(), candidates, contract, contract.ids)
    rows.append({"question": question, "raw_verdict": result.json(), "verdict": verdict})
    print(json.dumps({"question": question, "supported": verdict["supported"],
                      "passed": verdict["supported"] == [1]}), flush=True)
filename = "contrast-repaired.json" if last_only else "contrast-results.json"
Path(__file__).with_name(filename).write_text(json.dumps({"cases": rows, "usage": llm.usage.snapshot()}, indent=2), encoding="utf-8")
assert all(row["verdict"]["supported"] == [1] for row in rows)
