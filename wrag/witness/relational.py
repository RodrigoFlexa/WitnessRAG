"""Executor independente de consultas conjuntivas via junções SQLite.

Relações e constantes usam identidade lexical conservadora, sem feixe, top-k,
embeddings ou inferência de inversas. Exaustivo sobre os fatos visíveis; a saída
intermediária pode ser exponencial. Serve como controle da busca WITNESS-RAG.
"""
from __future__ import annotations

import sqlite3

from wrag import config as C
from wrag.util import canonical_symbol
from wrag.witness.query import is_var, var_name
from wrag.witness.search import SearchResult, Witness, _minimal_only


class SQLWitnessSearcher:
    def __init__(self, memory, cfg: C.WitnessConfig):
        self.kg = memory
        self.cfg = cfg
        self.allowed = None
        self._allowed_horizon = len(memory.facts)
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE TABLE facts (fid INTEGER PRIMARY KEY, s INTEGER, r TEXT, o INTEGER, active INTEGER)")
        self.db.executemany("INSERT INTO facts VALUES (?, ?, ?, ?, 1)",
                            [(i, f.subj_id, canonical_symbol(f.relation), f.obj_id)
                             for i, f in enumerate(memory.facts)])
        self.db.execute("CREATE INDEX sr ON facts (s, r)")
        self.db.execute("CREATE INDEX ro ON facts (r, o)")

    def rollback(self):
        pass  # este executor não realiza aquisição

    def close(self):
        self.db.close()

    def join(self, query):
        if not query.atoms or query.aggregation != "none" or query.answer_var not in query.variables():
            return SearchResult()
        self.db.execute("UPDATE facts SET active = ?", (int(self.allowed is None),))
        if self.allowed is not None:
            self.db.executemany("UPDATE facts SET active = 1 WHERE fid = ?", [(i,) for i in self.allowed])
        tables, conditions, params, variables = [], [], [], {}
        for i, atom in enumerate(query.atoms):
            alias = f"a{i}"
            tables.append(f"facts {alias}")
            conditions.extend([f"{alias}.r = ?", f"{alias}.active = 1"])
            params.append(canonical_symbol(atom.relation))
            for term, column in ((atom.subject, "s"), (atom.object, "o")):
                expression = f"{alias}.{column}"
                if is_var(term):
                    name = var_name(term)
                    if name in variables:
                        conditions.append(f"{expression} = {variables[name]}")
                    else:
                        variables[name] = expression
                else:
                    eid = self.kg.entity_id(term)
                    if eid is None:
                        return SearchResult(exhaustive=True)
                    conditions.append(f"{expression} = ?")
                    params.append(eid)
        projection = [f"a{i}.fid" for i in range(len(query.atoms))] + list(variables.values())
        sql = "SELECT " + ", ".join(projection) + " FROM " + ", ".join(tables)
        sql += " WHERE " + " AND ".join(conditions)
        witnesses = []
        n = len(query.atoms)
        for row in self.db.execute(sql, params):
            facts = tuple(sorted(set(row[:n])))
            bindings = {v: self.kg.entities[eid] for v, eid in zip(variables, row[n:])}
            pids = tuple(sorted({self.kg.facts[fid].pid for fid in facts}))
            witnesses.append(Witness(facts, bindings, 1.0, self.cfg.passage_penalty * len(pids),
                                     pids, bindings[query.answer_var]))
        return SearchResult(witnesses=_minimal_only(witnesses), exhaustive=True,
                            depth_reached=n if witnesses else 0)
