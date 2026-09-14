"""Controle relacional: mesmo compilador/leitor, SQL exato, sem aquisição."""
from dataclasses import replace

from wrag.methods.witnessrag import WitnessRAGRetriever
from wrag.witness.relational import SQLWitnessSearcher


class RelationalRetriever(WitnessRAGRetriever):
    name = "relational"

    def __init__(self, ctx):
        witness = replace(ctx.run.witness, grounding_mode="exact", enable_acquisition=False,
                          candidates_per_atom=0, beam_width=0, max_witnesses=0)
        super().__init__(replace(ctx, run=replace(ctx.run, witness=witness)))

    def index(self):
        super().index()
        self.searcher = SQLWitnessSearcher(self.memory, self.ctx.run.witness)

    def _partial_passages(self, query, k):
        return [], []

    def index_report(self):
        report = super().index_report()
        report["executor"] = "SQLite; junção exaustiva; sem aquisição"
        return report
