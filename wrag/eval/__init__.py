"""Avaliação: métricas, leitor comum, orquestração e relatório."""

from wrag.eval import metrics
from wrag.eval.reader import read
from wrag.eval.report import build_report
from wrag.eval.runner import run, run_dataset

__all__ = ["metrics", "read", "run", "run_dataset", "build_report"]
