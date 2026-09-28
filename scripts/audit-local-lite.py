#!/usr/bin/env python3
"""Offline audit of delivered evidence, using annotations ONLY after retrieval.

Annotated-turn presence is a retrieval diagnostic, not answer accuracy, semantic
sufficiency or a certificate. No gold data is ever passed back to the method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def evaluate(records, annotated_qa):
    out = []
    for row in records:
        index = int(row['qid'].split(':qa')[-1])
        qa = annotated_qa[index]
        if row.get('pergunta') and row['pergunta'] != qa['question']:
            raise ValueError('Question and annotation mismatch')
        evidence = set(qa.get('evidence', []))
        if not evidence:
            continue
        diag = row['diagnosticos']
        text = '\n'.join(b['text'] for b in diag.get('trechos_extras', []))
        delivered = set(re.findall(r'\[(D\d+:\d+)\]', text))
        out.append({'qid': row['qid'], 'category': row.get('tipo', ''),
                    'annotated_turn_recall': len(evidence & delivered) / len(evidence),
                    'all_annotated_turns_present': evidence <= delivered,
                    'missing_turns': sorted(evidence - delivered),
                    'latency': row['latencia_recuperacao_s']})
    return out


def summary(records):
    return {'n': len(records), 'mean_annotated_turn_recall': statistics.mean(r['annotated_turn_recall'] for r in records),
            'all_annotated_turns_present': sum(r['all_annotated_turns_present'] for r in records),
            'mean_retrieval_seconds': statistics.mean(r['latency'] for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--conversation', type=int, default=0)
    parser.add_argument('--before', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = rows(args.predictions)
    annotated = json.loads(args.annotations.read_text(encoding='utf-8'))[args.conversation]['qa']
    evaluated = evaluate(records, annotated)
    groups = {k: [r for r in evaluated if r['category'] == k] for k in sorted({r['category'] for r in evaluated})}
    checks = [c for row in records for c in row['diagnosticos']['support_reflector']['checks']]
    audit = {'protocol': 'offline_evidence_audit_not_answer_accuracy', 'summary': summary(evaluated),
             'categories': {k: summary(v) for k, v in groups.items()}, 'questions': evaluated,
             'missing_support_checks': sum(c['status'] == 'missing_support' for c in checks),
             'n_support_checks': len(checks),
             'inference_paths_disabled': all(row['diagnosticos']['online_inference']['llm_calls'] == 0
                                            and row['diagnosticos']['online_inference']['query_embedding_calls'] == 0
                                            and row['diagnosticos']['online_inference']['cross_encoder_pairs'] == 0 for row in records),
             'prediction_sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest()}
    if args.before:
        previous = evaluate(rows(args.before), annotated)
        previous_map = {r['qid']: r for r in previous}
        pairs = [(previous_map[r['qid']], r) for r in evaluated if r['qid'] in previous_map]
        audit['paired'] = {'before': summary([a for a, _ in pairs]), 'after': summary([b for _, b in pairs]),
                          'improved_evidence_coverage': sum(b['annotated_turn_recall'] > a['annotated_turn_recall'] for a, b in pairs),
                          'worsened_evidence_coverage': sum(b['annotated_turn_recall'] < a['annotated_turn_recall'] for a, b in pairs),
                          'timing_note': 'historical before run, not a simultaneous hardware benchmark'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k != 'questions'}, indent=2))


if __name__ == '__main__':
    main()
