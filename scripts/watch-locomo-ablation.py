#!/usr/bin/env python3
"""Live official LoCoMo F1 across the five component-study variants."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

VARIANTS = {
    'full': 'Completo', 'no-witness': 'Sem testemunhas',
    'no-time-reference': 'Sem referencia temporal', 'no-time-model': 'Sem modelagem temporal',
    'no-reflection': 'Sem reflexao',
}
CATEGORIES = ('single-hop', 'multi-hop', 'temporal', 'open-domain')


def load(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def records(root):
    """Conversation-qualified IDs; partial final JSONL writes are harmless."""
    rows = {}
    for path in sorted(root.glob('conversations/conv*/benchmark/*/locomo/witnessrag.jsonl')):
        conv = path.parents[3].name
        raw = path.read_text(encoding='utf-8')
        lines = raw.splitlines()
        for n, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                if n == len(lines)-1 and not raw.endswith('\n'):
                    continue
                raise ValueError(f'Malformed completed record: {path}:{n+1}')
            key = (conv, row['qid'])
            if key in rows:
                raise ValueError(f'Duplicate question ID: {key}')
            metric = row.get('f1_locomo')
            if not isinstance(metric, (int,float)) or not 0 <= metric <= 1:
                raise ValueError(f'Invalid official F1 for {key}')
            rows[key] = row
    return rows


def summarize(rows):
    values = list(rows.values())
    def cell(items):
        return {'n':len(items),'f1':100*sum(r['f1_locomo'] for r in items)/len(items) if items else None}
    return {'overall':cell(values),'by_category':{
        kind:cell([r for r in values if r.get('tipo') == kind]) for kind in CATEGORIES}}


def snapshot(root, expected=1540):
    state = load(root/'suite.json', {})
    all_rows = {name:records(root/name) for name in VARIANTS}
    reference = all_rows['full']
    result = {'updated_at':datetime.now(timezone.utc).isoformat(), 'metric':'official LoCoMo F1 x 100',
              'expected_per_variant':expected,'status':state.get('status','not_started'),'variants':{}}
    for name, rows in all_rows.items():
        status = (state.get('variants',{}).get(name) or {}).get('status','queued')
        if len(rows) == expected:
            status = 'complete' if load(root/name/'status.json',{}).get('status') == 'complete' else 'finalizing'
        common = set(rows) & set(reference)
        result['variants'][name] = {'label':VARIANTS[name],'status':status,
            'completed':len(rows),**summarize(rows), 'paired_with_full':{
                'n':len(common),'variant':summarize({k:rows[k] for k in common})['overall'],
                'full':summarize({k:reference[k] for k in common})['overall']}}
    return result


def render(data):
    timestamp = datetime.fromisoformat(data['updated_at']).astimezone().strftime('%d/%m %H:%M:%S')
    lines = [f'LoCoMo / Qwen2.5-14B / 40 fatos | {timestamp} | {data["status"]}',
             'F1 oficial (%) sobre questoes concluidas; entre parenteses: n da categoria.',
             'Parciais de variantes diferentes podem conter questoes diferentes; comparacao pareada em progress.json.',
             '',f'{"Configuracao":<25} {"Estado":<13} {"Questoes":>10} {"F1":>7} {"Single-hop":>13} {"Multi-hop":>13} {"Temporal":>13} {"Open-domain":>13}',
             '-'*126]
    def score(cell):
        return '-' if cell['f1'] is None else f"{cell['f1']:.2f} ({cell['n']})"
    for name, row in data['variants'].items():
        overall = '-' if row['overall']['f1'] is None else f"{row['overall']['f1']:.2f}"
        counts = f"{row['completed']}/{data['expected_per_variant']}"
        lines.append(f"{row['label']:<25} {row['status']:<13} {counts:>10} {overall:>7} " +
                     ' '.join(f"{score(row['by_category'][kind]):>13}" for kind in CATEGORIES))
    return '\n'.join(lines)


def write_snapshot(root, expected=1540):
    data = snapshot(root,expected)
    for filename, text in [('progress.json',json.dumps(data,ensure_ascii=False,indent=2)+'\n'),
                           ('progress.md','```text\n'+render(data)+'\n```\n')]:
        path = root/filename
        temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_text(text,encoding='utf-8')
        temp.replace(path)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--watch',action='store_true')
    p.add_argument('--interval',type=float,default=30)
    args = p.parse_args()
    if args.interval < 1:
        p.error('--interval must be >= 1')
    while True:
        if args.watch:
            print('\033[2J\033[H',end='')
        print(render(snapshot(args.output)),flush=True)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
