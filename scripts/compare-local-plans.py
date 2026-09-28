#!/usr/bin/env python3
"""Paired local-plan comparison; logical token usage includes cache hits."""
import argparse
import json
import statistics
from pathlib import Path


def load(root):
    root=Path(root)
    status_path=(root.parent if root.is_file() else root)/'run_status.json'
    if status_path.exists() and json.loads(status_path.read_text(encoding='utf-8')).get('status')=='invalid_aborted':
        raise ValueError('This run was marked invalid; use the corrected output folder')
    files=[root] if root.is_file() else ([root/'predictions.jsonl'] if (root/'predictions.jsonl').exists()
                                        else sorted(root.rglob('witnessrag.jsonl'),key=lambda p:p.stat().st_mtime))
    rows={}
    for file in files:
        for line in file.read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row.get('filtrada'):continue
            rows[row['qid']]=row
    return rows


def tokens(row):
    total=row.get('uso_llm',{}).get('total',{})
    return total.get('tokens_prompt',0)+total.get('tokens_resposta',0)


def compare(before,after):
    common=sorted(set(before)&set(after))
    if not common:raise ValueError('No matched questions')
    if any(before[q]['pergunta']!=after[q]['pergunta'] or before[q]['tipo']!=after[q]['tipo'] for q in common):
        raise ValueError('Question text or categories differ')
    def group(ids):
        f1=lambda rows:statistics.mean(rows[q]['f1_locomo'] for q in ids)
        bleu=lambda rows:statistics.mean(rows[q]['bleu1_locomo'] for q in ids)
        deltas=[after[q]['f1_locomo']-before[q]['f1_locomo'] for q in ids]
        return {'n':len(ids),'f1_before':f1(before),'f1_after':f1(after),'f1_delta':f1(after)-f1(before),
                'bleu_before':bleu(before),'bleu_after':bleu(after),
                'tokens_before':statistics.mean(tokens(before[q]) for q in ids),
                'tokens_after':statistics.mean(tokens(after[q]) for q in ids),
                'improved':sum(d>1e-8 for d in deltas),'worsened':sum(d<-1e-8 for d in deltas),
                'tied':sum(abs(d)<=1e-8 for d in deltas)}
    return {'paired':True,'question_ids':common,'total':group(common),
            'categories':{kind:group([q for q in common if after[q]['tipo']==kind])
                          for kind in sorted({after[q]['tipo'] for q in common})}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=compare(load(args.before),load(args.after))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# Comparação pareada de planos locais','',f'Antes: `{args.before}`. Depois: `{args.after}`.','',
           'Tokens lógicos incluem cache. Resultados de desenvolvimento; verifique modelos e memória antes de generalizar.','',
           '| Categoria | n | F1 antes | F1 depois | Δ F1 (pp) | BLEU antes | BLEU depois | Tokens antes | Tokens depois |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for kind,g in list(result['categories'].items())+[('total',result['total'])]:
        lines.append(f"| {kind} | {g['n']} | {100*g['f1_before']:.2f} | {100*g['f1_after']:.2f} | {100*g['f1_delta']:+.2f} | {100*g['bleu_before']:.2f} | {100*g['bleu_after']:.2f} | {g['tokens_before']:.0f} | {g['tokens_after']:.0f} |")
    g=result['total']
    lines += ['',f"F1 por pergunta: {g['improved']} melhoraram, {g['worsened']} pioraram e {g['tied']} empataram."]
    args.output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result['total'],ensure_ascii=False))


if __name__=='__main__':main()
