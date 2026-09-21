"""Offline postmortem. Raw-answer scores are diagnostics, never validated answers."""
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wrag.eval.locomo_official import question_score
from wrag.eval.reader_pilot import digest
from wrag.witness.verification import _quote_in_source


def main():
    run = ROOT / 'runs/locomo-reader-cited-01'
    out = ROOT / 'docs/audits/plan-repair-conv00/reader-failure'
    out.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(s) for s in (run/'pairs.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    assert digest(manifest['snapshots']) == manifest['context_sha256']
    snapshots = {s['qid']:s for s in manifest['snapshots']}
    baseline = ROOT/'runs/locomo-plan-repair-02/benchmark/20260920-234016-169721-qwen-pilot/locomo/witnessrag.jsonl'
    gold = {r['qid']:r for r in map(json.loads, baseline.read_text(encoding='utf-8').splitlines())}
    rows, counts = [], Counter()
    for p in pairs:
        assert p['context_hash'] == digest(snapshots[p['qid']])
        c = p['cited']; raw = c.get('raw_json')
        items = raw.get('items',[]) if isinstance(raw,dict) else []
        raw_answer = ', '.join(i.get('answer','') for i in items) or 'insufficient information'
        g = gold[p['qid']]
        raw_f1 = question_score(raw_answer,g['respostas_ouro'][0],g['categoria_locomo']) if p['eligible'] else p['common']['f1_locomo']
        status = ('truncated' if c.get('finish_reason')=='length' else
                  'local_rejection_present' if c.get('rejected') else 'generated_answer')
        row = {'qid':p['qid'],'tipo':p['tipo'],'question':g['pergunta'],
               'eligible':p['eligible'],'common':p['common']['answer'],'final':c['answer'],
               'raw_unvalidated':raw_answer if p['eligible'] else p['common']['answer'],
               'common_f1':p['common']['f1_locomo'],'final_f1':c['f1_locomo'],
               'raw_unvalidated_f1':raw_f1,'delta':p['delta'],'status':status}
        rows.append(row)
        if not p['eligible']:
            continue
        counts['eligible'] += 1
        counts['final_abstentions'] += c['answer']=='insufficient information'
        counts['one_json_item'] += len(items)==1
        counts['one_item_with_commas'] += len(items)==1 and ',' in items[0].get('answer','')
        counts['truncated'] += c.get('finish_reason')=='length'
        counts['rejected_items'] += sum(x['reason']=='invalid_provenance' for x in c.get('rejected',[]))
        sources={s['pid']:s['title']+'\n'+s['text'] for s in snapshots[p['qid']]['passages']}
        for item in items:
            for citation in item.get('evidence',[]):
                pid,quote=citation.get('pid'),citation.get('quote','')
                if pid in sources and _quote_in_source(quote,sources[pid]):
                    counts['citation_exact'] += 1
                elif any(_quote_in_source(quote,s) for s in sources.values()):
                    counts['citation_wrong_pid_only'] += 1
                else:
                    counts['citation_absent_from_all_five'] += 1
    summary={'counts':dict(counts),'common_f1_different_from_historical':sum(p['common']['f1_locomo']!=p['historical_f1'] for p in pairs),
             'context_hashes_verified':len(pairs),
             'warning':'Raw scores ignore provenance and can reward unsupported extra items. Not a proposed repair.'}
    for kind in ('single-hop','multi-hop'):
        subset=[r for r in rows if r['tipo']==kind]
        summary[kind]={k:mean(r[k] for r in subset) for k in ('common_f1','final_f1','raw_unvalidated_f1')}
        summary[kind]['loss_status']=dict(Counter(r['status'] for r in subset if r['delta']<0))
    with (out/'questions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
