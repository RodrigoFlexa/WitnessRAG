"""Read-only, question-level audit of the two local conv00 runs."""
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'audits' / 'plan-repair-conv00'


def load(p):
    return json.loads(p.read_text(encoding='utf-8'))


def rows(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x]


def ci(values):
    rng = random.Random(42)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(20000))
    return [draws[499], draws[19499]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    newpath = ROOT / 'runs/locomo-plan-repair-02/benchmark/20260920-234016-169721-qwen-pilot/locomo/witnessrag.jsonl'
    oldpath = next((ROOT / 'runs/locomo-plan-repair-01/conversations/conv00/benchmark').glob('*/locomo/witnessrag.jsonl'))
    new, old = rows(newpath), rows(oldpath)
    old = {r['qid']: r for r in old}
    assert len(new) == len(old) == 102 and {r['qid'] for r in new} == old.keys()
    records = []
    for r in new:
        d = r['diagnosticos']; ps = d['planos_compilados']; o = old[r['qid']]
        top = set(r['recuperadas']); gold = set(r['passagens_ouro'])
        frontier = set(d['pesquisa_provas']['fronteira'])
        acquired = {p for a in d['pesquisa_provas']['acoes'] for p in a['passagens']}
        candidates = {c['pid'] for p in ps for cs in p.get('trilha_juncao', {}).get('top_candidates', []) for c in cs}
        records.append(dict(qid=r['qid'], tipo=r['tipo'], question=r['pergunta'], gold=' | '.join(r['respostas_ouro']),
            answer=r['resposta'], old_answer=o['resposta'], f1=r['f1_locomo'], old_f1=o['f1_locomo'],
            delta=r['f1_locomo']-o['f1_locomo'], all_support=gold <= top,
            missing=';'.join(sorted(gold-top)), missing_in_frontier=';'.join(sorted((gold-top)&frontier)),
            missing_in_acquisition=';'.join(sorted((gold-top)&acquired)),
            missing_in_logged_candidates=';'.join(sorted((gold-top)&candidates)),
            closed=sum(p['fechou'] for p in ps), covered_closed=sum(p['fechou'] and bool((p.get('obrigacoes') or {}).get('cobre_pergunta')) for p in ps),
            zero_candidates=sum(0 in p['candidatos_por_atomo'] for p in ps),
            binding_conflict=sum(p['executavel'] and not p['fechou'] and bool(p['candidatos_por_atomo']) and all(p['candidatos_por_atomo']) for p in ps),
            proof=d['classe_prova'], failure=d['classe_falha_plano'],
            verified=d['verificacao']['avaliadas'], accepted=d['verificacao']['aceitas'],
            verification_failures=json.dumps(d['verificacao']['rejeicoes_por_tipo']),
            context_changed=d['contexto_alterado_pelo_witness'],
            same_order=o['recuperadas']==r['recuperadas'], same_set=set(o['recuperadas'])==top,
            llm=r['uso_llm']['total']['chamadas'], hits=r['uso_llm']['total']['em_cache'],
            seconds=r['latencia_recuperacao_s']+r['latencia_leitura_s']))
    with (OUT/'questions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    summary={}
    for kind in ('all','single-hop','multi-hop'):
        rr=[r for r in records if kind=='all' or r['tipo']==kind]
        raw=[r for r in new if kind=='all' or r['tipo']==kind]
        summary[kind]={'n':len(rr),'f1':mean(r['f1'] for r in rr),'old_f1':mean(r['old_f1'] for r in rr),
          'wins_losses_ties':[sum(r['delta']>0 for r in rr),sum(r['delta']<0 for r in rr),sum(r['delta']==0 for r in rr)],
          'same_order':sum(r['same_order'] for r in rr),'same_set':sum(r['same_set'] for r in rr),
          'context_changed':sum(r['context_changed'] for r in rr), 'proof':dict(Counter(r['proof'] for r in rr)),
          'failure':dict(Counter(r['failure'] for r in rr)),
          'questions_any_closed':sum(r['closed']>0 for r in rr),'questions_covered_closed':sum(r['covered_closed']>0 for r in rr),
          'questions_zero_candidates':sum(r['zero_candidates']>0 for r in rr),
          'questions_binding_conflict':sum(r['binding_conflict']>0 for r in rr),
          'verification':dict(sum((Counter(r['diagnosticos']['verificacao']['rejeicoes_por_tipo']) for r in raw),Counter())),
          'verified':sum(r['verified'] for r in rr),'accepted':sum(r['accepted'] for r in rr),
          'llm':mean(r['llm'] for r in rr),'hits':sum(r['hits'] for r in rr),'seconds':mean(r['seconds'] for r in rr),
          'complete_support':{str(b):{'n':sum(r['all_support']==b for r in rr),'f1':mean([r['f1'] for r in rr if r['all_support']==b] or [0])} for b in (True,False)},
          'missing_in_frontier':sum(bool(r['missing_in_frontier']) for r in rr),
          'missing_in_acquisition':sum(bool(r['missing_in_acquisition']) for r in rr),
          'missing_in_logged_candidates':sum(bool(r['missing_in_logged_candidates']) for r in rr)}
    mh=[r for r in records if r['tipo']=='multi-hop']
    summary['multi-hop']['conditional_question_bootstrap_ci95']=ci([r['f1'] for r in mh])
    summary['multi-hop']['paired_conditional_question_bootstrap_ci95']=ci([r['delta'] for r in mh])
    summary['old_without_guard_multi_f1']=mean(r.get('f1_locomo_sem_guarda', r['f1_locomo']) for r in old.values() if r['tipo']=='multi-hop')
    by_id={r['qid']:r for r in new}
    swaps=[]
    for r in old.values():
        if r['diagnosticos']['classe_prova']!='nenhuma':
            continue
        d=r['diagnosticos']; base=by_id[r['qid']]['recuperadas']
        used={p for a in d['pesquisa_provas']['acoes'] for p in a['passagens']}
        fifth=next((p for p in d['pesquisa_provas']['fronteira'] if p not in base and p not in used),None)
        swaps.append(r['recuperadas']==base[:4]+[fifth])
    summary['old_fallback_swap_reconstruction']={'matched':sum(swaps),'n':len(swaps)}
    summary['all_missing_supports_in_frontier']=sum(bool(set(r['passagens_ouro'])-set(r['recuperadas'])) and
        (set(r['passagens_ouro'])-set(r['recuperadas'])) <= set(r['diagnosticos']['pesquisa_provas']['fronteira']) for r in new)
    control=ROOT/'runs/locomo-controlled-01/controlled/158a0d2a68689a0db8839076'
    cells=[load(p)['cells'] for p in (control/'answers').glob('*.json')]
    summary['control_conv00']={cell:{kind:mean(c[cell]['f1_locomo'] for c in cells if c[cell]['tipo']==kind)
                                    for kind in ('single-hop','multi-hop')} for cell in cells[0]}
    summary['control_reader_pairs']={}
    for base,change in [('evidence/common','evidence/proof'),('soft-v2/common','soft-v2/proof')]:
        assert all(c[base]['retrieval_hash']==c[change]['retrieval_hash'] and c[base]['recuperadas']==c[change]['recuperadas'] for c in cells)
        deltas=[c[change]['f1_locomo']-c[base]['f1_locomo'] for c in cells if c[base]['tipo']=='multi-hop']
        summary['control_reader_pairs'][base+' -> '+change]={'delta':mean(deltas),'wins':sum(x>0 for x in deltas),'losses':sum(x<0 for x in deltas),'ties':sum(x==0 for x in deltas)}
    memory=load(control/'memory.json')
    assert memory['sha256']==hashlib.sha256((control/'memory.pkl').read_bytes()).hexdigest()
    summary['verified_memory_sha256']=memory['sha256']
    summary['input_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (newpath,oldpath)}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    for r in mh:
        print(json.dumps(r,ensure_ascii=False))


if __name__=='__main__':
    main()
