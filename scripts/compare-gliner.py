import json, glob, random, statistics as st
from collections import defaultdict
from wrag.eval.locomo_official import score_record
def load(dirs):
    rows = {}
    for d in dirs:
        for f in glob.glob(f'{d}/conversations/*/benchmark/*/locomo/witnessrag.jsonl'):
            for l in open(f):
                r = json.loads(l)
                if 'f1_locomo' not in r: r.update(score_record(r))
                rows[r['qid']] = r
    return rows
base = load(['runs/base-v4', 'runs/base-v4-rest'])
typ = load(['runs/gliner-rank'])
common = sorted(set(base) & set(typ))
cats = ['single-hop', 'multi-hop', 'temporal', 'open-domain']
def f1(rows, ids): return 100 * st.mean(rows[q]['f1_locomo'] for q in ids) if ids else float('nan')
conv = lambda q: q.split(':')[1]
convs = sorted({conv(q) for q in common})
print(f'perguntas pareadas: {len(common)} em {len(convs)} conversas')
print(f'{"categoria":12s} {"n":>5s} {"sem":>7s} {"com":>7s} {"delta":>7s}  IC95 (bootstrap por conversa)')
random.seed(0)
for cat in ['todas'] + cats:
    ids = [q for q in common if cat == 'todas' or base[q]['tipo'] == cat]
    by = defaultdict(list)
    for q in ids: by[conv(q)].append(q)
    deltas = []
    for _ in range(2000):
        sample = [q for c in random.choices(list(by), k=len(by)) for q in by[c]]
        deltas.append(f1(typ, sample) - f1(base, sample))
    deltas.sort()
    print(f'{cat:12s} {len(ids):5d} {f1(base, ids):7.2f} {f1(typ, ids):7.2f} {f1(typ, ids)-f1(base, ids):+7.2f}  [{deltas[50]:+.2f}; {deltas[1950]:+.2f}]')
# onde o tipo agiu
applied = changed_first = diff_ctx = diff_ans = 0
for q in common:
    d = typ[q].get('diagnosticos', {})
    tipos = [c.get('tipos') for c in d.get('ciclos', []) if isinstance(c, dict) and c.get('tipos')]
    if any(t.get('aplicado') for t in tipos): applied += 1
    if typ[q]['recuperadas'] != base[q]['recuperadas']: diff_ctx += 1
    if typ[q]['resposta'] != base[q]['resposta']: diff_ans += 1
print(f'tipo aplicado (ordenou) em {applied} perguntas; contexto diferente em {diff_ctx}; resposta diferente em {diff_ans}')
better = sum(typ[q]['f1_locomo'] > base[q]['f1_locomo'] for q in common)
worse = sum(typ[q]['f1_locomo'] < base[q]['f1_locomo'] for q in common)
print(f'perguntas que melhoraram {better}, pioraram {worse}')
