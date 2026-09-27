"""Separação do tipo: cosseno (bge-m3) contra GLiNER nas candidatas registradas."""
import json, re, sys, glob
from collections import defaultdict
import numpy as np
log = sys.argv[1]
gold = {}
for f in glob.glob('/home/claude/wr/runs/base-v4/conversations/*/data/locomo.json'):
    for q in json.load(open(f)):
        gold[q['id']] = [str(q['answer'])] if not isinstance(q['answer'], list) else [str(a) for a in q['answer']]
STOP = set("a an the of and or to in on at for with my his her their is was".split())
tok = lambda t: {w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP}
recs = [json.loads(l) for l in open(log)]
rows, seen = [], set()
for r in recs:
    kind = r.get('tipo_resposta') or ''
    if not kind or r['qid'] not in gold: continue
    g = set().union(*[tok(a) for a in gold[r['qid']]])
    for c in r['candidatas'][:12]:
        key = (r['qid'], c['resposta'], kind)
        if key in seen: continue
        seen.add(key)
        rows.append(dict(qid=r['qid'], kind=kind, ans=c['resposta'], falas=c['falas'],
                         pos=bool(tok(c['resposta']) & g), plural=len(r['candidatas'])))
print('registros', len(recs), 'com tipo', len({x['qid'] for x in rows}), 'candidatas', len(rows),
      'positivas', sum(x['pos'] for x in rows))
from wrag.witness.typing import GlinerTyper
t = GlinerTyper()
for x in rows:
    x['gl'] = t.score(x['ans'], x['falas'], x['kind'])
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('BAAI/bge-m3', device='cpu')
va = m.encode([x['ans'] for x in rows], normalize_embeddings=True, batch_size=32)
vk = m.encode([x['kind'] for x in rows], normalize_embeddings=True, batch_size=32)
for x, a, k in zip(rows, va, vk): x['cos'] = float(a @ k)
def auc(score, rows):
    p = [score(x) for x in rows if x['pos']]; n = [score(x) for x in rows if not x['pos']]
    if not p or not n: return float('nan')
    return np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in p for b in n])
gl = lambda x: 0.5 if x['gl'] is None else x['gl']
print('AUC cosseno', round(auc(lambda x: x['cos'], rows), 3), ' AUC GLiNER', round(auc(gl, rows), 3))
loc = [x for x in rows if x['gl'] is not None]
print('localizadas', len(loc), 'de', len(rows))
for th in (0.2, 0.3, 0.4, 0.5):
    keep = [x for x in rows if x['gl'] is None or x['gl'] >= th]
    tp = sum(x['pos'] for x in keep); P = sum(x['pos'] for x in rows)
    print(f'limiar {th}: mantém {len(keep)}/{len(rows)}; positivas mantidas {tp}/{P}; negativas cortadas {sum(not x["pos"] for x in rows)-sum(not x["pos"] for x in keep)}/{sum(not x["pos"] for x in rows)}')
json.dump(rows, open(log + '.scored.json', 'w'), ensure_ascii=False)
by = defaultdict(list)
for x in rows: by[x['kind']].append(x)
for k, xs in sorted(by.items(), key=lambda kv: -len(kv[1]))[:12]:
    print(k, [(x['ans'][:25], '+' if x['pos'] else '-', None if x['gl'] is None else round(x['gl'], 2), round(x['cos'], 2)) for x in xs[:6]])
