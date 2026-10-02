"""Multi-origin local programs with reserved search budgets and role checks.

Graph executions establish conjunctive bindings only. Question parsing, type
checks, local relevance and source antecedents are explicit hypotheses. All
facts and quotations come from the existing memory; no generative call occurs.
"""
from __future__ import annotations

import itertools
import re
import time
from collections import defaultdict, deque
from dataclasses import replace

import numpy as np

from wrag.methods.base import RetrievalResult
from wrag.util import canonical_symbol
from wrag.witness.local_contract import SourceIndex, reading_contract, stems
from wrag.witness.local_plans import Candidate, LocalPlanner, terms
from wrag.witness.plan import ProofPlan
from wrag.witness.query import Atom, ConjunctiveQuery, is_var


class MultiOriginPlanner(LocalPlanner):
    def __init__(self, r, question):
        super().__init__(r, question)
        # Keep literal entities AND factual starting hypotheses. A graph noun
        # such as "book" is a possible start, not automatically the answer.
        self.mentions = self.literal_mentions()
        self.reading = reading_contract(self.text, self.dated, [name for name,_ in self.mentions])
        self.reading.temporal = self.contract
        # Operation refinements must survive the base temporal contract.
        refined = reading_contract(self.text,self.dated,[name for name,_ in self.mentions]).temporal
        self.contract.operation = refined.operation
        self.contract.pending = list(dict.fromkeys(self.contract.pending + [p for p in refined.pending
            if p!='event_relative_anchor_unresolved' or self.contract.reference_fact is None]))
        self.owner_nodes = set().union(*(nodes for _,nodes in self.mentions)) if self.mentions else set()
        speakers={turn.speaker.casefold() for turns in self.dated.turns.values() for turn in turns if turn.speaker}
        self.participants=[(name,nodes) for name,nodes in self.mentions if name.casefold() in speakers]
        self.participant_nodes=set().union(*(nodes for _,nodes in self.participants)) if self.participants else set()
        order = sorted(self.available,key=lambda i:(-float(self.sim[i]),i))[:self.cfg.local_plan_starts]
        factual = [node for i in order for node in (self.node(self.memory.facts[i].subj_id), self.node(self.memory.facts[i].obj_id)) if node>=0]
        literal = [node for _,nodes in self.mentions for node in sorted(nodes)]
        self.anchors = list(dict.fromkeys(literal + self.anchors + factual))[:self.cfg.local_plan_starts]
        self.generation_budget = self.cfg.local_plan_executions
        self.phase_counts = defaultdict(int)
        self._relation_terms = {rel:stems(rel) for rel in self.by_relation}
        self._fact_terms = {i:stems(self.memory.facts[i].verbalize()) for i in self.available}
        self._all = {}
        self.timings = {}
        self._source_terms = {}
        self.reference = None
        self.anchor_resolution = None
        self.reference_applied = []
        if self.cfg.temporal_reference and self.cfg.study_ablation not in {"no-time-reference", "no-time-model"}:
            from wrag.witness.temporal_reference import plan_reference
            self.reference = plan_reference(self.r, self.text)
            self.apply_reference()

    def apply_reference(self):
        """Record the declared temporal reference; never rewrite the search.

        In the conv03 study (v3), letting the LLM reference change the program
        search (date operation, event anchors, order periods) cost 9.6 F1
        points on the 17 questions it touched, while projecting candidate times
        for date questions gained 4.7 on 20. The reference therefore keeps the
        rule-based contract intact (explicit question dates still scope the
        search through the grammar) and acts where it helped: deciding that a
        question asks for a time and projecting that time from the witnesses.
        """
        from wrag.witness.temporal_reference import asks_time
        ref = self.reference
        if ref is None or not ref.valid:
            return
        self.reference_applied = ["date_question"] if asks_time(ref, self.text, self.contract.operation) else []

    def temporal_ok(self,fid,scoped):
        # A proposed future event cannot witness an explicitly occurred event.
        # This is independent of recency; missing modality remains uncertain.
        if scoped and self.memory.facts[fid].kind == 'plan' and re.search(
                r'\bdid\b|\bhas .*gone\b|\bwent\b|\bhappened\b|\bhow many times\b',self.text,re.I):
            return False
        return super().temporal_ok(fid,scoped)

    def literal_mentions(self):
        found = []
        for name,nodes in self.entity_nodes.items():
            if len(name)<3:
                continue
            match = re.search(r"(?<!\w)"+re.escape(name)+r"(?!\w)",self.text,re.I)
            if match and match[0][0].isupper():
                found.append((match[0],nodes))
        chosen=[]
        for name,nodes in sorted(found,key=lambda item:(-len(item[0]),item[0])):
            if not any(name.casefold() in full.casefold() for full,_ in chosen):
                chosen.append((name,nodes))
        return sorted(chosen,key=lambda item:self.text.casefold().find(item[0].casefold()))[:6]

    def relation_intent(self, relation):
        target=self.reading.predicate_terms
        return len(target & self._relation_terms.get(canonical_symbol(relation),stems(relation)))/max(1,len(target))

    def checks(self, c, w):
        atoms=c.query.atoms
        bindings=w.bindings
        constants={n for value in c.query.constants() for n in self.entity_nodes.get(canonical_symbol(value),set())}
        covered=[bool(nodes & constants) for _,nodes in self.mentions]
        owners=sum(covered)/max(1,len(covered)) if covered else 1.0
        intent=set().union(*(self._relation_terms.get(canonical_symbol(a.relation),stems(a.relation)) for a in atoms))
        matched=self.reading.predicate_terms & intent
        target=len(matched)/max(1,len(self.reading.predicate_terms))
        projection=True
        # Safe contradiction: a named participant cannot be the requested book,
        # topic, place or activity. Keep it for expansion, exclude as an answer.
        answer_nodes=self.entity_nodes.get(canonical_symbol(w.answer),set())
        if (self.reading.head or self.contract.expected_type == "place") and answer_nodes & self.owner_nodes:
            projection=False
        if re.match(r'^(?:what|which|where)\b',self.text,re.I) and answer_nodes & self.participant_nodes:
            projection=False
        # An active read/buy/paint question asks for the object, while "who
        # supports X" asks for the subject. Enforce only when a literal predicate
        # match exists; otherwise this grammar leaves the decision uncertain.
        if c.query.answer_var == "x" and matched and self.reading.answer_role != "either":
            slots=[a for a in atoms if self._relation_terms.get(canonical_symbol(a.relation),set()) & matched]
            desired=[getattr(a,self.reading.answer_role) for a in slots]
            if not any(value=="?x" for value in desired):
                projection=False
        if self.contract.expected_type == "person" and w.answer and re.fullmatch(r"\d+",w.answer):
            projection=False
        scope_supported=True
        if self.reading.subject_scope:
            scope_terms=stems(self.reading.subject_scope)
            relevant=[i for i in w.facts if self._relation_terms.get(canonical_symbol(self.memory.facts[i].relation),set()) & matched]
            for i in relevant:
                if i not in self._source_terms:self._source_terms[i]=stems(self.dated.excerpt(i,window=0))
            scope_supported=any(scope_terms & (self._fact_terms[i] | self._source_terms[i]) for i in relevant)
            projection=projection and scope_supported
        # Explicit both/each is a reliable structural requirement. Other named
        # entities may be qualifiers; coverage remains a ranking hypothesis.
        required = not self.reading.conjunction or all(covered)
        if self.reading.conjunction:
            # Merely mentioning both owners is insufficient: each must occupy
            # the owner role of the requested predicate, not e.g. paint(A,x)
            # AND love(B,x). This also works for owners bound through a bridge.
            actor_slot='obj_id' if self.reading.answer_role=='subject' else 'subj_id'
            requested_owners={self.node(getattr(self.memory.facts[i],actor_slot))
                              for atom in atoms if self._relation_terms.get(canonical_symbol(atom.relation),set()) & matched
                              for i in self.bound_fact_ids(atom,w)}
            required=required and bool(matched) and all(bool(nodes & requested_owners) for _,nodes in self.mentions)
        shared_literal=True
        if self.reading.conjunction and c.query.answer_var=='x':
            generic=stems('book painting art object item thing subject topic person place event')
            values=[]
            for atom in atoms:
                slots=[slot for slot in ('subject','object') if getattr(atom,slot)=='?x']
                if not slots:continue
                words=set().union(*(stems(getattr(self.memory.facts[i],slot))-generic
                                  for i in self.bound_fact_ids(atom,w) for slot in slots))
                values.append(words)
            # A broad semantic entity cluster is insufficient to identify the
            # same specific value for explicit both/each. Leave uncertain
            # aliases to the reader via source rescue, not a claimed answer plan.
            shared_literal=bool(values) and bool(set.intersection(*values))
            projection=projection and shared_literal
        if c.query.answer_var=='t':
            event_ids=self.event_fact_ids(c,w)
            words=set().union(*(self._fact_terms[i] for i in event_ids)) if event_ids else set()
            target=len(self.reading.predicate_terms & words)/max(1,len(self.reading.predicate_terms))
            event_nodes={self.node(eid) for i in event_ids for eid in
                         (self.memory.facts[i].subj_id,self.memory.facts[i].obj_id)}
            # A temporal bridge to the requested entity must not substitute the
            # timestamp of an unrelated event for that entity's requested event.
            projection = projection and all(bool(nodes & event_nodes) for _,nodes in self.mentions)
            anchor=self.contract.period.interval
            for i in event_ids:
                value=self.dated.fact_interval[i]
                if not value or not anchor or self.dated.fact_time_source[i]!='expressao':continue
                side=self.contract.temporal_side
                conflict=(value.end>=anchor.start if side=='before' else
                          value.start<=anchor.end if side=='after' else
                          (value.end<anchor.start or value.start>anchor.end) if side=='within' else False)
                # Keeping an ongoing/said fact as uncertain is valid; projecting
                # its known out-of-scope timestamp as the requested date is not.
                if conflict:projection=False
        return {"owner_coverage":owners,"predicate_coverage":target,
                "projection_compatible":projection,"required_conjunction":required,
                "subject_scope_supported":scope_supported,"shared_value_literal_support":shared_literal}

    def event_fact_ids(self,c,w):
        return self.bound_fact_ids(c.query.atoms[0],w)

    def bound_fact_ids(self,atom,w):
        result=[]
        for i in w.facts:
            f=self.memory.facts[i]
            if canonical_symbol(f.relation)!=canonical_symbol(atom.relation):continue
            if all(self.node(eid) in self.entity_nodes.get(canonical_symbol(
                    w.bindings.get(term[1:],'') if is_var(term) else term),set())
                   for term,eid in ((atom.subject,f.subj_id),(atom.object,f.obj_id))):
                result.append(i)
        return result

    def score_cheap(self,c):
        best=-1
        for w in c.witnesses:
            check=self.checks(c,w)
            ids=list(w.facts)
            sem=max(0.,float(np.mean(self.sim[ids])))
            fact_words=set().union(*(self._fact_terms[i] for i in ids))
            coverage=len(stems(self.text)&fact_words)/max(1,len(stems(self.text)))
            structure=.45*check['owner_coverage']+.4*check['predicate_coverage']+.15*check['projection_compatible']
            value=.4*sem+.45*structure+.15*coverage
            if value>best:
                best,c.coverage,c.temporal=value,coverage,self.temporal_score(c,w)
        c.score=best

    def candidate(self,atoms):
        # V1 repeats many equivalent permutations. Sorting intersection atoms
        # gives shared memo entries without changing the answer projection.
        atoms=list(atoms)
        if len(atoms)>1 and all(any(t=="?x" for t in (a.subject,a.object)) for a in atoms):
            atoms.sort(key=lambda a:(a.subject,a.relation,a.object))
        if self.contract.operation=='date' or self.reading.count_unit=='occurrence':
            # Put the event asked about before contextual predicates; its date
            # is the projection and its timestamp receives the temporal scope.
            atoms.sort(key=lambda a:(-self.relation_intent(a.relation),a.subject,a.relation))
            atoms=[replace(a,time='?t' if i==0 and self.contract.operation=='date' else '') for i,a in enumerate(atoms)]
        c=super().candidate(atoms)
        if c:
            c.query.types={'x':self.reading.head} if self.reading.head else {}
            self._all.setdefault(c.signature(),c)
        return c

    def seed_proposals(self):
        by_origin=[]
        for anchor in self.anchors:
            facts=sorted(set(self.incident.get(anchor,[])),key=lambda i:(
                -self.relation_intent(self.memory.facts[i].relation),-float(self.sim[i]),i))
            proposals=[];seen=set()
            # Reranking width and generation branching have independent limits.
            # A wider reranker must not consume all search before intersections.
            for i in facts[:max(16,self.cfg.local_plan_beam)]:
                f=self.memory.facts[i]
                atom=Atom(f.relation,f.subject,'?x') if self.node(f.subj_id)==anchor else Atom(f.relation,'?x',f.object)
                key=(atom.subject,atom.relation,atom.object)
                if key not in seen:
                    seen.add(key);proposals.append([atom])
            by_origin.append(iter(proposals))
        return self.round_robin(by_origin)

    @staticmethod
    def round_robin(iterators):
        queue=deque(iterators)
        while queue:
            it=queue.popleft()
            try:
                yield next(it)
                queue.append(it)
            except StopIteration:
                pass

    def intersection_proposals(self,seeds):
        answer_index=defaultdict(list)
        for c in seeds:
            nodes={n for w in c.witnesses for n in self.entity_nodes.get(canonical_symbol(w.bindings.get('x','')),set())}
            for n in nodes:
                answer_index[n].append(c)
        pairs={}
        for candidates in answer_index.values():
            for a,b in itertools.combinations(candidates,2):
                if set(a.query.constants())==set(b.query.constants()):continue
                key=tuple(sorted((a.signature(),b.signature())))
                atoms=a.query.atoms+b.query.atoms
                owners={n for atom in atoms for v in (atom.subject,atom.object) if not is_var(v)
                        for n in self.entity_nodes.get(canonical_symbol(v),set())}
                coverage=sum(bool(nodes&owners) for _,nodes in self.mentions)
                priority=(coverage,sum(self.relation_intent(atom.relation) for atom in atoms),a.score+b.score)
                pairs[key]=(priority,atoms)
        for _,atoms in sorted(pairs.values(),key=lambda row:row[0],reverse=True):
            yield atoms

    def expansions(self,c):
        terminal='?x';intermediate=f'?y{len(c.query.atoms)}'
        prefix=[replace(a,subject=intermediate if a.subject==terminal else a.subject,
                        object=intermediate if a.object==terminal else a.object,time='') for a in c.query.atoms]
        bound={n for w in c.witnesses for n in self.entity_nodes.get(canonical_symbol(w.bindings.get('x','')),set())}
        incident={i for n in bound for i in self.incident.get(n,[])}
        seen=set()
        for i in sorted(incident,key=lambda i:(-self.relation_intent(self.memory.facts[i].relation),-float(self.sim[i]),i))[:max(24,self.cfg.local_plan_beam)]:
            f=self.memory.facts[i]
            next_atoms=[]
            if self.node(f.subj_id) in bound:next_atoms.append(Atom(f.relation,intermediate,'?x'))
            if self.node(f.obj_id) in bound:next_atoms.append(Atom(f.relation,'?x',intermediate))
            for atom in next_atoms:
                key=(atom.subject,atom.relation,atom.object)
                if key not in seen:
                    seen.add(key);yield prefix+[atom]
            # Preserve the requested value while attaching a second owner or
            # qualifier, instead of always moving the answer to the next node.
            if self.node(f.subj_id) in bound and self.node(f.obj_id) in self.owner_nodes:
                yield c.query.atoms+[Atom(f.relation,'?x',f.object)]
            if self.node(f.obj_id) in bound and self.node(f.subj_id) in self.owner_nodes:
                yield c.query.atoms+[Atom(f.relation,f.subject,'?x')]

    def consume(self,proposals,quota,phase):
        start=self.executions;out=[]
        for atoms in proposals:
            if self.executions-start>=quota or self.executions>=self.generation_budget:
                self.truncations.append('phase_budget:'+phase);break
            c=self.candidate(atoms)
            if c:
                c.witnesses=[w for w in c.witnesses if len(w.facts)==len(c.query.atoms)]
                if c.witnesses:out.append(c)
        self.phase_counts[phase]+=self.executions-start
        return out

    def frontier(self,candidates):
        groups=defaultdict(list)
        for c in candidates:
            constants=tuple(sorted(c.query.constants()))
            groups[(constants,c.query.shape())].append(c)
        # Reserve representation of different origins and program shapes.
        iterators=[iter(sorted(values,key=lambda c:(-c.score,c.signature()))) for _,values in sorted(groups.items())]
        return list(itertools.islice(self.round_robin(iterators),self.cfg.local_plan_beam*max(1,len(self.mentions))))

    def build(self):
        start=time.perf_counter();budget=self.generation_budget
        seeds=self.consume(self.seed_proposals(),max(1,int(.2*budget)),'seeds')
        # Intersections get their own reservation BEFORE broad walks. Increasing
        # branching cannot starve the conjunction phase anymore.
        intersections=self.consume(self.intersection_proposals(seeds),max(1,int(.3*budget)),'intersections')
        frontier=self.frontier(seeds+intersections)
        depth=min(self.cfg.max_atoms,self.cfg.local_plan_depth)
        for level in range(1,depth):
            eligible=[c for c in frontier if len(c.query.atoms)==level]
            quota=max(1,(budget-self.executions)//max(1,depth-level))
            extensions=self.consume(self.round_robin([iter(self.expansions(c)) for c in eligible]),quota,'depth_'+str(level+1))
            frontier=self.frontier(extensions+[c for c in intersections if len(c.query.atoms)==level+1])
        self.generated_total=len(self._all)
        # Invalid projections can still be useful prefixes. They are not
        # prioritized as complete answer programs when compatible ones exist.
        all_candidates=[c for c in self._all.values() if c.witnesses]
        compatible=[c for c in all_candidates if any(self.checks(c,w)['projection_compatible'] and self.checks(c,w)['required_conjunction'] for w in c.witnesses)]
        pool=self.frontier(compatible or all_candidates)
        remaining=sorted(compatible or all_candidates,key=lambda c:(-c.score,c.signature()))
        chosen={c.signature():c for c in pool[:self.cfg.local_plan_candidates]}
        for c in remaining:
            if len(chosen)>=self.cfg.local_plan_candidates:break
            chosen.setdefault(c.signature(),c)
        if len(all_candidates)>len(chosen):self.truncations.append('program_candidates')
        self.timings['build_seconds']=round(time.perf_counter()-start,4)
        return list(chosen.values())

    def rerank(self,candidates):
        start=time.perf_counter()
        # Validate bindings BEFORE a witness limit. File-order placeholders may
        # otherwise hide the specific value of a correctly generated program.
        queues=[]
        for c in candidates:
            valid=[w for w in c.witnesses if self.checks(c,w)['projection_compatible']
                   and self.checks(c,w)['required_conjunction']]
            valid.sort(key=lambda w:(-float(np.mean(self.sim[list(w.facts)])),w.facts))
            # Materialize the current program: a generator expression would
            # capture the loop variable c and relabel every queue as the last
            # program when the round robin is consumed later.
            queues.append(iter([(c,w) for w in valid]))
        pairs=list(itertools.islice(self.round_robin(queues),self.cfg.local_plan_candidates*3))
        if self.cfg.fact_rerank and pairs:
            from wrag.witness.rerank import get_reranker
            texts=[]
            for c,w in pairs:
                program=' AND '.join(f'{a.relation}({a.subject}, {a.object})' for a in c.query.atoms)
                evidence='\n'.join(self.memory.facts[i].verbalize() for i in w.facts)
                source='\n'.join(self.dated.excerpt(i)[:360] for i in w.facts)
                texts.append(f'Directed retrieval program: {program}\nEvidence:\n{evidence}\nSource:\n{source}')
            semantic=get_reranker(self.cfg.fact_rerank).score(self.search_text,texts)
            scorer='cross_encoder_relevance_experimental'
        else:
            semantic=[max(0.,float(np.mean(self.sim[list(w.facts)]))) for _,w in pairs]
            scorer='bi_encoder_fallback'
        results=[]
        for (c,w),sem in zip(pairs,semantic):
            temp=self.temporal_score(c,w)
            check=self.checks(c,w)
            structural=.5*check['owner_coverage']+.5*check['predicate_coverage']
            grade=int(check['projection_compatible'])+int(check['required_conjunction'])
            weight=self.contract.time_weight
            score=grade+(1-weight)*(.6*sem+.4*structural)+weight*temp
            results.append((score,c,w,sem,temp))
        results.sort(key=lambda row:(-row[0],row[1].signature(),row[2].facts))
        self.timings['rerank_seconds']=round(time.perf_counter()-start,4)
        return results,scorer

    def occurrence_key(self,fid):
        fact=self.memory.facts[fid]
        value=self.dated.fact_interval[fid]
        # Exact event dates may unite duplicate predicates across turns. Broad
        # intervals and session timestamps cannot establish event identity.
        if self.dated.fact_time_source[fid]=='expressao' and value and value.start==value.end:
            return ('explicit_event',self.node(fact.subj_id),self.node(fact.obj_id),str(value.start))
        return ('source_mention',*self.dated.fact_turn[fid])


def refresh_delivery(r,info,indices,acquired):
    """Delivery record after the reflection replaced filler facts."""
    dated,memory=r.dated,r.memory
    info=dict(info)
    info.update(indices=list(indices),n=len(indices),adquiridos=list(acquired),fontes=[
        {"indice":i,"fid":memory.facts[i].fid,"pid":memory.facts[i].pid,
         "turn_id":(dated.turns[pid][pos].turn_id if 0<=pos<len(dated.turns.get(pid,[])) else "")}
        for i in indices for pid,pos in [dated.fact_turn[i]]])
    return info


def complete_requirements(r,question,info,protected):
    """Planner requirements -> executor support check -> reflection completion.

    Returns (re-rendered facts or None, delivery info, acquired facts,
    diagnostics, member-table text)."""
    cfg=r.ctx.run.witness
    if not cfg.requirements:
        return None,info,[],None,''
    from wrag.witness.requirements import check_and_complete
    indices=list(info.get('indices',[]))
    new,acquired,diag=check_and_complete(r,question.question,indices,protected,
                                         acquire=cfg.study_ablation!='no-reflection')
    table=''
    scans=diag.pop('_scan_objects',[]) if diag else []
    if scans:
        from wrag.witness.member_scan import render_table
        table=render_table(scans,cfg.member_scan_max)
    if not acquired:
        return None,info,[],diag,table
    return r._render_fact_ids(new),refresh_delivery(r,info,new,acquired),acquired,diag,table


def reference_diagnostics(planner,times):
    if planner.reference is None:
        return None
    from wrag.witness.timeline import format_interval
    anchor=None
    if planner.anchor_resolution:
        anchor={key:(format_interval(value) if key=='interval' else value)
                for key,value in planner.anchor_resolution.items()}
    period=planner.contract.period
    return {'reference':planner.reference.to_dict(),'applied':list(planner.reference_applied),
            'anchor':anchor,'candidate_times':times,'operation':planner.contract.operation,
            'period':period.to_dict(),'side':planner.contract.temporal_side,
            'time_weight':planner.contract.time_weight,
            'now':planner.dated.last.isoformat() if planner.dated.last else ''}


def retrieve_local_v2(r,question,k,pool_pids,pool_scores,planner_class=MultiOriginPlanner):
    cfg=r.ctx.run.witness
    if cfg.study_ablation == 'no-witness':
        # The literal requested baseline: individual relevance-ranked facts,
        # no conjunctive compiler, graph search, source rescue or plan hints.
        facts,summary,info=r._fact_context(question,None,None,None,pool_pids[:k])
        assert not summary
        rendered,info,acquired,requirements,table=complete_requirements(r,question,info,set())
        if rendered is not None:facts=rendered
        if table:facts+='\n\n'+table
        reference=None
        if cfg.temporal_reference:
            from wrag.witness.local_plans import contract as temporal_contract
            from wrag.witness.temporal_reference import asks_time,plan_reference
            reference=plan_reference(r,question.question)
        return RetrievalResult(pids=pool_pids[:k],scores=pool_scores[:k],diagnostics={
            'controlador':'local-v2-ablation','study_ablation':'no-witness',
            'planejamento':{'chamadas':0,'chamadas_plano':0,'chamadas_verificacao':0,
                            'replanejamentos':0,'planos_distintos':0},
            'local_plans':{'version':'v2','generated':0,'executed':0,'selected':[]},
            'requisitos':requirements,
            'fatos_entregues':info,'trechos_extras':[{'title':'Retrieved memory facts','text':facts}],
            'temporal_reference':({'reference':reference.to_dict(),
                                   'applied':(['date_question'] if asks_time(reference,question.question,
                                              temporal_contract(question.question,r.dated).operation) else []),
                                   'anchor':None,'candidate_times':[],'operation':'',
                                   'now':r.dated.last.isoformat() if r.dated.last else ''} if reference else None),
            'leitura_fatos':'bitemporal' if cfg.fact_time=='both' else True})
    if cfg.fact_delivery!='facts' or cfg.summary_reflection or cfg.plan_router or cfg.multiplan_portfolio:
        raise ValueError('Local plans require facts only, no summaries or LLM router')
    if min(cfg.local_plan_beam,cfg.local_plan_candidates,cfg.local_plan_depth,cfg.local_plan_keep,
           cfg.local_plan_executions,cfg.local_plan_starts)<1:
        raise ValueError('Local plan budgets must be positive')
    planner=planner_class(r,question);candidates=planner.build();ranked,scorer=planner.rerank(candidates)
    selected=[];packages=[];seen_queries=set();covered=set();groups=[];selected_events=[]
    for item in ranked:
        score,c,w,_,_=item
        if c.signature() in seen_queries:continue
        check=planner.checks(c,w)
        if not check['projection_compatible'] or not check['required_conjunction']:continue
        witnesses=[w]
        if planner.contract.operation in {'set','count'}:
            witnesses=[v for v in c.witnesses if planner.checks(c,v)['projection_compatible']]
        ids=list(dict.fromkeys(i for v in witnesses for i in v.facts))
        if set(ids)<=covered:continue
        event_ids=set(planner.event_fact_ids(c,w))
        if planner.contract.operation not in {'set','count'} and any(
                w.answer==old[2].answer and event_ids & old_ids for old,old_ids in selected_events):continue
        if planner.contract.reference_fact is not None:ids.append(planner.contract.reference_fact)
        ids=list(dict.fromkeys(ids))
        # A set is a bundle of complete witnesses, never a sliced join. Admit
        # as many complete members as fit while reporting incomplete retrieval.
        if len(covered|set(ids))>cfg.fact_budget:
            ids=list(w.facts)
            if planner.contract.reference_fact is not None:ids.append(planner.contract.reference_fact)
        seen_queries.add(c.signature());selected.append(item);packages.append(ids);covered.update(ids)
        selected_events.append((item,event_ids))
        if len(selected)>=cfg.local_plan_keep:break
    accepted={'selecionadas':[(ProofPlan(query=c.query,valid=True),w) for _,c,w,_,_ in selected], 'pacotes':packages}
    facts,summary,info=r._fact_context(question,None,accepted,None,pool_pids[:k])
    assert not summary,'Local retrieval unexpectedly generated a summary'
    # Proof facts (packages of the selected witnesses) are never displaced.
    before=set(info.get('indices',[]))
    protected={i for ids in packages if set(ids)<=before for i in ids}|set(info.get('indices',[])[:info.get('prova',0)])
    rendered,info,acquired,requirements,table=complete_requirements(r,question,info,protected)
    if rendered is not None:facts=rendered
    delivered=set(info.get('indices',[]))
    retained=[item for item,ids in zip(selected,packages) if set(ids)<=delivered]
    source_ids=[i for ids in packages if set(ids)<=delivered for i in ids]
    # Facts acquired by the reflection bring their original turns.
    source_ids += [i for i in acquired if i not in source_ids]
    source_ids += [i for i in info.get('indices',[]) if i not in source_ids][:4]
    excerpts,turn_ids=r._dialogue_block([(i,) for i in dict.fromkeys(source_ids)],0)
    if not hasattr(r,'_local_source_index') or r._local_source_index_identity!=(id(r.dated),len(r.memory.facts)):
        r._local_source_index=SourceIndex(r.dated)
        r._local_source_index_identity=(id(r.dated),len(r.memory.facts))
    supports=r._local_source_index.support(planner.search_text,planner.reading)
    # Original-turn rescue does not assert any new memory fact or resolved
    # reference. Keep an independent, bounded literal source budget.
    support_lines=[];used_chars=0;seen_turns=set(turn_ids)
    for row in supports:
        key=row['turn_id'] or (row['pid'],row['position'])
        if key in seen_turns:continue
        seen_turns.add(key)
        body=row['text'][:650]
        if len(body)<len(row['text']):body+=' [excerpt truncated]'
        date_label=f" ({row['when']})" if row['when'] else ''
        line=f"[{row['turn_id'] or row['pid']}] {row['speaker']}{date_label}: {body}"
        if used_chars+len(line)>cfg.excerpt_max_chars:continue
        support_lines.append(line);used_chars+=len(line)
    if planner.reading.count_unit=='occurrence':
        grouped=defaultdict(list)
        event_ids={i for _,c,_,_,_ in retained for w in c.witnesses
                   for i in planner.event_fact_ids(c,w) if i in delivered}
        for fid in sorted(event_ids):grouped[planner.occurrence_key(fid)].append(fid)
        groups=[{'identity':list(key),'facts':ids,'status':'mention_group_not_verified_occurrence'} for key,ids in grouped.items()]
    from wrag.witness.temporal_reference import candidate_times
    reference=planner.reference
    # Projected event times serve the reflection's date check only. Shown to
    # the reader they helped on conv03 (+4.7, 20 questions) and hurt on conv07
    # (-22, 14 questions): the reader's context stays exactly the v2 context.
    times=(candidate_times(planner,retained)
           if reference is not None and 'date_question' in planner.reference_applied else [])
    head=planner.reading.instructions()
    text=head+'\n\nRetrieved facts (joins are unverified candidates):\n'+facts
    if excerpts:text+='\n\nOriginal source turns:\n'+excerpts
    if support_lines:text+='\n\nAdditional original turns (candidate support, not inferred facts):\n'+'\n'.join(support_lines)
    if groups:text+='\n\nSource mention groups for counting (not a certified count):\n'+'\n'.join(str(g['identity'])+' facts '+str(g['facts']) for g in groups)
    if table:text+='\n\n'+table
    text+='\n\nPending checks: '+'; '.join(planner.contract.pending+['question_semantics_and_qualifiers_require_reader_check'])
    def describe(item):
        score,c,w,sem,temp=item
        query_data=c.query.to_dict();query_data['uses_annotations']=False
        package=next((ids for old,ids in zip(selected,packages) if old[1].signature()==c.signature()),list(w.facts))
        return {'query':query_data,'facts':list(w.facts),'package_facts':package,'answer_binding':w.answer,
                'enumeration_complete':False,
                'score':round(score,6),'semantic':round(sem,6),'temporal':round(temp,6),
                'contract_checks':planner.checks(c,w),'verified':False}
    diagnostics={'controlador':'local-multiplan-v2','rota':'local','classe_prova':'candidata_nao_verificada',
                 'study_ablation':cfg.study_ablation or 'full',
                 'motivo_parada':'planos_locais_entregues' if retained else 'recuperacao_local_sem_pacote',
                 'planejamento':{'chamadas':0,'chamadas_plano':0,'chamadas_verificacao':0,'replanejamentos':0,'planos_distintos':len(candidates)},
                 'local_plans':{'version':'v2','contract':planner.reading.to_dict(),'anchors':planner.anchors,
                    'generated':planner.generated_total,'executed':planner.executions,'reranked_plans':len(candidates),
                    'ranked_pairs':len(ranked),'scorer':scorer,'truncations':sorted(set(planner.truncations)),
                    'temporal_rejections':planner.rejected_temporal,'phase_executions':dict(planner.phase_counts),
                    'timings':planner.timings,'selected':[describe(item) for item in retained],
                    'top_candidates':[describe(item) for item in ranked[:8]],'source_turns':turn_ids,
                    'additional_source_turns':[{k:v for k,v in row.items() if k!='text'} for row in supports],
                    'count_groups':groups},
                 'fatos_entregues':info,'trechos_extras':[{'title':'Local logical retrieval','text':text}],
                 'temporal_reference':reference_diagnostics(planner,times),
                 'requisitos':requirements,
                 'leitura_fatos':('atemporal' if cfg.study_ablation=='no-time-model' else
                                 'bitemporal' if cfg.fact_time=='both' else True)}
    return RetrievalResult(pids=pool_pids[:k],scores=pool_scores[:k],diagnostics=diagnostics)
