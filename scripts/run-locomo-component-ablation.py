#!/usr/bin/env python3
"""Sequential, resumable local-v2 LoCoMo component study against an existing vLLM."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec = importlib.util.spec_from_file_location('ablation_watch',ROOT/'scripts/watch-locomo-ablation.py')
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--gpu',default='0')
    p.add_argument('--port',type=int,default=8096)
    p.add_argument('--concurrency',type=int,default=2)
    p.add_argument('--hours',type=float,default=168)
    p.add_argument('--interval',type=float,default=60)
    p.add_argument('--dry-run',action='store_true')
    return p


def commands(args):
    common = [sys.executable,'-u','-m','wrag.pilot','--backend','vllm','--existing-server',
        '--port',str(args.port),'--model','Qwen/Qwen2.5-14B-Instruct',
        '--tokenizer-model','Qwen/Qwen2.5-14B-Instruct','--gpu',args.gpu,
        '--concurrency',str(args.concurrency),'--dataset','locomo','--locomo-conversation','all',
        '--methods','witnessrag','--embed-model','BAAI/bge-m3','--embed-device','cuda',
        '--locomo-chunk-tokens','2048','--locomo-ie-window-tokens','512','--top-k','5',
        '--qa-max-tokens','128','--witness-candidate-pool','20','--answer-set','--temporal-annotations',
        '--evidence-reader','--binding-aware-grounding','--vocab-compile','--hybrid-fallback',
        '--dialogue-ie','--gap-context-rescue','--proof-controller','--ie-style','memory',
        '--fact-delivery','facts','--fact-fill','question','--fact-time','both','--fact-budget','40',
        '--fact-rerank','cross-encoder/ms-marco-MiniLM-L6-v2','--local-plans','--reader-reflection',
        '--no-proof-verify','--local-plan-version','v2','--local-plan-beam','32',
        '--local-plan-candidates','96','--local-plan-executions','4000','--local-plan-starts','12',
        '--cache-dir',str(args.cache),'--hours',str(args.hours)]
    return {name:common+['--study-ablation',name,'--output',str(args.output/name)] for name in watch.VARIANTS}


def identity(args):
    from benchmarks.memoryagentbench.protocol import code_hash
    launch_hash = hashlib.sha256((ROOT/'scripts/run-locomo-component-ablation.py').read_bytes()+
                               (ROOT/'scripts/watch-locomo-ablation.py').read_bytes()).hexdigest()
    return {'schema':1,'code_hash':code_hash(ROOT),'launcher_hash':launch_hash,
            'model':'Qwen/Qwen2.5-14B-Instruct','gpu':args.gpu,'port':args.port,
            'concurrency':args.concurrency,'cache':str(args.cache),'fact_budget':40,
            'expected_questions':1540,'variants':list(watch.VARIANTS),'latency_study':False}


def save(path,value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def validated_complete(output):
    rows = watch.records(output)
    if watch.load(output/'status.json',{}).get('status') != 'complete' or len(rows) != 1540:
        return False
    conversations = watch.load(output/'conversations.json',{}).get('conversas',[])
    if len(conversations) != 10 or {int(c['conversa']) for c in conversations} != set(range(10)):
        raise ValueError('Complete state has inconsistent conversation coverage')
    for manifest in output.glob('conversations/conv*/benchmark/*/run.json'):
        if not manifest.is_file():
            return False
    return True


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid,signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        # Preserve PID: never allow a resumed suite to duplicate a live writer.
        pass


def main():
    args = parser().parse_args()
    args.output = args.output.resolve(); args.cache = args.cache.resolve()
    if args.gpu != '0' or args.concurrency not in (1,2) or args.interval < 5:
        raise ValueError('This shared-GPU study requires GPU 0, concurrency 1 or 2, interval >= 5')
    from wrag.pilot import make_plan, parser as pilot_parser
    jobs = commands(args)
    for name,command in jobs.items():
        make_plan(pilot_parser().parse_args(command[4:]),args.output/name)
    if args.dry_run:
        print(json.dumps(jobs,indent=2))
        return
    import fcntl
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'suite.lock').open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A study writer is already active')
        current_identity = identity(args)
        state = watch.load(args.output/'suite.json')
        if state and state['identity'] != current_identity:
            raise ValueError('Resume would mix configurations or code versions')
        state = state or {'identity':current_identity,'variants':{}}
        active = state.get('active_pid')
        if active:
            try:
                os.kill(active,0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f'Previous worker {active} is still running')
        state.update(status='running',launcher_pid=os.getpid(),active_pid=None)
        save(args.output/'suite.json',state)
        env = dict(os.environ)
        for key in list(env):
            if key.startswith('WRAG_'):
                del env[key]
        env.update(PYTHONUNBUFFERED='1',PYTHONHASHSEED='42',WRAG_EMBED_STRICT_DEVICE='1',
            WRAG_EMBED_MAX_SEQ_LENGTH='0',WRAG_EMBED_BATCH_SIZE='8',
            WRAG_LLM_CACHE='1',WRAG_EMBED_CACHE='1',WRAG_CONTINUE_ON_CONTENT_FILTER='1')
        process = None
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/v1/models',timeout=15) as response:
                models = json.load(response)
            if current_identity['model'] not in {row['id'] for row in models['data']}:
                raise ValueError('Unexpected vLLM model')
            for name,command in jobs.items():
                if validated_complete(args.output/name):
                    state['variants'][name]={'status':'complete'}
                    save(args.output/'suite.json',state)
                    continue
                if (args.output/name/'pilot.json').exists():
                    command = command+['--resume']
                state['variants'][name]={'status':'running'}
                save(args.output/'suite.json',state)
                print(f'\nINICIANDO: {watch.VARIANTS[name]} | 1540 questoes | GPU 0 | 40 fatos',flush=True)
                with (args.output/f'{name}.launcher.log').open('a',encoding='utf-8') as log:
                    process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    state['active_pid']=process.pid
                    save(args.output/'suite.json',state)
                    while process.poll() is None:
                        data = watch.write_snapshot(args.output)
                        text = watch.render(data)
                        print('\n'+text,flush=True)
                        with (args.output/'progress.log').open('a',encoding='utf-8') as panel:
                            panel.write(text+'\n\n')
                        try:
                            process.wait(timeout=args.interval)
                        except subprocess.TimeoutExpired:
                            pass
                state['active_pid']=None
                if process.returncode != 0 or not validated_complete(args.output/name):
                    state['variants'][name]={'status':'failed','exit_code':process.returncode}
                    raise RuntimeError(f'{name} stopped; preserve output and inspect {name}.launcher.log / benchmark.log')
                state['variants'][name]={'status':'complete'}
                save(args.output/'suite.json',state)
            state['status']='complete'
            # The strict final comparison uses exactly the same question IDs.
            all_rows = {name:watch.records(args.output/name) for name in watch.VARIANTS}
            reference = set(all_rows['full'])
            if any(set(rows) != reference for rows in all_rows.values()):
                raise ValueError('Completed variants do not contain the same question IDs')
        except BaseException as exc:
            stop_owned(process)
            state['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
            state['error']=str(exc)
            if process is None or process.poll() is not None:
                state['active_pid']=None
            save(args.output/'suite.json',state)
            watch.write_snapshot(args.output)
            raise
        save(args.output/'suite.json',state)
        print('\n'+watch.render(watch.write_snapshot(args.output)),flush=True)


if __name__ == '__main__':
    main()
