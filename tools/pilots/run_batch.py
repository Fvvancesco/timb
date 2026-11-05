"""Run a bounded list of isolated jobs; preserve errors and resume only identical inputs."""
import argparse, concurrent.futures, hashlib, json, os, pathlib, signal, subprocess, time
from result_helpers import known_pilot_censor
P = pathlib.Path(os.environ.get('ICON_PILOT_DIR', '/home/francesco/ICON-experiments/pilots')).resolve()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def cleanup_owned(out):
    registry=out/'owned-processes.json'
    if not registry.exists(): return
    record=json.loads(registry.read_text())
    for pid in [record.get('server'),*record.get('clients',[])]:
        if not pid: continue
        try:
            proc=pathlib.Path('/proc')/str(pid)
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            cwd=str((proc/'cwd').resolve())
            ours=(pid==record.get('server') and cwd==record['server_cwd'] and 'pokemon-showdown' in cmd) or (pid in record.get('clients',[]) and record['wrapper'] in cmd)
            if ours and os.getpgid(pid)==pid: os.killpg(pid,signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError, PermissionError): pass

def cache_hashes():
    base=P/'foul-play/fp/data'
    files=sorted([p for d in ('smogon_stats_cache','pkmn_sets_cache') for p in (base/d).rglob('*') if p.is_file()])
    return {str(p.relative_to(base)):sha(p) for p in files}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('jobs',nargs='+'); ap.add_argument('--workers',type=int,choices=range(1, 15),required=True); ap.add_argument('--output',required=True); ap.add_argument('--allow-known-pilot-censor',action='store_true'); ap.add_argument('--stop-file',type=pathlib.Path); args=ap.parse_args()
    root=pathlib.Path(args.output).resolve(); root.mkdir(parents=True,exist_ok=True)
    jobs=[pathlib.Path(x).resolve() for x in args.jobs]; specs=[json.loads(x.read_text()) for x in jobs]
    if len({s['job_id'] for s in specs})!=len(specs): raise ValueError('Duplicate jobs')
    code=[P/'seeded_worker.py',P/'result_helpers.py',P/'foul_packed_wrapper.py',P/'replay_check.cjs']
    config={'workers':args.workers,'jobs':{s['job_id']:sha(p) for s,p in zip(specs,jobs)},'code':{p.name:sha(p) for p in code},'server_hashes':{str(i):sha(P/f'worker-server-{i}/icon-patch-hashes.json') for i in range(args.workers)}, 'inference_cache':cache_hashes()}
    if args.allow_known_pilot_censor:
        config['allow_known_pilot_censor']=True
    if args.stop_file:
        config['collector_code'] = {p.name: sha(p) for p in
                                    (pathlib.Path(__file__), pathlib.Path(__file__).with_name('result_helpers.py'))}
    for i in range(args.workers):
        hashes=json.loads((P/f'worker-server-{i}/icon-patch-hashes.json').read_text())
        if any(sha(pathlib.Path(path))!=expected for path,expected in hashes.items()): raise RuntimeError('Server patch hash mismatch')
    meta=root/'batch-config.json'
    if meta.exists() and json.loads(meta.read_text())!=config: raise RuntimeError('Resume rejected: batch inputs or code differ')
    if not meta.exists(): meta.write_text(json.dumps(config,indent=2)+'\n')
    start=time.monotonic(); pending=list(zip(jobs,specs)); active={}; results=[]; failed=False
    def stopped():
        return args.stop_file is not None and args.stop_file.exists()

    def publish():
        summary={'workers':args.workers,'elapsed_seconds':round(time.monotonic()-start,3),
                 'attempted':len(results),'unscheduled':len(pending),'in_flight':len(active),
                 'stop_requested':stopped(),'results':results}
        temp=root/'batch-summary.json.tmp'
        temp.write_text(json.dumps(summary,indent=2)+'\n')
        temp.replace(root/'batch-summary.json')

    def censor(out, result):
        if not args.allow_known_pilot_censor or result.get('status') != 'error':
            return None
        logs = (path.read_text(errors='replace') for path in sorted(out.glob('client-*.log')))
        reason = known_pilot_censor(result, logs)
        if reason:
            return {'job_id': result['job_id'], 'status': 'censored',
                    'censor_reason': reason, 'replay_match': False}
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        def one(slot,job,spec):
            out=root/spec['job_id']; result=out/'result.json'
            if result.exists():
                saved=json.loads((out/'job.json').read_text()); previous=json.loads(result.read_text())
                if saved != spec: raise RuntimeError('Resume rejected: job input differs')
                resumed = censor(out, previous) or {'job_id':spec['job_id'],'status':previous['status'],'replay_match':json.loads((out/'replay-check.json').read_text())['match'] if (out/'replay-check.json').exists() else False}
                return {**resumed, 'resumed': True}
            out.mkdir(parents=True,exist_ok=True)
            with (out/'worker.log').open('w') as log:
                p=subprocess.Popen([str(P/'poke-venv/bin/python'),str(P/'seeded_worker.py'),str(job),str(out),'--server',str(P/f'worker-server-{slot}'),'--port',str(18766+slot)],stdout=log,stderr=subprocess.STDOUT)
                try: p.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try: p.wait(timeout=15)
                    except subprocess.TimeoutExpired: cleanup_owned(out); p.kill(); p.wait()
                    cleanup_owned(out)
                    return {'job_id':spec['job_id'],'status':'error','error':'worker timeout','replay_match':False}
            if cache_hashes()!=config['inference_cache']: return {'job_id':spec['job_id'],'status':'error','error':'inference cache changed','replay_match':False}
            if p.returncode != 0:
                recognized = censor(out, json.loads(result.read_text())) if result.exists() else None
                return recognized or {'job_id':spec['job_id'],'status':'error','exit_code':p.returncode,'replay_match':False}
            if not result.exists(): return {'job_id':spec['job_id'],'status':'error','exit_code':p.returncode,'replay_match':False}
            r=json.loads(result.read_text()); records=list((out/'engine').glob('*.json'))
            replay=False
            if records and (out/'public.log').exists():
                p=subprocess.run(['node',str(P/'replay_check.cjs'),str(P/f'worker-server-{slot}'),str(records[0]),str(out/'public.log'),str(out/'replay-check.json')],env={**__import__('os').environ,'NODE_PATH':str(P/'../project/tools/showdown/node_modules')},capture_output=True,text=True,timeout=25)
                (out/'replay-process.log').write_text(p.stdout+p.stderr); replay=p.returncode==0
            return {'job_id':spec['job_id'],'status':r['status'],'replay_match':replay,'wall_seconds':r['game_seconds_approx']}
        available=list(range(args.workers))
        while pending or active:
            while pending and available and not failed and not stopped():
                slot=available.pop(0); job,spec=pending.pop(0); active[pool.submit(one,slot,job,spec)]=slot
            if not active: break
            done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                slot=active.pop(future); available.append(slot)
                try: r=future.result()
                except Exception as e: r={'status':'error','error':repr(e),'replay_match':False}
                results.append(r); print(json.dumps(r),flush=True)
                if r['status'] != 'censored' and (r['status']=='error' or not r['replay_match']):
                    failed=True
            publish()
    publish()
    if failed: raise SystemExit('Stopped after incompatibility; inspect preserved records')
if __name__=='__main__': main()
