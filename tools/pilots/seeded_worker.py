"""One isolated seeded self-play job; inputs are JSON, artifacts written atomically."""
import argparse, asyncio, hashlib, json, os, pathlib, random, signal, socket, subprocess, sys, time
import websockets
from result_helpers import classify_engine_result, team_differences
P = pathlib.Path(os.environ.get('ICON_PILOT_DIR', '/home/francesco/ICON-experiments/pilots')).resolve()
def atomic(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)+'\n'); os.replace(tmp,path)
def stop(p):
    if p and p.poll() is None:
        try: os.killpg(p.pid,signal.SIGTERM); p.wait(timeout=4)
        except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL); p.wait()
        except ProcessLookupError: pass
async def get_public(room,port):
    async with websockets.connect(f'ws://127.0.0.1:{port}/showdown/websocket') as ws:
        await ws.recv(); await ws.send(f'|/join {room}')
        for _ in range(8):
            msg=await asyncio.wait_for(ws.recv(),3)
            if msg.startswith('>'+room+'\n|init|battle'): return msg
    raise RuntimeError('Missing public room log')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('output'); ap.add_argument('--server',default=str(P/'seeded-server')); ap.add_argument('--port',type=int,default=18766); args=ap.parse_args()
    spec=json.loads(pathlib.Path(args.spec).read_text()); out=pathlib.Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    final=out/'result.json'
    if final.exists(): raise SystemExit('Result already exists; refusing overwrite')
    server=pathlib.Path(args.server).resolve(); cfg=(server/'config/config.js').read_text()
    assert "exports.bindaddress = '127.0.0.1'" in cfg and 'exports.nothrottle = true' in cfg
    assert spec['format']=='gen9ou' and len(spec['players'])==2 and len(spec['team_files'])==2
    assert 1 <= spec['max_turns'] <= 200 and spec['wall_limit_seconds'] == 120
    assert spec['players'][0].lower()!=spec['players'][1].lower()
    seed=spec['engine_seed']; assert isinstance(seed,str) and seed.startswith('gen5,')
    assert f'exports.port = {args.port};' in cfg and 18766 <= args.port <= 18779
    atomic(out/'job.json',spec)
    for i,team in enumerate(spec['team_files']):
        raw=pathlib.Path(team).read_bytes()
        if 'team_text_sha256' in spec and hashlib.sha256(raw).hexdigest()!=spec['team_text_sha256'][i]: raise ValueError('Team source hash mismatch')
        (out/f'team-{i}.txt').write_text(raw.decode().strip())
        (out/f'team-{i}.packed').write_text(spec['packed_teams'][i])
    os.makedirs(out/'engine',exist_ok=True)
    env=os.environ.copy(); env.update(ICON_JOB_SPEC=str(out/'job.json'),ICON_OUTPUT_DIR=str(out/'engine'),NODE_PATH=str(P/'../project/tools/showdown/node_modules'))
    clients=[]; serverproc=None; handles=[]; t0=time.monotonic(); error=None; data=None
    registry=out/'owned-processes.json'
    def register(): atomic(registry, {'server':serverproc.pid if serverproc else None, 'clients':[p.pid for p in clients], 'server_cwd':str(server), 'wrapper':str(P/'foul_packed_wrapper.py')})
    def terminate_requested(signum, frame): raise RuntimeError('Worker termination requested')
    signal.signal(signal.SIGTERM, terminate_requested)
    try:
        serverlog=(out/'server.log').open('w'); handles.append(serverlog)
        serverproc=subprocess.Popen(['node','pokemon-showdown','start','--skip-build',str(args.port)],cwd=server,env=env,stdout=serverlog,stderr=subprocess.STDOUT,start_new_session=True)
        register()
        deadline=time.monotonic()+25
        while time.monotonic()<deadline:
            if serverproc.poll() is not None: raise RuntimeError('Server exited during startup')
            try:
                with socket.create_connection(('127.0.0.1',args.port),timeout=.2): break
            except OSError: time.sleep(.1)
        else: raise TimeoutError('Server startup')
        # Per-process Python RNG is seeded before imports; engine search RNG remains uncontrolled.
        site=out/'python-seeding'; site.mkdir(exist_ok=True)
        (site/'sitecustomize.py').write_text('import os, random\nrandom.seed(int(os.environ["ICON_PYTHON_SEED"]))\n')
        for side in (1,0):
            log=(out/f'client-{side}.log').open('w'); handles.append(log)
            clientenv=os.environ.copy(); clientenv.update(PYTHONPATH=str(P/'foul-play'),ICON_PYTHON_SEED=str(spec['python_seeds'][side]),ICON_PACKED_TEAM=str(out/f'team-{side}.packed'),ICON_PYTHON_STATE_FILE=str(out/f'python-state-{side}.sha256'))
            cmd=[str(P/'.venv/bin/python'),'-u',str(P/'foul_packed_wrapper.py'),'--websocket-uri',f'ws://127.0.0.1:{args.port}/showdown/websocket','--ps-username',spec['players'][side], '--bot-mode','accept_challenge' if side==1 else 'challenge_user','--pokemon-format','gen9ou','--search-time-ms','25','--search-parallelism','1','--team-preview-search-time-ms','25','--team-preview-search-parallelism','1','--search-threads','1','--run-count','1','--team-name',str(out/f'team-{side}.txt'),'--log-level','INFO']
            if side==0: cmd+=['--user-to-challenge',spec['players'][1]]
            clients.append(subprocess.Popen(cmd,cwd=P/'foul-play',env=clientenv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            register()
            if side==1:
                deadline=time.monotonic()+30
                while time.monotonic()<deadline:
                    if 'Waiting for a gen9ou challenge' in (out/'client-1.log').read_text(): break
                    if clients[-1].poll() is not None: raise RuntimeError('Receiver exited before ready')
                    time.sleep(.1)
                else: raise TimeoutError('Receiver did not become ready')
        started=time.monotonic(); deadline=started+130
        while time.monotonic()<deadline:
            files=list((out/'engine').glob('battle-gen9ou-*.json'))
            if files:
                data=json.loads(files[0].read_text()); room=files[0].stem
                (out/'public.log').write_text(asyncio.run(get_public(room,args.port)))
                break
            if any(p.poll() is not None for p in clients): raise RuntimeError('Client exited before engine record')
            if serverproc.poll() is not None: raise RuntimeError('Server exited during game')
            time.sleep(.1)
        else: raise TimeoutError('No engine record within 130 seconds')
        for side in (0,1):
            actual=(out/f'python-state-{side}.sha256').read_text().strip()
            expected=hashlib.sha256(repr(random.Random(spec['python_seeds'][side]).getstate()).encode()).hexdigest()
            if actual != expected: raise ValueError(f'Python RNG seed verification failed for side {side}')
        assert data['seed']==seed, f"Engine seed mismatch: {data['seed']} != {seed}"
        assert [data['p1'],data['p2']]==spec['players'], 'Engine side order mismatch'
        assert data['inputLog'] and data['inputLog'][0].startswith('>start ')
        for side in (0,1):
            diffs=team_differences(spec['expected_teams'][side], data[f'p{side+1}team'])
            if diffs: raise ValueError(f'Team {side+1} mismatch: {diffs}')
    except Exception as e: error=repr(e)
    finally:
        for p in clients: stop(p)
        stop(serverproc)
        for h in handles: h.close()
        if registry.exists(): registry.unlink()
    forced=data.get('forced_stop') if data else None
    status, outcome = classify_engine_result(data, error)
    result={'job_id':spec['job_id'],'status':status, 'outcome':outcome, 'forced_stop':forced,'requested_engine_seed':seed,'actual_engine_seed':data.get('seed') if data else None,'python_seeds':spec['python_seeds'],'python_seed_state_verified':bool(data and error is None),'rust_rng_controlled':False,'wall_seconds_total':round(time.monotonic()-t0,3),'game_seconds_approx':round(time.monotonic()-started,3) if 'started' in locals() else None,'current_request_turn':data.get('current_request_turn') if data else None,'engine_turn':data.get('turns') if data else None,'completed_turns': (data.get('current_request_turn', 0) - 1 if forced == 'turn_limit' else data.get('turns')) if data else None,'error':error,'server_config_sha256':hashlib.sha256(cfg.encode()).hexdigest()}
    atomic(final,result); print(json.dumps(result),flush=True)
    if error: raise SystemExit(1)
if __name__=='__main__': main()
