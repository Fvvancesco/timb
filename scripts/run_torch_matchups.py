"""Compare three CUDA architectures in the existing team-disjoint nested CV."""
import argparse
from collections import Counter
import fcntl
from functools import lru_cache
import json
import os
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

import run_expanded_search as base
from probe_complete_sets import explicit_sets, STATS
from icon_pokemon.adapter import sha256
from icon_pokemon.expanded_search import candidates
from icon_pokemon.supervised import metrics

ROOT = base.ROOT
OUT = ROOT / 'reports/torch_matchups_v1'
CONFIG = ROOT / 'config/torch_matchups_v1.json'


def prepare():
    base.CONFIG = CONFIG
    base.prepare(OUT)
    protocol = base.read(OUT / 'protocol.json')
    if protocol['counts']['decisive'] != 7470:
        raise ValueError('Expected 7470 decisive development games')
    print('Nested development inputs verified', flush=True)


def member_features(record):
    details = record['set']
    values = {f'species={record["species_id"]}': 1.0}
    for field in ('item', 'ability', 'nature', 'teraType', 'gender', 'hpType'):
        values[f'{field}={details.get(field) or "<absent>"}'] = 1.0
    values.update({f'move={move["id"]}': 1.0 for move in record['moves']})
    values.update({f'type={kind}': 1.0 for kind in record['types']})
    values['level'] = details.get('level', 100) / 100
    for stat in STATS:
        values[f'base/{stat}'] = record['base_stats'][stat] / 255
        values[f'ev/{stat}'] = (details.get('evs') or {}).get(stat, 0) / 252
        values[f'iv/{stat}'] = (details.get('ivs') or {}).get(stat, 31) / 31
    return values


class Data:
    def __init__(self):
        protocol = base.checked(OUT)
        reference = OUT / 'local_source_hashes.json'
        if reference.exists():
            base.verify(base.read(reference))
        self.config = protocol['config']
        self.digest = sha256(OUT / 'protocol.json')
        stored = joblib.load(OUT / 'development.joblib')
        self.x, self.y, self.plans = stored['x']['v1'], stored['y'], stored['plans']
        catalog = base.read(ROOT / 'data/processed/dataset_v1/catalog.json')
        self.teams = {k: t['set_ids'] for k, t in catalog['teams'].items()
                      if t['split'] in ('train', 'validation')}
        ids = {i for team in self.teams.values() for i in team}
        records = {r['id']: r for r in catalog['sets'] if r['id'] in ids}
        self.pairs = [(a, b) for a, b, _ in stored['examples']]
        self.members = {i: member_features(r) for i, r in records.items()}
        self.explicit = {k: {n: v for n, v in explicit_sets(ids, records).items()
                            if not n.startswith('set/')} for k, ids in self.teams.items()}

    @lru_cache(maxsize=3)
    def fold(self, plan_id, fold_id, arm, device):
        import torch

        plan = next(p for p in self.plans if p['id'] == plan_id)
        split = plan['split'] if fold_id == -1 else plan['inner'][fold_id]
        train, valid = split['train'], split['validation']
        fit_teams = sorted({k for i in train for k in self.pairs[i]})
        if set(fit_teams) & {k for i in valid for k in self.pairs[i]}:
            raise ValueError('Fold has overlapping teams')
        a, b = np.hsplit(self.x[train], 2)
        scaler = StandardScaler().fit(np.vstack((a, b)))
        vec = DictVectorizer(sparse=False, dtype=np.float32)
        lookup = None
        if arm == 'B':
            vec.fit([self.explicit[k] for k in fit_teams])
            lookup = dict(zip(self.teams, vec.transform([self.explicit[k] for k in self.teams]), strict=True))
        elif arm == 'C':
            vec.fit([self.members[i] for k in fit_teams for i in self.teams[k]])
            lookup = {k: vec.transform([self.members[i] for i in ids]) for k, ids in self.teams.items()}

        def tensor(array):
            return torch.as_tensor(np.asarray(array, dtype=np.float32), device=device)

        def materialize(indices):
            sides = np.stack([scaler.transform(half) for half in np.hsplit(self.x[indices], 2)], axis=1)
            extra = None
            if lookup is not None:
                extra = np.asarray([[lookup[k] for k in self.pairs[i]] for i in indices])
            if arm == 'B':
                sides = np.concatenate((sides, extra), axis=2)
            return tensor(sides), tensor(extra) if arm == 'C' else None, tensor(self.y[indices])

        return materialize(train), materialize(valid), valid


def fit(data, plan, fold_id, candidate, device, epochs, save_predictions=False):
    import torch
    from torch import nn

    arm, params = candidate['family'], candidate['params']
    (x, members, y), (vx, vm, vy), indices = data.fold(plan['id'], fold_id, arm, device)
    torch.manual_seed(plan['seed'])

    def mlp(dim, widths, dropout):
        layers = []
        for width in widths:
            layers.extend((nn.Linear(dim, width), nn.ReLU(), nn.Dropout(dropout)))
            dim = width
        return nn.Sequential(*layers, nn.Linear(dim, 1))

    class Matchup(nn.Module):
        def __init__(self):
            super().__init__()
            width = params['width']
            self.encoder = nn.Sequential(nn.Linear(members.shape[-1], width), nn.ReLU(),
                                         nn.Linear(width, width), nn.ReLU())
            self.interact = (nn.Sequential(nn.Linear(width * 2, width), nn.ReLU())
                             if params['pairwise'] else None)
            self.head = mlp(x.shape[-1] * 2 + width * (3 if self.interact else 2),
                            [64, 32], .15)

        def forward(self, flat, sets):
            encoded = self.encoder(sets)
            left, right = encoded.unbind(dim=1)
            parts = [flat.flatten(1), left.mean(1), right.mean(1)]
            if self.interact is not None:
                n_left, n_right = left.shape[1], right.shape[1]
                pairs = torch.cat((left[:, :, None].expand(-1, -1, n_right, -1),
                                   right[:, None].expand(-1, n_left, -1, -1)), dim=-1)
                parts.append(self.interact(pairs).mean(dim=(1, 2)))
            return self.head(torch.cat(parts, dim=1)).squeeze(1)

    model = (Matchup() if arm == 'C' else mlp(x.shape[-1] * 2, params['widths'], params['dropout'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['decay'], foreach=True)
    criterion = nn.BCEWithLogitsLoss()

    def forward(flat, sets):
        return model(flat, sets) if arm == 'C' else model(flat.flatten(1)).squeeze(1)

    @torch.no_grad()
    def evaluate(flat, sets, labels):
        model.eval()
        probabilities = []
        for i in range(0, len(labels), 512):
            batch = flat[i:i+512]
            tokens = None if sets is None else sets[i:i+512]
            direct = forward(batch, tokens).sigmoid()
            reversed_score = forward(batch.flip(1), None if tokens is None else tokens.flip(1)).sigmoid()
            probabilities.append((direct + 1 - reversed_score) / 2)
        p = torch.cat(probabilities).cpu().numpy()
        return metrics(labels.cpu().numpy(), p), p

    start = perf_counter()
    checkpoints = {}
    for epoch in range(1, max(epochs)+1):
        model.train()
        shuffled = torch.randperm(len(y), device=device)
        for batch in shuffled.split(data.config['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(forward(x[batch], None if members is None else members[batch]), y[batch])
            loss.backward()
            optimizer.step()
        if epoch in epochs:
            score, p = evaluate(vx, vm, vy)
            checkpoints[str(epoch)] = score
    result = {'validation': checkpoints, 'seconds': perf_counter()-start,
              'parameters': sum(parameter.numel() for parameter in model.parameters()),
              'gpu': torch.cuda.get_device_name(device)}
    if save_predictions:
        result.update(train=evaluate(x, members, y)[0], probabilities=p.tolist(), indices=indices,
                      train_count=len(y), validation_count=len(vy))
    return result


def run(action, gpu):
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = f'cuda:{gpu}'
    torch.cuda.set_device(device)
    data = Data()
    base.freeze(OUT / 'runtime.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
                'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                'precision': 'float32, deterministic algorithms, TF32 disabled',
                'protocol_sha256': data.digest}) if gpu == 0 else None
    grid = candidates(data.config)
    epochs = data.config['epochs']
    if action == 'pilot':
        choices = [next(c for c in reversed(grid) if c['family'] == arm) for arm in ('A', 'B', 'C')]
        results = []
        for c in choices:
            result = fit(data, data.plans[0], 0, c, device, epochs)
            results.append({'arm': c['family'], 'seconds': result['seconds'], 'parameters': result['parameters']})
        base.atomic(OUT / f'pilot-gpu{gpu}.json', results)
        print(json.dumps(results), flush=True)
        return
    for plan in data.plans:
        folder = OUT / 'checkpoints' / plan['id']
        for i, candidate in enumerate(grid):
            if action != 'grid' or (i + int(plan['outer'])) % 2 != gpu:
                continue
            path = folder / f'candidate-{i:03}.json'
            if base.checkpoint(path, data.digest, candidate=candidate) is not None:
                continue
            scores = [fit(data, plan, fold, candidate, device, epochs) for fold in range(len(plan['inner']))]
            base.atomic(path, {'protocol_sha256': data.digest, 'candidate': candidate, 'folds': scores})
            if i % 12 in (0, 1):
                print(f'{plan["id"]} candidate {i+1}/{len(grid)} GPU {gpu}', flush=True)
        if action == 'grid':
            print(f'{plan["id"]} inner shard complete GPU {gpu}', flush=True)
    if action != 'finalize':
        return
    outer = []
    for plan in data.plans:
        folder = OUT / 'checkpoints' / plan['id']
        stored = [base.checkpoint(folder / f'candidate-{i:03}.json', data.digest, candidate=c)
                  for i, c in enumerate(grid)]
        if any(record is None for record in stored):
            raise ValueError(f'Incomplete inner search for {plan["id"]}')
        for arm in ('A', 'B', 'C'):
            options = [(float(np.mean([f['validation'][str(e)]['log_loss'] for f in r['folds']])), i, e)
                       for i, r in enumerate(stored) if r['candidate']['family'] == arm for e in epochs]
            loss, index, epoch = min(options)
            candidate = grid[index]
            path = folder / f'outer-{arm}.json'
            record = base.checkpoint(path, data.digest, candidate=candidate, epoch=epoch)
            if record is None:
                score = fit(data, plan, -1, candidate, device, [epoch], True)
                record = {'plan': plan['id'], 'arm': arm, 'protocol_sha256': data.digest,
                          'inner_log_loss': loss, 'candidate': candidate, 'epoch': epoch, **score}
                base.atomic(path, record)
            outer.append(record)
    comparison = {}
    control = {r['plan']: r for r in outer if r['arm'] == 'A'}
    for arm in ('A', 'B', 'C'):
        selected = [r for r in outer if r['arm'] == arm]
        summary = {}
        for metric in ('log_loss', 'brier', 'accuracy'):
            values = [r['validation'][str(r['epoch'])][metric] for r in selected]
            summary[metric] = {'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1))}
        summary['paired_log_loss_minus_A'] = {
            r['plan']: r['validation'][str(r['epoch'])]['log_loss'] -
            control[r['plan']]['validation'][str(control[r['plan']]['epoch'])]['log_loss'] for r in selected}
        summary['epochs'] = dict(Counter(r['epoch'] for r in selected))
        comparison[arm] = summary
    base.atomic(OUT / 'summary.json', {'protocol_sha256': data.digest, 'complete': True,
                                       'comparison': comparison, 'outer_results': outer})
    print(json.dumps(comparison), flush=True)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'pilot', 'grid', 'finalize'))
    parser.add_argument('--gpu', type=int, choices=(0, 1), default=0)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    # Guard each GPU against duplicate workers. Finalization additionally checks every checkpoint.
    lock_name = 'prepare.lock' if args.action == 'prepare' else f'gpu{args.gpu}.lock'
    with (OUT / lock_name).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'prepare':
            prepare()
        else:
            run(args.action, args.gpu)


if __name__ == '__main__':
    main()
