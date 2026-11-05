"""Compare train-KB RDF2Vec features with identities and explicit capabilities."""
import argparse
from collections import defaultdict
from importlib.metadata import version
import json
from pathlib import Path
from random import Random
from urllib.parse import quote

import numpy as np
from joblib import Parallel, delayed
from rdflib import RDF, URIRef
from sklearn.neural_network import MLPClassifier

from run_team_profile_extension import (
    CATALOG, DATA, MODEL_SEEDS, OUTPUT, REPORT, ROOT, input_matrix, masked_cases,
    metrics, neural_probabilities, profile_matrix, rank_key, score,
)
from run_team_completion_extension import freeze
from run_team_profiles_external import TEST as PRIOR_EXTERNAL, prior_teams
from icon_pokemon.adapter import save_json, sha256, showdown
from icon_pokemon.evaluation import grouped_interval
from icon_pokemon.knowledge import ALL_CAPABILITIES, P, build_graph

PROTOCOL = ROOT / 'config/rdf2vec_external_v1.json'
TEST = DATA.with_name('external_metamon32.json')
OUT = OUTPUT / 'rdf2vec'
SOURCE = ('https://huggingface.co/datasets/jakegrigsby/metamon-raw-replays/resolve/'
          '060367c26630e0d36308fd42673e5cf2f4b4421a/data/train-00032-of-00046.parquet')
DIMENSIONS = (16, 32)
WALKS, DEPTH, SEED, W2V_EPOCHS = 64, 4, 307, 15


def protocol():
    inputs = (CATALOG, DATA, PRIOR_EXTERNAL, REPORT/'audit.json', OUTPUT/'selection.json',
              ROOT/'ontology/pokemon.ttl', ROOT/'ontology/move_semantics.ttl',
              ROOT/'ontology/move_annotations.json', ROOT/'tools/showdown/export-move-semantics.cjs',
              ROOT/'tools/showdown/package-lock.json', ROOT/'src/icon_pokemon/knowledge.py',
              ROOT/'src/icon_pokemon/move_semantics.py', Path(__file__))
    return {'source_url': SOURCE, 'min_rating': 1600, 'format': 'gen9ou',
            'models': ['identity', 'kb', 'rdf2vec'], 'primary': 'Top-5 delta vs identity',
            'secondary': ['Top-1', 'Log loss on common in-vocabulary targets'],
            'choice': 'Select RDF2Vec dimension by top-5 on original validation; log loss breaks ties.',
            'test': 'Distinct compositions from Metamon shard 32, excluding prior splits and catalog.',
            'network': {'dimensions': 32, 'epochs': 35, 'seeds': list(MODEL_SEEDS)},
            'embedding': {'dimensions': list(DIMENSIONS), 'walks_per_species': WALKS,
                          'depth_edges': DEPTH, 'seed': SEED, 'epochs': W2V_EPOCHS,
                          'window': 4, 'negative_samples': 5, 'architecture': 'skip-gram'},
            'input_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in inputs}}


def used_compositions():
    _, previous = prior_teams()
    external = json.loads(PRIOR_EXTERNAL.read_text())['teams']
    return previous | {tuple(t) for t in external}


def prepare(path):
    from run_human_expanded import ingest

    freeze(PROTOCOL, protocol())
    rows, species, counts = ingest([path], 1600)
    teams = {tuple(sorted(species[n]['base_species'] for n in side))
             for row in rows.values() for side in row['sides']}
    valid = {t for t in teams if len(t) == len(set(t)) == 6}
    previous = used_compositions()
    test = sorted(valid - previous)
    if len(test) < 2:
        raise ValueError('Insufficient unused compositions')
    freeze(TEST, {'source_url': SOURCE, 'source_sha256': sha256(path), 'source_counts': counts,
                  'protocol_sha256': sha256(PROTOCOL), 'previous_excluded': len(valid & previous),
                  'teams': [list(t) for t in test]})
    print(f'Frozen test: {len(test)} compositions; {len(valid & previous)} excluded as previously used.')


def graph_walks():
    catalog = json.loads(CATALOG.read_text())
    ids = {i for t in catalog['teams'].values() if t['split'] == 'train' for i in t['set_ids']}
    sets = sorted((r for r in catalog['sets'] if r['id'] in ids), key=lambda r: r['id'])
    semantics = showdown('export-move-semantics.cjs', {'format': catalog['format'],
        'move_ids': sorted({m['id'] for r in sets for m in r['moves']})})
    _, graph = build_graph({**catalog, 'sets': sets, 'teams': []}, move_semantics=semantics)
    relations = {P.hasSpecies, P.hasMove, P.hasType, P.hasAbility, P.hasItem}
    classes = {klass for pair in ALL_CAPABILITIES.values() for klass in pair}
    triples = {(str(s), str(p), str(o)) for s, p, o in graph
               if isinstance(s, URIRef) and isinstance(o, URIRef)
               and (p in relations or (p == RDF.type and o in classes))}
    bases = {r['base_species']: str(P['base-species/'+quote(r['base_species'], safe='')]) for r in sets}
    triples.update((str(P['species/'+r['species_id']]), str(P.hasBaseSpecies), bases[r['base_species']])
                   for r in sets)
    edges = defaultdict(list)
    for a, predicate, b in sorted(triples):
        edges[a].append((predicate, b))
        edges[b].append(('inverse:'+predicate, a))
    rng = Random(SEED)
    sequences = []
    for root in bases.values():
        for _ in range(WALKS):
            node = root
            walk = [node]
            for _ in range(DEPTH):
                predicate, node = rng.choice(edges[node])
                walk.extend([predicate, node])
            sequences.append(walk)
    return bases, sequences, {'train_sets': len(sets), 'species': len(bases),
                              'graph_nodes': len(edges), 'triples': len(triples), 'walks': len(sequences)}


def embedding_profiles(sequences, bases, dimension):
    from gensim.models import Word2Vec

    model = Word2Vec(sentences=sequences, vector_size=dimension, window=4, min_count=1,
                     sg=1, negative=5, epochs=W2V_EPOCHS, seed=SEED, workers=1)
    return {name: {'rdf2vec': model.wv.get_vector(node, norm=True).tolist()}
            for name, node in bases.items()}


def train_models(x, y, size):
    models = []
    for seed in MODEL_SEEDS:
        model = MLPClassifier(hidden_layer_sizes=(32,), activation='relu', solver='adam', alpha=.01,
                              batch_size=256, learning_rate_init=.005, random_state=seed)
        for _ in range(35):
            model.partial_fit(x, y, classes=np.arange(size))
        models.append(model)
    return models


def predict(models, x, visible, y):
    probabilities = np.mean([neural_probabilities(model, x, visible) for model in models], axis=0)
    return score(probabilities, y)[1]


def evaluate():
    if json.loads(PROTOCOL.read_text()) != protocol():
        raise ValueError('Frozen inputs differ')
    test_manifest = json.loads(TEST.read_text())
    if test_manifest['protocol_sha256'] != sha256(PROTOCOL):
        raise ValueError('Test protocol mismatch')
    teams = json.loads(DATA.read_text())['splits']
    vocab = json.loads((OUTPUT / 'selection.json').read_text())['candidate_species']
    kb = json.loads((REPORT/'audit.json').read_text())['profiles']
    bases, walks, summary = graph_walks()
    embeddings = {f'rdf{k}': embedding_profiles(walks, bases, k) for k in DIMENSIONS}
    print('RDF2Vec graph and walks:', summary, flush=True)
    feature_sets = {'identity': None, 'kb': (kb, 'kb'),
                    **{name: (p, 'rdf2vec') for name, p in embeddings.items()}}

    def features(visible, name):
        profile = feature_sets[name]
        return (input_matrix(visible, len(vocab)) if profile is None
                else profile_matrix(visible, vocab, *profile))

    train, y = masked_cases(teams['train'], vocab)
    visible, target = masked_cases(teams['validation'], vocab)
    models = dict(zip(feature_sets, Parallel(n_jobs=4)(delayed(train_models)(
        features(train, name), y, len(vocab)) for name in feature_sets), strict=True))
    validation = {name: metrics(predict(m, features(visible, name), visible, target))
                  for name, m in models.items()}
    choice = min(embeddings, key=lambda name: rank_key(validation[name]))
    selection = {'validation': validation, 'selected': choice, 'graph': summary,
                 'dependencies': {name: version(name) for name in ('gensim','numpy','scipy','scikit-learn','smart-open','wrapt')},
                 'protocol_sha256': sha256(PROTOCOL)}
    freeze(OUT/'selection.json', selection)
    print('Validation:', validation, 'selected:', choice, flush=True)
    test = [tuple(t) for t in test_manifest['teams']]
    if len(set(test)) != len(test) or set(test) & used_compositions():
        raise ValueError('Invalid holdout compositions')
    visible, y = masked_cases(test, vocab)
    compared = {'identity': 'identity', 'kb': 'kb', 'rdf2vec': choice}
    scores = {label: predict(models[name], features(visible, name), visible, y)
              for label, name in compared.items()}
    covered = y >= 0
    groups = np.repeat(np.arange(len(test)), 6)
    contrasts = {}
    for new, ref in [('rdf2vec','identity'), ('rdf2vec','kb'), ('kb','identity')]:
        delta = scores[new] - scores[ref]
        contrasts[f'{new}_minus_{ref}'] = {
            'top1_top5': grouped_interval(delta[:,:2], groups.tolist()),
            'log_loss_covered': grouped_interval(delta[covered,2], groups[covered].tolist())}
    report = {'selection': selection, 'test_teams': len(test), 'test_masks': len(y),
              'out_of_vocabulary_targets': int((~covered).sum()),
              'test_sha256': sha256(TEST), 'metrics': {k: metrics(v) for k,v in scores.items()},
              'comparisons': contrasts}
    save_json(OUT/'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'evaluate'))
    parser.add_argument('parquet', nargs='?', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.parquet is None:
            parser.error('prepare requires the pinned Metamon shard')
        prepare(args.parquet)
    else:
        evaluate()
