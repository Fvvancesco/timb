"""Validate complete solver experiments and generate the report without rerunning search."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import sha256, save_json
from icon_pokemon.solver_reporting import summarize_runs, verify_run_set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=ROOT / 'reports/solver_runtime_distribution')
    parser.add_argument('--breadth', type=Path, default=ROOT / 'reports/solver_breadth')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/solver_comparison.json')
    args = parser.parse_args()
    protocol_path = ROOT / 'config/solver_evaluation_v1.json'
    protocol = json.loads(protocol_path.read_text())
    for path, digest in protocol['hashes'].items():
        if sha256(ROOT / path) != digest:
            raise ValueError('Frozen solver protocol dependency changed')
    instance_path = ROOT / protocol['instance_manifest']
    instances = {i['id']: i for i in json.loads(instance_path.read_text())['instances']}
    target_path = ROOT / protocol['targets']
    targets = json.loads(target_path.read_text())['targets']
    sources, summaries, configs = {}, {}, {}
    for stage, folder, ids, seeds in (
        ('runtime', args.runtime, protocol['representative_instances'], protocol['seeds']),
        ('breadth', args.breadth, list(instances), protocol['breadth_seeds'])):
        config = json.loads((folder / 'config.json').read_text())
        # The producer writes summary.json only once every planned run is complete.
        raw_summary = json.loads((folder / 'summary.json').read_text())
        runs = [json.loads(line) for line in (folder / 'runs.jsonl').read_text().splitlines()]
        verify_run_set(runs, ids, protocol['methods'], seeds)
        if (config['manifest_sha256'] != sha256(instance_path)
                or config['targets_sha256'] != sha256(target_path)
                or config['budget_seconds'] != protocol['budget_seconds']
                or config['validator_mode'] != 'persistent'
                or config['seeds'] != seeds or set(config['instance_ids']) != set(ids)
                or raw_summary['runs'] != len(runs)):
            raise ValueError('Experiment differs from its frozen protocol')
        if any(r['budget_seconds'] != config['budget_seconds'] for r in runs):
            raise ValueError('Different cutoff in run')
        configs[stage] = config
        summaries[stage] = {identifier: {
            'kind': instances[identifier]['kind'],
            'catalog_size': len(instances[identifier]['catalog_ids']),
            'target': targets[identifier],
            'methods': {method: summarize_runs([r for r in runs if r['instance_id'] == identifier
                                                and r['method'] == method], targets[identifier]['quality'])
                        for method in protocol['methods']}} for identifier in ids}
        sources[stage] = {str(folder / name): sha256(folder / name)
                          for name in ('config.json', 'runs.jsonl', 'summary.json')}
    for key in ('machine', 'platform', 'code_hashes', 'timing_scope'):
        if configs['runtime'][key] != configs['breadth'][key]:
            raise ValueError(f'Stages not comparable: {key}')
    report = {'scope': 'Completed sequential experiments on one host; training-domain construction requests',
              'protocol_sha256': sha256(protocol_path), 'machine': configs['runtime']['machine'],
              'platform': configs['runtime']['platform'], 'source_hashes': sources,
              'stages': summaries,
              'limits': ['Per-instance curves include all planned runs, including failures.',
                         'Unattained percentiles are null, not estimates using successes alone.',
                         'Certificate events apply only to CP-SAT; their absence in baselines is not a failed search.',
                         'Repeated CP-SAT seeds may follow identical trajectories; see distinct_incumbent_sequences.',
                         'Twenty development requests are not a claim about all possible construction tasks.',
                         'Symbolic objective quality is not battle win rate.']}
    save_json(args.output, report)
    lines = ['# Confronto dei costruttori e distribuzioni del tempo al successo', '',
             f"900 run su tre istanze (100 seed per metodo), più 300 run su venti istanze (cinque seed per metodo). "
             f"Budget 10 secondi; stessa macchina `{report['machine']}`, esecuzione sequenziale, "
             'un thread CP-SAT, KB e validatore già caricati. Gli insuccessi restano nei denominatori.', '',
             'La soglia di qualità è fissata dal pilot prima dei run: confronta lessicograficamente '
             'membri conservati, copertura offensiva, opzioni difensive statiche e ridondanza. '
             'Solo le soglie con certificato sono chiamate ottime.', '',
             '| Istanza | Catalogo | Metodo | Valida entro 10 s | Soglia entro 10 s | Mediana tempo alla soglia | p90 tempo alla soglia | Sequenze distinte |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    def seconds(value):
        return 'non raggiunto' if value is None else f'{value:.4f} s'
    for identifier, item in summaries['runtime'].items():
        for method, result in item['methods'].items():
            e, n = result['events'], result['runs']
            q = e['quality']['quantiles_seconds']
            lines.append(f"| {identifier} ({item['kind']}) | {item['catalog_size']} | {method} | "
                         f"{e['valid']['successes']}/{n} | {e['quality']['successes']}/{n} | "
                         f"{seconds(q['0.5'])} | {seconds(q['0.9'])} | {result['distinct_incumbent_sequences']} |")
    lines += ['', '## Verifica su tutte le venti richieste', '',
              '| Metodo | Valida entro 10 s | Soglia entro 10 s | Certificati entro 10 s |',
              '|---|---:|---:|---:|']
    for method in protocol['methods']:
        rows = [x['methods'][method] for x in summaries['breadth'].values()]
        n = sum(x['runs'] for x in rows)
        valid = sum(x['events']['valid']['successes'] for x in rows)
        quality = sum(x['events']['quality']['successes'] for x in rows)
        proof = (str(sum(x['events']['certificate']['successes'] for x in rows)) + f'/{n}') if method == 'cp_sat' else 'non applicabile'
        lines.append(f'| {method} | {valid}/{n} | {quality}/{n} | {proof} |')
    lines += ['', 'Ogni istanza ha lo stesso numero di repliche: il riepilogo dà peso uguale alle istanze. '
              'Le curve complete per ogni metodo/istanza sono nel JSON, insieme agli stati di arresto. '
              'I percentili usano tutti i run: se meno del 90% raggiunge la soglia, il p90 è non raggiunto entro il limite.', '',
              'Il numero di sequenze distinte esclude i timestamp e confronta le squadre e qualità degli incumbent. '
              'Una sola sequenza CP-SAT osservata significa che i seed non hanno prodotto percorsi osservabili diversi; '
              'le cento ripetizioni descrivono soprattutto variabilità di cronometraggio. Non basta cambiare il seed '
              'per dimostrare cento ricerche indipendenti.', '',
              'Preparazione e warm-up sono riportati nei `summary.json` sorgente e restano fuori dal tempo della singola richiesta. '
              'Il pilot con validatore a freddo e il lotto locale interrotto non sono mescolati a questi tempi.', '',
              'Riepilogo riproducibile: `.venv/bin/python scripts/summarize_solver_suite.py --runtime CARTELLA_RUNTIME '
              '--breadth CARTELLA_BREADTH`. Il comando rifiuta run mancanti, duplicati, host o codice differenti.']
    args.output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
