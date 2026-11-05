"""Summarize completed development comparisons; never read final outcomes."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = {}
    for kind, prefix in [('main', 'supervised'), ('neural', 'neural'), ('trees', 'trees')]:
        for kb in ('v1', 'v2'):
            path = ROOT / f'reports/{prefix}_v2v3v4_kb_{kb}.json'
            reports[kind, kb] = (path, json.loads(path.read_text()))
    reference = reports['main', 'v1'][1]
    summary = []
    for (kind, kb), (path, report) in reports.items():
        if sorted(report['input_hashes'].values()) != sorted(reference['input_hashes'].values()):
            raise ValueError('Development input digests differ')
        if report['seeds'] != reference['seeds'] or report['outcomes'] != reference['outcomes']:
            raise ValueError('Development seeds or outcomes differ')
        folds = []
        for seed, run in report['runs'].items():
            if run['status'] != 'ok':
                raise ValueError('Incomplete development run')
            for a, b in zip(reference['runs'][seed]['folds'], run['folds'], strict=True):
                if a['outer'] != b['outer'] or a['inner'] != b['inner']:
                    raise ValueError('Fold counts differ')
                folds.append(b)
        for method in folds[0]['metrics']:
            if method in ('neutral', 'train_rate') and (kind, kb) != ('main', 'v1'):
                continue
            summary.append({'method': method, 'representation': kb,
                            'additional_after_core_freeze': kind == 'trees',
                            'metrics': {metric: {'mean': mean(f['metrics'][method][metric] for f in folds),
                                               'sd': stdev(f['metrics'][method][metric] for f in folds)}
                                        for metric in ('log_loss', 'brier', 'accuracy')},
                            'chosen_parameters': dict(Counter(str(f['metrics'][method]['parameter'])
                                                              for f in folds if 'parameter' in f['metrics'][method])),
                            'source': str(path.relative_to(ROOT))})
    report = {'scope': 'Development only: repeated nested group CV, 3 outer x 3 inner, seeds 11/23/47',
              'outcomes': reference['outcomes'], 'rows': summary,
              'limits': ['Equal-weight means over nine outer folds; SD is descriptive, not a confidence interval.',
                         'Folds overlap and cannot be treated as nine independent experiments.',
                         'v1 already includes four KB capability counts; v2 includes twelve. Neither is a no-KB ablation.',
                         'Tree and forest comparisons are additional development experiments; the frozen final model is unchanged.'],
              'source_hashes': {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
                                for path, _ in reports.values()}}
    (ROOT / 'reports/development_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Confronto supervisionato sullo sviluppo v2–v4', '',
             '5.463 esiti decisivi, stesse partizioni per composizione, CV annidata 3×3 con seed 11/23/47. '
             'Media e deviazione standard sui nove fold esterni. La deviazione standard descrive la dispersione: '
             'non è un intervallo di confidenza, perché i fold condividono osservazioni.', '',
             '| Metodo | Rappresentazione | Log loss ↓ (media ± SD) | Brier ↓ (media ± SD) | Accuracy (media ± SD) |',
             '|---|---|---:|---:|---:|']
    for row in summary:
        m = row['metrics']
        lines.append(f"| {row['method']} | {row['representation']} | "
                     f"{m['log_loss']['mean']:.5f} ± {m['log_loss']['sd']:.5f} | "
                     f"{m['brier']['mean']:.5f} ± {m['brier']['sd']:.5f} | "
                     f"{100*m['accuracy']['mean']:.2f}% ± {100*m['accuracy']['sd']:.2f} pp |")
    lines += ['', 'La random forest v1 ottiene la log loss media più bassa di questo confronto aggiuntivo. '
              'Non sostituisce il boosting già congelato e valutato nel test finale. Per una conferma indipendente '
              'di una futura versione con random forest servirebbe un nuovo protocollo e un nuovo test riservato.', '',
              'v1: 236 caratteristiche, incluse quattro capacità originarie della KB. v2: 252 caratteristiche, '
              'con dodici capacità. Il confronto riguarda i conteggi usati dal predittore, non elimina la KB '
              'dai vincoli e dalle spiegazioni. L’estensione non migliora in media nessuna delle famiglie provate.', '',
              'Albero: profondità 2/4/8, almeno 20 osservazioni per foglia. Random forest: 200 alberi, '
              'profondità 4/8/illimitata, almeno 10 osservazioni per foglia, max_features=sqrt, bootstrap, '
              'quattro processi/thread di lavoro. Scelta della profondità solo nei fold interni. '
              'Probabilità simmetrizzate come negli altri modelli. Nessuna nuova griglia è stata aggiunta dopo questi risultati.', '',
              'Per riprodurre il riepilogo: `.venv/bin/python scripts/summarize_development.py`. '
              'Per rifare gli esperimenti aggiuntivi sul remoto: '
              '`.venv/bin/python scripts/probe_tree_models.py --kb-version v1`, poi `v2`. '
              'Non eseguire training mentre il benchmark dei tempi usa lo stesso host.', '',
              'L’accuracy della costante p=0,5 usa la soglia >=0,5 e quindi predice sempre la stessa classe: '
              'non rappresenta un classificatore che estrae casualmente le etichette.']
    (ROOT / 'reports/CONFRONTO_MODELLI.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
