"""Export scientific figures from saved reports; no fitting or simulation."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/icon-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'random': '#757575', 'greedy': '#d97706', 'cp_sat': '#0072b2'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', type=Path, default=ROOT / 'reports/solver_comparison.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/figures')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    generated = []

    def save(fig, name):
        for extension in ('png', 'svg', 'pdf'):
            fig.savefig(args.output / f'{name}.{extension}', dpi=170, bbox_inches='tight')
        generated.append(name)
        plt.close(fig)

    development = ROOT / 'reports/development_comparison.json'
    if development.exists():
        report = json.loads(development.read_text())
        rows = [r for r in report['rows'] if r['method'] not in ('neutral', 'train_rate')]
        labels = {'logistic': 'Logistica', 'boosting': 'Boosting', 'mlp': 'MLP',
                  'decision_tree': 'Albero', 'random_forest': 'Random forest (aggiuntiva)'}
        fig, ax = plt.subplots(figsize=(9.5, 5.7), layout='constrained')
        for index, row in enumerate(rows):
            m = row['metrics']['log_loss']
            ax.errorbar(m['mean'], index, xerr=m['sd'], fmt='o', capsize=3,
                        color='#0072b2' if row['representation'] == 'v1' else '#d97706')
        ax.set_yticks(range(len(rows)), [f"{labels[r['method']]} · {r['representation']}" for r in rows])
        ax.invert_yaxis()
        ax.axvline(np.log(2), color='#666666', ls='--', label='Costante p=0,5')
        ax.set(title='Sviluppo: confronto dei predittori', xlabel='Log loss · minore è meglio')
        ax.grid(axis='x', alpha=.2)
        ax.legend(loc='lower right')
        fig.suptitle('Media ± SD dei 9 fold esterni; fold sovrapposti, barre non interpretabili come IC', fontsize=10)
        save(fig, 'development_models')
    final = ROOT / 'reports/final_evaluation.json'
    if final.exists():
        report = json.loads(final.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
        labels = ['Casuale nel pool', 'Simbolico', 'Supervisionato']
        for i, method in enumerate(('random', 'symbolic', 'supervised')):
            value = report['selectors'][method]['mean_request_win_rate']
            avg, (lo, hi) = value['mean'][0], value['interval_95'][0]
            axes[0].errorbar(avg, i, xerr=[[avg-lo], [hi-avg]], fmt='o', color='#0072b2', capsize=4)
        axes[0].set(yticks=range(3), yticklabels=labels, xlim=(0, 1),
                    title='Media delle win rate per richiesta', xlabel='Vittorie fra gli esiti decisivi')
        pairs = ['supervised_minus_symbolic', 'supervised_minus_random', 'symbolic_minus_random']
        for i, pair in enumerate(pairs):
            value = report['paired_selector_differences'][pair]
            avg, (lo, hi) = value['mean'][0], value['interval_95'][0]
            axes[1].errorbar(avg, i, xerr=[[avg-lo], [hi-avg]], fmt='o', color='#d97706', capsize=4)
        axes[1].axvline(0, color='#666666', ls='--')
        axes[1].set(yticks=range(3), yticklabels=['Sup. − simbolico', 'Sup. − casuale', 'Simbolico − casuale'],
                    xlim=(-.3, .3), title='Differenze appaiate', xlabel='Differenza di win rate (punti percentuali)')
        for ax in axes:
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylim(2.6, -.6)
            ax.grid(axis='x', alpha=.2)
        fig.suptitle('Test finale · 20 richieste · IC bootstrap 95% per richiesta (10.000 repliche)', fontsize=11)
        save(fig, 'final_selectors')
    if args.solver.exists():
        report = json.loads(args.solver.read_text())
        instances = report['stages']['runtime']
        fig, axes = plt.subplots(2, len(instances), figsize=(12, 6.8), sharex=True, sharey=True, layout='constrained')
        for col, (identifier, instance) in enumerate(instances.items()):
            for row, event in enumerate(('valid', 'quality')):
                ax = axes[row, col]
                for method, result in instance['methods'].items():
                    curve = result['events'][event]['curve']
                    ax.step([max(1e-4, p['seconds']) for p in curve], [p['fraction'] for p in curve],
                            where='post', color=COLORS[method], label=method)
                ax.set(xscale='log', xlim=(1e-4, 10), ylim=(-.02, 1.03))
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.grid(alpha=.2)
                if row == 0:
                    ax.set_title(f"{identifier} · {instance['catalog_size']} set\n{instance['kind']}")
                else:
                    ax.set_xlabel('Tempo (s, scala logaritmica)')
                if col == 0:
                    ax.set_ylabel('Prima squadra valida' if event == 'valid' else 'Soglia di qualità raggiunta')
        axes[0, -1].legend(loc='lower right')
        fig.suptitle('Distribuzioni empiriche · 100 run per metodo/istanza · insuccessi inclusi', fontsize=12)
        save(fig, 'solver_runtime_distributions')
    (args.output / 'environment.json').write_text(json.dumps({
        'versions': {package: version(package) for package in ('numpy', 'matplotlib')},
        'generated': generated, 'solver_present': args.solver.exists(),
        'note': 'Figures use saved metrics, never refit or change experiment results.'}, indent=2) + '\n')
    print(', '.join(generated))


if __name__ == '__main__':
    main()
