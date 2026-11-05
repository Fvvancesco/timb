"""Execute the frozen solver evaluation sequentially, with resumable child runs."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'config/solver_evaluation_v1.json').read_text())
    common = [sys.executable, str(ROOT / 'scripts/run_solver_benchmark.py'),
              '--budget', str(manifest['budget_seconds']), '--targets', str(ROOT / manifest['targets'])]
    for name, arguments in [
        ('solver_runtime_distribution', ['--instances', *manifest['representative_instances'],
                                         '--seeds', *map(str, manifest['seeds'])]),
        ('solver_breadth', ['--seeds', *map(str, manifest['breadth_seeds'])]),
    ]:
        output = ROOT / 'reports' / name
        output.mkdir(parents=True, exist_ok=True)
        if (ROOT / 'reports/solver_suite.stop').exists():
            print('Suite stopped before next stage.', flush=True)
            return
        print(f'Starting or resuming {name}', flush=True)
        with (output / 'execution.log').open('a') as log:
            subprocess.run(common + arguments + ['--output', str(output)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if (output / 'stop.requested').exists():
            return
        print(f'Completed {name}', flush=True)


if __name__ == '__main__':
    main()
