#!/usr/bin/env python3
"""Local launcher for the three standalone SWT workflows."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def read_config(path):
    """Read literal KEY=value settings; never execute a configuration file."""
    values = {}
    for line in path.read_text().splitlines():
        tokens = shlex.split(line, comments=True)
        if not tokens:
            continue
        key, sep, value = ' '.join(tokens).partition('=')
        if not sep or not key.replace('_', '').isalnum():
            raise ValueError(f'Expected literal KEY=value: {line}')
        values[key] = shlex.split(value.strip().strip('()'))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow', choices=('swt_transport', 'dssf_path_movie', 'swt_kernel'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    env = dict(os.environ, MPLBACKEND='Agg')
    if args.workflow == 'swt_transport':
        if args.output_root:
            parser.error('Set log_root and plot_root in the transport JSON.')
        config = (args.config or ROOT / 'swt_transport_job.json').resolve()
        command = [sys.executable, str(ROOT / 'swt_transport_job.py'), '--config', str(config)]
    else:
        kernel = args.workflow == 'swt_kernel'
        config = (args.config or ROOT / ('swt_kernel_job.conf' if kernel else 'swt_path_movie_job.conf')).resolve()
        values = read_config(config)
        label = values.pop('RUN_LABEL')[0]
        if Path(label).name != label or label in ('.', '..'):
            raise ValueError('RUN_LABEL must be one directory name')
        output = (args.output_root or Path('results') / args.workflow / label).resolve()
        aliases = {'J_PM_RATIO': 'j-pm', 'J_PMPM_RATIO': 'j-pmpm', 'J_ZZ_EV': 'j-zz-ev',
                   'FIELD_PHIS_DEGREES': 'phi-degrees' if kernel else 'field-phis',
                   'LATTICE_CONSTANT': 'lattice-constant'}
        script = 'plot_swt_kernel.py' if kernel else 'plot_swt_high_symmetry_dssf_movie.py'
        command = [sys.executable, str(ROOT / script), '--output-root', str(output)]
        for key, value in values.items():
            option = aliases.get(key, key.lower().replace('_', '-'))
            command.extend(['--' + option, *value])
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
