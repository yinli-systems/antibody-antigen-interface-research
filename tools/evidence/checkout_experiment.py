#!/usr/bin/env python3
"""Create a NEW detached worktree at a recorded experiment revision; never launch jobs."""
import argparse
import json
import subprocess
from pathlib import Path
from core import ROOT


def plan_command(name, destination, root=ROOT):
    root = Path(root)
    registry = json.loads((root/'tools/evidence/EXPERIMENT_REVISIONS.json').read_text())
    if name not in registry['experiments']:
        raise ValueError('unknown experiment: '+name)
    entry = registry['experiments'][name]
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('refusing an existing destination: '+str(destination))
    if destination == root.resolve() or root.resolve() in destination.parents:
        raise ValueError('destination must be outside this repository')
    resolved = subprocess.check_output(['git','-C',str(root),'rev-parse',entry['commit']+'^{commit}'],text=True).strip()
    if resolved != entry['commit']:
        raise ValueError('source commit did not resolve exactly')
    for path in entry['entrypoints']:
        subprocess.run(['git','-C',str(root),'cat-file','-e',resolved+':'+path],check=True)
    return ['git','-C',str(root),'worktree','add','--detach',str(destination),resolved],entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', nargs='?')
    parser.add_argument('destination', nargs='?')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    registry = json.loads((ROOT/'tools/evidence/EXPERIMENT_REVISIONS.json').read_text())
    if args.list:
        for name, entry in registry['experiments'].items():
            print(name, entry['commit'], entry['purpose'])
        return
    if not args.experiment or not args.destination:
        parser.error('experiment and a new destination are required (or --list)')
    command, entry = plan_command(args.experiment,args.destination)
    if args.dry_run:
        print(json.dumps({'command':command,'entry':entry},indent=2))
        return
    subprocess.run(command,check=True)
    print('Prepared exact source at '+str(Path(args.destination).resolve()))
    print('No GPU jobs started. Read '+entry['read_first']+' before running experiments.')

if __name__=='__main__': main()
