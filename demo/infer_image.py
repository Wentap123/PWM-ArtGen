"""Single image to articulated meshes, with separately configured Python environments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out_dir', required=True, type=Path)
    ap.add_argument('--stage', choices=['all', 'preprocess', 'inference', 'retrieval'], default='all')
    for name in ('image', 'seg_checkpoint', 'rmbg_model', 'vae_path', 'ckpt', 'database_root', 'graph_json', 'assignments_json', 'hashbook'):
        ap.add_argument('--' + name, type=Path)
    ap.add_argument('--seg_backend', choices=['sam3', 'sam'])
    ap.add_argument('--category', help='Override the graph API object category.')
    ap.add_argument('--graph_model', help='Graph API model (default: gpt-5-2025-08-07).')
    ap.add_argument('--device', help='Segmentation device (default: cuda). PWM inference uses CUDA when available.')
    ap.add_argument('--confidence', type=float, help='SAM3 confidence threshold (default: 0.5).')
    ap.add_argument('--no-postprocess', action='store_true', default=None)
    ap.add_argument('--dry-run', action='store_true', help='Show stage commands without loading models, writing, or calling APIs.')
    return ap


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(args):
    output = args.out_dir.expanduser().resolve()
    manifest_path = output / 'run.json'
    old = read_json(manifest_path) if manifest_path.exists() else None
    defaults = dict(seg_backend='sam3', device='cuda', confidence=.5,
                    ckpt=str(ROOT / 'checkpoints/pwm_final.pt'),
                    graph_model='gpt-5-2025-08-07', no_postprocess=False)
    config = dict(old['config'] if old else defaults)
    for name, value in vars(args).items():
        if name in ('out_dir', 'stage', 'dry_run') or value is None:
            continue
        config[name] = str(value.expanduser().resolve()) if isinstance(value, Path) else value
    if not config.get('image') or not Path(config['image']).is_file():
        raise ValueError('Supply --image with an existing file for a new run.')
    if Path(config['image']).is_relative_to(output):
        raise ValueError('The input image must be outside --out_dir.')
    if not 0 < config['confidence'] <= 1:
        raise ValueError('--confidence must be in (0, 1].')
    if config['seg_backend'] != 'sam' and config.get('assignments_json'):
        raise ValueError('--assignments_json applies only to SAM.')
    config['image_sha256'] = digest(config['image'])
    for name in ('graph_json', 'assignments_json', 'hashbook'):
        if config.get(name):
            config[name + '_sha256'] = digest(config[name])
    if old and config != old['config']:
        # Missing downstream paths may be supplied after a preprocess-only run.
        additions = {k for k in config if k not in old['config']}
        allowed = {'vae_path', 'database_root'}
        if not old['stages'].get('retrieval'):
            allowed.update(('hashbook', 'hashbook_sha256'))
        if not additions <= allowed or any(config[k] != v for k, v in old['config'].items()):
            raise ValueError('Run configuration or input content changed; choose a fresh --out_dir.')
    if not old and output.exists() and any(output.iterdir()):
        raise ValueError('A new run requires an empty --out_dir.')
    return output, {'config': config, 'stages': old['stages'] if old else {}}


def commands(config, output, stage):
    python = os.environ.get('PYTHON_BIN', sys.executable)
    backend_python = os.environ.get(config['seg_backend'].upper() + '_PYTHON',
                                   os.environ.get('PREPROCESS_PYTHON', python))
    stages = ['preprocess', 'inference', 'retrieval'] if stage == 'all' else [stage]
    result = []
    for name in stages:
        if name == 'preprocess':
            cmd = [backend_python, '-m', 'preprocess.run', '--run', str(output)]
        elif name == 'inference':
            cmd = ['bash', str(ROOT / 'scripts/infer.sh'), '--data_root', str(output / 'preprocess/data'),
                   '--ckpt', config['ckpt'], '--vae_path', config.get('vae_path', ''),
                   '--out_dir', str(output / 'predictions')]
        else:
            cmd = ['bash', str(ROOT / 'scripts/retrieve.sh'), '--pred_root', str(output / 'predictions'),
                   '--data_root', str(output / 'preprocess/data'), '--database_root', config.get('database_root', ''),
                   '--out_dir', str(output / 'retrieval'), '--workers', '1']
            if config['no_postprocess']:
                cmd.append('--no-postprocess')
            if config.get('hashbook'):
                cmd.extend(['--hashbook', config['hashbook']])
        result.append((name, cmd))
    return result


def run(args):
    output, manifest = configure(args)
    config = manifest['config']
    def has_outputs(name):
        if name == 'preprocess':
            return ((output / 'preprocess/prepared.json').is_file()
                    and any((output / 'preprocess/data').glob('test/*/input/waiting_use/view_id_00/joint_*/mask_renum_bbox.json')))
        if name == 'inference':
            return any((output / 'predictions/input').glob('joint_*/geom_pred.json'))
        return any((output / 'retrieval').glob('0@*@input/0/object_pwm.json'))
    for name, status in manifest['stages'].items():
        if status == 'complete' and not has_outputs(name):
            raise ValueError(f'Completed {name} outputs are missing; restore them or choose a fresh --out_dir.')
    pending = [(name, cmd) for name, cmd in commands(config, output, args.stage)
               if manifest['stages'].get(name) != 'complete']
    # Validate downstream resources before running paid API calls.
    for name, _ in pending:
        required = {'preprocess': ['seg_checkpoint'], 'inference': ['vae_path', 'ckpt'],
                    'retrieval': ['database_root']}[name]
        for key in required:
            if not config.get(key) or not Path(config[key]).exists():
                raise ValueError(f'{name} requires an existing --{key}.')
    if args.dry_run:
        import shlex
        for name, cmd in pending:
            print(f'[{name}] {shlex.join(cmd)}')
        return 0
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'run.json'
    write_json(manifest_path, manifest)
    for name, cmd in pending:
        prerequisites = {'preprocess': None, 'inference': 'preprocess', 'retrieval': 'inference'}
        prior = prerequisites[name]
        if prior and manifest['stages'].get(prior) != 'complete':
            raise ValueError(f'Complete {prior} before running {name}.')
        if name == 'retrieval' and manifest['stages'].get(name) in ('failed', 'running') and (output / 'retrieval').exists():
            raise ValueError('A retrieval attempt failed. Inspect outputs and move the retrieval subdirectory aside before retrying.')
        print(f'[{name}] Starting', flush=True)
        manifest['stages'][name] = 'running'
        write_json(manifest_path, manifest)
        env = dict(os.environ, PYTHON_BIN=os.environ.get('PYTHON_BIN', sys.executable))
        try:
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            if not has_outputs(name):
                raise ValueError(f'{name} exited without producing its expected outputs.')
        except BaseException:
            manifest['stages'][name] = 'failed'
            write_json(manifest_path, manifest)
            raise
        manifest['stages'][name] = 'complete'
        write_json(manifest_path, manifest)
    print(f'Results: {output}', flush=True)
    return 0


def main():
    try:
        return run(parser().parse_args())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f'[FAIL] {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
