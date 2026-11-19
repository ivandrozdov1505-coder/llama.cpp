#!/usr/bin/env python3
"""Compare two Dory llama-server builds with identical requests and settings."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

PROMPTS = [
    'Explain step by step how a computer processor executes instructions. Continue with detailed examples.',
    ('The research station has three rooms: laboratory, library, workshop. The library contains 42 books. ' * 30)
    + '\nWrite a detailed plan for organizing the station.',
    ('The research station has three rooms: laboratory, library, workshop. The library contains 42 books. ' * 100)
    + '\nWrite a detailed plan for organizing the station.',
]


def request(base, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=240) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--runtime-dir', type=Path, help='Optional CUDA runtime DLL directory on Windows')
    parser.add_argument('--output', type=Path, default=Path('dory-results.json'))
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--loops', type=int, default=2)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--port', type=int, default=18091)
    args = parser.parse_args()
    base = f'http://127.0.0.1:{args.port}'
    env = os.environ.copy()
    if args.runtime_dir:
        env['PATH'] = str(args.runtime_dir.resolve()) + os.pathsep + env.get('PATH', '')
    output = {'context': args.context, 'loops': args.loops, 'tokens': args.tokens, 'results': []}
    for run, label in enumerate(['baseline', 'candidate', 'candidate', 'baseline']):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', args.port)) == 0:
                raise RuntimeError(f'Port {args.port} is already occupied')
        executable = args.baseline if label == 'baseline' else args.candidate
        command = [str(executable.resolve()), '-m', str(args.model.resolve()), '--alias', 'dory',
                   '--host', '127.0.0.1', '--port', str(args.port), '--parallel', '1',
                   '--flash-attn', 'off', '--no-context-shift', '--jinja', '--cache-type-v', 'f16',
                   '--ctx-size', str(args.context), '-ngl', '99', '--batch-size', '256',
                   '--ubatch-size', '128', '--cache-type-k', 'q8_0', '--threads', '5',
                   '--threads-batch', '5', '--temp', '1', '--top-p', '0.95', '--top-k', '0',
                   '--min-p', '0.01', '--repeat-penalty', '1', '--n-predict', '16384',
                   '--reasoning-budget', '-1', '--reasoning', 'on', '--no-reasoning-preserve',
                   '--override-kv', f'dory.num_loops=int:{args.loops},dory.recurrent_kv_cache_mode=str:last_loop']
        log_path = args.output.with_name(args.output.stem + f'-{run}.log')
        with log_path.open('wb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                for _ in range(120):
                    if process.poll() is not None:
                        raise RuntimeError(f'{label} exited; see {log_path}')
                    try:
                        request(base, '/health')
                        break
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(1)
                else:
                    raise RuntimeError('Model loading timed out')
                payload = {'temperature': 0, 'seed': 42, 'ignore_eos': True, 'cache_prompt': False}
                request(base, '/completion', {**payload, 'prompt': 'Warm up the model.', 'n_predict': 32})
                for index, prompt in enumerate(PROMPTS):
                    result = request(base, '/completion', {**payload, 'prompt': prompt, 'n_predict': args.tokens})
                    row = {'run': run, 'engine': label, 'prompt': index,
                           'timings': result['timings'], 'content': result['content']}
                    output['results'].append(row)
                    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
                    print(label, index, result['timings'], flush=True)
            finally:
                process.terminate()
                try:
                    process.wait(15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                time.sleep(2)


if __name__ == '__main__':
    main()
