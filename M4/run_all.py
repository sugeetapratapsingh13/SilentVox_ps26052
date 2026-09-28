from pathlib import Path
import csv
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from config import ROOT, DIRS

EXPECTED = [
    DIRS['results'] / 'audio_metrics.csv',
    DIRS['results'] / 'block_latency_benchmark.csv',
    DIRS['results'] / 'robustness_tests.csv',
    DIRS['results'] / 'output_simulation.wav',
]
PLOT_GLOB = DIRS['plots'] / '*.png'

stages = [
    ('Audio analysis', [sys.executable, '-m', 'analysis.run_analysis']),
    ('Robustness', [sys.executable, '-m', 'analysis.robustness']),
    ('Block benchmark', [sys.executable, 'benchmark.py']),
    ('Unit tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py']),
]

# Remove generated evidence so stale files cannot make validation appear successful.
for p in EXPECTED:
    p.unlink(missing_ok=True)
for p in DIRS['plots'].glob('*.png'):
    p.unlink()

results=[]
for name, cmd in stages:
    print(f'\n>>> {" ".join(cmd)}')
    proc = subprocess.run(cmd, cwd=ROOT)
    status = 'PASS' if proc.returncode == 0 else 'FAIL'
    evidence = []
    if name == 'Audio analysis': evidence = ['results/audio_metrics.csv', 'plots/*.png']
    elif name == 'Robustness': evidence = ['results/robustness_tests.csv']
    elif name == 'Block benchmark': evidence = ['results/block_latency_benchmark.csv']
    else: evidence = ['tests/test_*.py']
    results.append({'test_name':name,'command': ' '.join(['python'] + cmd[1:] if cmd and cmd[0] == sys.executable else cmd),'status':status,'evidence_file':'; '.join(evidence),'timestamp_utc':datetime.now(timezone.utc).isoformat()})
    print(f'[{status}] {name}')
    if proc.returncode != 0:
        break

missing = [str(p.relative_to(ROOT)) for p in EXPECTED if not p.exists()]
plot_count = len(list(DIRS['plots'].glob('*.png')))
results.append({'test_name':'Expected evidence files','command':'runner file check','status':'PASS' if not missing else 'FAIL','evidence_file':'; '.join(missing) if missing else 'all expected result files present','timestamp_utc':datetime.now(timezone.utc).isoformat()})
results.append({'test_name':'Generated plot evidence','command':'runner plot check','status':'PASS' if plot_count > 0 else 'FAIL','evidence_file':f'{plot_count} PNG plots in plots/','timestamp_utc':datetime.now(timezone.utc).isoformat()})

out = DIRS['results'] / 'validation_summary.csv'
with open(out,'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)

failed = [r for r in results if r['status'] == 'FAIL']
print('\nM4 SOFTWARE VALIDATION')
for r in results: print(f"[{r['status']}] {r['test_name']}")
if failed:
    print('M4 software/digital validation FAILED.')
    raise SystemExit(1)
print('M4 software/digital validation completed.')
print(f'Validation summary: {out}')
