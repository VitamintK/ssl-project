import json
import math
from collections import defaultdict
from pathlib import Path

# ── File picker ───────────────────────────────────────────────────────────────
_default = Path('results/all_downstream_tasks.json')
_archive = sorted(Path('results/archive').glob('downstream_results_*.json'), reverse=True)

_choices = [(_default, 'results/all_downstream_tasks.json (default)')]
_choices += [(_p, f'archive/{_p.name}') for _p in _archive[:10]]

print("Select results file:")
for _i, (_, _name) in enumerate(_choices):
    print(f"  [{_i}] {_name}")

_picked = None
while _picked is None:
    try:
        _idx = int(input("Index (default 0): ").strip() or "0")
        if 0 <= _idx < len(_choices):
            _picked = _choices[_idx][0]
        else:
            print(f"Enter 0–{len(_choices) - 1}.")
    except (ValueError, EOFError):
        _picked = _choices[0][0]

print(f"Loading {_picked}\n")
with open(_picked) as f:
    data = json.load(f)


def _mean(vs):
    return sum(vs) / len(vs)

def _std(vs):
    if len(vs) < 2:
        return 0.0
    m = _mean(vs)
    return math.sqrt(sum((v - m) ** 2 for v in vs) / (len(vs) - 1))

def fmt(values):
    if len(values) == 1:
        return f"{values[0]:.4f}"
    mean = _mean(values)
    return f"{mean:.4f}  (seeds: {', '.join(f'{v:.4f}' for v in values)})"


# ── Plain-text summary ────────────────────────────────────────────────────────
for run in data['run']:
    label = run['experiment_label']
    task_id = run.get('task_id', label.split()[-1])
    results = run['results']

    model_dir = results[0].get('model_dir')
    dir_suffix = f"  |  {model_dir}" if model_dir else ""
    print(f"{label}{dir_suffix}")

    if task_id in ('A', 'B', 'C', 'D'):
        mse_vals  = [r['val_metrics']['mse']          for r in results]
        base_vals = [r['val_metrics']['baseline_mse'] for r in results]
        print(f"  mse={fmt(mse_vals)}  baseline={fmt(base_vals)}")
    else:
        val_payoffs  = [r['val_metrics']['avg_empirical_payoff']     for r in results]
        val_exploits = [r['val_metrics']['avg_exact_exploitability'] for r in results]
        has_control  = 'control_metrics' in results[0]
        print(f"  val   payoff={fmt(val_payoffs)}  exploit={fmt(val_exploits)}")
        if has_control:
            ctrl_payoffs  = [r['control_metrics']['avg_empirical_payoff']     for r in results]
            ctrl_exploits = [r['control_metrics']['avg_exact_exploitability'] for r in results]
            print(f"  ctrl  payoff={fmt(ctrl_payoffs)}  exploit={fmt(ctrl_exploits)}")
    print()


# ── LaTeX table rows ──────────────────────────────────────────────────────────

# Metric keys per task_id.  Tasks A-D are regression tasks → mse/baseline_mse.
# Task E is the best-response learner → avg_empirical_payoff vs control.
def _get_metrics(result, task_id):
    if task_id in ('A', 'B', 'C', 'D'):
        mse  = result.get('val_metrics', {}).get('mse')
        base = result.get('val_metrics', {}).get('baseline_mse')
    else:  # task_id == 'E' or unknown
        mse  = result.get('val_metrics',     {}).get('avg_empirical_payoff')
        base = result.get('control_metrics', {}).get('avg_empirical_payoff')
    return mse, base

GAME_ORDER = ['kuhn_poker', 'leduc_poker']

# Encoder: detected by substring search in the label, checked in priority order
ENCODER_ORDER = [
    ('identity',                  'Identity'),
    ('reconstruction-autoencoder','Weight'),
    ('functional-autoencoder',    'Functional'),
    ('trajectory-encoder',        'Trajectory'),
    ('neupl',                     'NeuPL'),
]

TASK_RENAME = {
    'Task A': 'Task A',
    'Task B': 'Task B',
    'Task C': 'Task C',
    'Task D': 'Task C',   # relabelled
    'Task E': 'Task D',   # relabelled
}


def _parse_label(label):
    """Return (game_short, encoder_key, player, task_display) from an experiment label."""
    parts = label.split()
    game = parts[0]
    task_raw = f"{parts[-2]} {parts[-1]}"   # e.g. "Task E"
    task_display = TASK_RENAME.get(task_raw, task_raw)
    encoder_key = None
    for key, _ in ENCODER_ORDER:
        if key in label:
            encoder_key = key
            break
    # Player: look for 'p0'/'p1' or bare '0'/'1' after the game token
    player = None
    for part in parts[1:]:
        if part in ('p0', 'p1', '0', '1'):
            player = part.lstrip('p')   # normalise to '0' or '1'
            break
    return game, encoder_key, player, task_display


def _latex_cell(vals, n_decimals=3):
    """Return 'mean $\\pm$std' (no space before std, matching template style)."""
    m = _mean(vals)
    if len(vals) < 2:
        return f"{m:.{n_decimals}f}"
    return f"{m:.{n_decimals}f} $\\pm${_std(vals):.{n_decimals}f}"


def _imp_cell(mse_vals, base_vals):
    """Improvement: positive = model beats baseline."""
    m = _mean(mse_vals)
    b = _mean(base_vals)
    if b == 0:
        return "--"
    pct = (b - m) / b * 100
    return f"{pct:+.1f}\\%"


# Group: (encoder_key, task_display, player) -> game -> [(mse, baseline), ...]
grouped = defaultdict(lambda: defaultdict(list))

for run in data['run']:
    label = run['experiment_label']
    task_id = run.get('task_id', label.split()[-1])
    game, encoder_key, player, task_display = _parse_label(label)
    if encoder_key is None:
        continue
    for result in run['results']:
        mse, base = _get_metrics(result, task_id)
        if mse is None or base is None:
            continue
        grouped[(encoder_key, task_display, player)][game].append((mse, base))


# Emit rows (no header — caller pastes these inside their tabular environment)
current_encoder = None
for encoder_key, encoder_display in ENCODER_ORDER:
    row_keys = sorted({(task, player)
                       for (enc, task, player) in grouped
                       if enc == encoder_key})
    if not row_keys:
        continue

    if current_encoder is not None:
        print(r'\midrule')
    current_encoder = encoder_key

    for r_idx, (task, player) in enumerate(row_keys):
        encoder_col = (r'{\textbf{' + encoder_display + r'}}') if r_idx == 0 else ''
        task_col = f"{task} (p{player})" if player is not None else task
        game_cells = []
        for game in GAME_ORDER:
            pairs = grouped[(encoder_key, task, player)].get(game, [])
            if not pairs:
                game_cells += ['--', '--', '--']
            else:
                mse_vals  = [p[0] for p in pairs]
                base_vals = [p[1] for p in pairs]
                game_cells += [
                    _latex_cell(mse_vals),
                    _latex_cell(base_vals),
                    _imp_cell(mse_vals, base_vals),
                ]
        # columns: Encoder & Task & K_mse & K_base & K_imp & & L_mse & L_base & L_imp
        cols = [encoder_col, task_col] + game_cells[:3] + [''] + game_cells[3:]
        print(' & '.join(cols) + r' \\')
