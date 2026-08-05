import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    '--aggregate-randall', action='store_true',
    help='Aggregate NeuPL randall_loss=True/False variants into one row (default: keep separate)',
)
args = parser.parse_args()
aggregate_randall = args.aggregate_randall

# ── File picker ───────────────────────────────────────────────────────────────
_default = Path('results/all_downstream_tasks.json')
_archive = sorted(Path('results/archive').glob('downstream_results_*.json'), reverse=True)

_choices = [(_default, 'results/all_downstream_tasks.json (default)')]
_choices += [(_p, f'archive/{_p.name}') for _p in _archive[:40]]

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
    sign = '-' if pct < 0 else ''
    return f"{sign}{abs(pct):.1f}\\%"


def _get_randall_loss(run):
    """Return the randall_loss bool from a run's config or label, or None if absent."""
    cfg = run.get('config', {})
    for key in ('use_randall_loss', 'randall_loss'):
        if key in cfg:
            return cfg[key]
    label = run.get('experiment_label', '')
    if 'randloss=True' in label:
        return True
    if 'randloss=False' in label:
        return False
    return None


def _encoder_display_name(encoder_key, base_display, randall_loss):
    """Return display name, appending +RL/-RL suffix for NeuPL variants when split."""
    if encoder_key == 'neupl' and randall_loss is not None:
        return base_display + ('+RL' if randall_loss else '-RL')
    return base_display


def _get_exploit(result, task_id):
    if task_id == 'E':
        return result.get('val_metrics', {}).get('avg_exact_exploitability')
    return None


def _abs_imp_cell(mse_vals, base_vals, n_decimals=3):
    """Absolute improvement (model payoff − control payoff); positive = model wins."""
    diff = _mean(mse_vals) - _mean(base_vals)
    return f"{diff:+.{n_decimals}f}"


# Group: (encoder_key, task_id, task_display, player, randall_loss) -> game -> [(mse, base, exploit), ...]
# task_id kept separately so Task E rows can be split into their own table.
grouped = defaultdict(lambda: defaultdict(list))

for run in data['run']:
    label = run['experiment_label']
    task_id = run.get('task_id', label.split()[-1])
    game, encoder_key, player, task_display = _parse_label(label)
    if encoder_key is None:
        continue
    randall = None if aggregate_randall else _get_randall_loss(run)
    for result in run['results']:
        mse, base = _get_metrics(result, task_id)
        if mse is None or base is None:
            continue
        exploit = _get_exploit(result, task_id)
        grouped[(encoder_key, task_id, task_display, player, randall)][game].append((mse, base, exploit))


def _regression_cells(pairs):
    if not pairs:
        return ['--', '--', '--']
    mse_vals  = [p[0] for p in pairs]
    base_vals = [p[1] for p in pairs]
    return [_latex_cell(mse_vals), _latex_cell(base_vals), _imp_cell(mse_vals, base_vals)]


def _task_e_cells(pairs):
    if not pairs:
        return ['--', '--', '--', '--']
    mse_vals     = [p[0] for p in pairs]
    base_vals    = [p[1] for p in pairs]
    exploit_vals = [p[2] for p in pairs if p[2] is not None]
    exploit_cell = _latex_cell(exploit_vals) if exploit_vals else '--'
    return [
        _latex_cell(mse_vals),
        _latex_cell(base_vals),
        _abs_imp_cell(mse_vals, base_vals),
        exploit_cell,
    ]


def _emit_encoder_rows(task_filter_fn, game_cells_fn):
    """Emit LaTeX rows for tasks matching task_filter_fn (no header, no tabular wrapper)."""
    current_encoder = None
    for encoder_key, encoder_display_base in ENCODER_ORDER:
        row_keys = sorted({(tid, task, player, randall)
                           for (enc, tid, task, player, randall) in grouped
                           if enc == encoder_key and task_filter_fn(tid)})
        if not row_keys:
            continue
        if current_encoder is not None:
            print(r'\midrule')
        current_encoder = encoder_key
        for r_idx, (tid, task, player, randall) in enumerate(row_keys):
            encoder_col = (r'{\textbf{' + encoder_display_base + r'}}') if r_idx == 0 else ''
            rl_tag = ' (+RL)' if randall else (' (-RL)' if randall is not None else '')
            task_col = f"{task} (p{player}){rl_tag}" if player is not None else f"{task}{rl_tag}"
            game_cells = []
            for game in GAME_ORDER:
                pairs = grouped[(encoder_key, tid, task, player, randall)].get(game, [])
                game_cells += game_cells_fn(pairs)
            mid = len(game_cells) // 2
            cols = [encoder_col, task_col] + game_cells[:mid] + [''] + game_cells[mid:]
            print(' & '.join(cols) + r' \\')


# ── LaTeX table 1: Tasks A–D (regression, percent improvement) ───────────────
print('\n% Tasks A–D')
_emit_encoder_rows(lambda tid: tid in ('A', 'B', 'C', 'D'), _regression_cells)

# ── LaTeX table 2: Task E (best-response, absolute improvement + exploitability)
print('\n% Task E (displayed as Task D)')
_emit_encoder_rows(lambda tid: tid == 'E', _task_e_cells)
