import json

with open('results/all_downstream_tasks.json') as f:
    data = json.load(f)

for run in data['run']:
    label = run['experiment_label']
    results = run['results']

    val_payoffs = [r['val_metrics']['avg_empirical_payoff'] for r in results]
    val_exploits = [r['val_metrics']['avg_exact_exploitability'] for r in results]

    has_control = 'control_metrics' in results[0]
    if has_control:
        ctrl_payoffs = [r['control_metrics']['avg_empirical_payoff'] for r in results]
        ctrl_exploits = [r['control_metrics']['avg_exact_exploitability'] for r in results]

    def fmt(values):
        if len(values) == 1:
            return f"{values[0]:.4f}"
        mean = sum(values) / len(values)
        return f"{mean:.4f}  (seeds: {', '.join(f'{v:.4f}' for v in values)})"

    print(f"{label}")
    print(f"  val   payoff={fmt(val_payoffs)}  exploit={fmt(val_exploits)}")
    if has_control:
        print(f"  ctrl  payoff={fmt(ctrl_payoffs)}  exploit={fmt(ctrl_exploits)}")
    print()
