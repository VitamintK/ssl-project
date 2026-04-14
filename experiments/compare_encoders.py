"""
Compare trajectory encoder (contrastive) vs Grover encoder (hybrid) on downstream tasks.

Trains both encoders on the same agent population, then evaluates on:
  - Task D: Agent identification (k-NN and linear probe)
  - Task A: Fixed opponent payoff prediction (linear probe)
  - Task B: Agent vs agent pairwise payoff prediction (linear probe)

This provides a direct comparison of contrastive (NT-Xent) vs hybrid
generative-discriminative (imitation + triplet) objectives for learning
policy representations.
"""

import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import pyspiel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from utils import PPOAgentPolicy, get_device_string, make_diverse_random_kuhn_poker_layer_init
from trajectory_encoder import (
    TrajectoryEncoderConfig,
    TrajectoryEncoderTrainer,
    TrajectoryEncoderAdapter,
)
from grover_implementation import GroverConfig, GroverTrainer, GroverAdapter, GroverModelWrapper
from agent_identification import (
    set_seed, collect_embeddings, evaluate_knn, evaluate_linear_probe,
)


def train_contrastive_encoder(game, policies, config_overrides: dict, device: str):
    """Train our contrastive trajectory encoder."""
    config = TrajectoryEncoderConfig(
        hidden_dim=128, num_layers=4, num_heads=8, dropout=0.1,
        num_transitions=200, batch_size=16, epochs=200, lr=1e-4,
        weight_decay=1e-5, val_split=0.2, lr_scheduler="cosine",
        temperature=0.07, normalize=True, device=device, seed=42,
    )
    for k, v in config_overrides.items():
        setattr(config, k, v)

    trainer = TrajectoryEncoderTrainer(config, game, policies)
    model, history = trainer.train()

    adapter = TrajectoryEncoderAdapter(model, game, config, policies=policies)
    adapter.set_normalization_stats(
        trainer.train_dataset.state_min, trainer.train_dataset.state_max,
    )
    return model, adapter, config


def train_grover(game, policies, config_overrides: dict, device: str):
    """Train the Grover hybrid encoder."""
    config = GroverConfig(
        hidden_dim=100, embed_dim=128, policy_hidden_dim=64,
        lambda_weight=0.1,
        num_transitions=200, batch_size=16, epochs=200, lr=1e-4,
        weight_decay=1e-5, val_split=0.2, lr_scheduler="cosine",
        normalize=True, trajectories_per_policy=50,
        device=device, seed=42,
    )
    for k, v in config_overrides.items():
        setattr(config, k, v)

    trainer = GroverTrainer(config, game, policies)
    encoder, history = trainer.train()

    adapter = GroverAdapter(encoder, game, config, policies=policies)
    adapter.set_normalization_stats(
        trainer.train_dataset.state_min, trainer.train_dataset.state_max,
    )

    # Save checkpoint
    checkpoint_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"grover_{game.get_type().short_name}.pt"
    obs_dim = game.information_state_tensor_shape()[0]
    num_actions = game.num_distinct_actions()
    torch.save({
        "model_state_dict": encoder.state_dict(),
        "config": config,
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "state_min": trainer.train_dataset.state_min,
        "state_max": trainer.train_dataset.state_max,
    }, checkpoint_path)
    print(f"Saved Grover checkpoint to {checkpoint_path}")

    # Wrap for embed_trajectory compatibility
    wrapped_model = GroverModelWrapper(encoder, obs_dim, num_actions)

    return wrapped_model, adapter, config


def evaluate_agent_identification(
    name, game, model, config, state_min, state_max, agents,
    train_traj, test_traj, opponent_pool, device,
):
    """Run agent identification evaluation for a given encoder."""
    print(f"\n{'='*60}")
    print(f"Agent Identification: {name}")
    print(f"{'='*60}")

    from agent_identification import embed_trajectory

    def _collect(agents, n_traj):
        embeddings, labels = [], []
        for aid, agent in enumerate(tqdm(agents, desc=f"  {name} embedding")):
            for _ in range(n_traj):
                emb = embed_trajectory(
                    game, model, agent, config,
                    state_min, state_max,
                    opponent_pool=opponent_pool, device=device,
                )
                embeddings.append(emb)
                labels.append(aid)
        return np.array(embeddings), np.array(labels)

    train_emb, train_labels = _collect(agents, train_traj)
    test_emb, test_labels = _collect(agents, test_traj)

    knn = evaluate_knn(train_emb, train_labels, test_emb, test_labels)
    linear = evaluate_linear_probe(
        train_emb, train_labels, test_emb, test_labels,
        num_classes=len(agents), device=device,
    )
    return {"knn": knn, "linear_probe": linear}


def evaluate_payoff_prediction(name, game, encoder_fn, agents, device):
    """Run Task A: payoff prediction against uniform random."""
    from open_spiel.python import policy as policy_lib
    from downstream import PayoffPredictor

    print(f"\n{'='*60}")
    print(f"Task A (Payoff Prediction): {name}")
    print(f"{'='*60}")

    opponent = policy_lib.UniformRandomPolicy(game)

    embeddings = []
    for agent in tqdm(agents, desc=f"  {name} encoding"):
        emb = encoder_fn(agent).detach().cpu().numpy()
        embeddings.append(emb)

    from config import ModelConfig
    predictor = PayoffPredictor(
        game=game,
        p1_policies=[PPOAgentPolicy(game, a, 0, False) for a in agents],
        p2_policies=[opponent],
        p1_embeddings=embeddings,
        p2_embeddings=[np.array([0])],
        model_config=ModelConfig(hidden_dims=[], dropout=0.0, num_epochs=100, batch_size=16, learning_rate=1e-4),
        device="cpu",
    )

    predictor.compute_ground_truth_payoffs()
    predictor.train_with_agent_level_split(validation_split=0.2)
    val_metrics = predictor.evaluate(eval_set="val")

    train_payoffs = predictor.ground_truth_payoffs[predictor.trainer.train_indices]
    val_payoffs = predictor.ground_truth_payoffs[predictor.trainer.val_indices]
    baseline_mse = np.mean((val_payoffs - np.mean(train_payoffs)) ** 2)

    improvement = (1 - val_metrics["mse"] / baseline_mse) * 100
    print(f"  MSE: {val_metrics['mse']:.6f}  Baseline: {baseline_mse:.6f}  Improvement: {improvement:.1f}%")

    return {
        "mse": val_metrics["mse"],
        "baseline_mse": baseline_mse,
        "improvement_pct": improvement,
    }


def evaluate_pairwise_payoff(name, game, encoder_fn, agents, device):
    """Run Task B: agent vs agent pairwise payoff prediction."""
    from downstream import PayoffPredictor

    print(f"\n{'='*60}")
    print(f"Task B (Pairwise Payoff Prediction): {name}")
    print(f"{'='*60}")

    embeddings = []
    for agent in tqdm(agents, desc=f"  {name} encoding"):
        emb = encoder_fn(agent).detach().cpu().numpy()
        embeddings.append(emb)

    from config import ModelConfig
    predictor = PayoffPredictor(
        game=game,
        p1_policies=[PPOAgentPolicy(game, a, 0, False) for a in agents],
        p2_policies=[PPOAgentPolicy(game, a, 1, False) for a in agents],
        p1_embeddings=embeddings,
        p2_embeddings=embeddings,
        model_config=ModelConfig(hidden_dims=[], dropout=0.0, num_epochs=100, batch_size=16, learning_rate=1e-4),
        device="cpu",
    )

    predictor.compute_ground_truth_payoffs()
    predictor.train_with_agent_level_split(validation_split=0.2)
    val_metrics = predictor.evaluate(eval_set="val")

    train_payoffs = predictor.ground_truth_payoffs[predictor.trainer.train_indices]
    val_payoffs = predictor.ground_truth_payoffs[predictor.trainer.val_indices]
    baseline_mse = np.mean((val_payoffs - np.mean(train_payoffs)) ** 2)

    improvement = (1 - val_metrics["mse"] / baseline_mse) * 100
    print(f"  MSE: {val_metrics['mse']:.6f}  Baseline: {baseline_mse:.6f}  Improvement: {improvement:.1f}%")

    return {
        "mse": val_metrics["mse"],
        "baseline_mse": baseline_mse,
        "improvement_pct": improvement,
    }


def evaluate_policy_retrieval(
    name, game, model, config, state_min, state_max, agents,
    train_traj, test_traj, opponent_pool, device, k_values=(1, 5, 10),
):
    """Run policy retrieval: for each test embedding, find nearest train embeddings."""
    from sklearn.metrics.pairwise import cosine_similarity
    from agent_identification import embed_trajectory

    print(f"\n{'='*60}")
    print(f"Policy Retrieval: {name}")
    print(f"{'='*60}")

    def _collect(agents, n_traj):
        embeddings, labels = [], []
        for aid, agent in enumerate(tqdm(agents, desc=f"  {name} embedding")):
            for _ in range(n_traj):
                emb = embed_trajectory(
                    game, model, agent, config,
                    state_min, state_max,
                    opponent_pool=opponent_pool, device=device,
                )
                embeddings.append(emb)
                labels.append(aid)
        return np.array(embeddings), np.array(labels)

    train_emb, train_labels = _collect(agents, train_traj)
    test_emb, test_labels = _collect(agents, test_traj)

    # Compute cosine similarity between test and train embeddings
    sim_matrix = cosine_similarity(test_emb, train_emb)  # (n_test, n_train)

    results = {}
    for k in k_values:
        hits = 0
        for i in range(len(test_emb)):
            top_k_indices = np.argsort(sim_matrix[i])[::-1][:k]
            top_k_labels = train_labels[top_k_indices]
            if test_labels[i] in top_k_labels:
                hits += 1
        recall = hits / len(test_emb)
        results[f"recall@{k}"] = recall

    return results


def evaluate_strategy_classification(
    name, game, model, config, state_min, state_max, agents,
    opponent_pool, device, num_buckets=5,
):
    """Classify policies into aggression buckets using a linear probe on embeddings."""
    from agent_identification import embed_trajectory

    print(f"\n{'='*60}")
    print(f"Strategy Classification: {name}")
    print(f"{'='*60}")

    # Import aggression computation
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from visualize_embeddings import compute_aggression

    # Compute aggression for each agent
    print(f"  Computing aggression for {len(agents)} agents...")
    aggression_scores = []
    for agent in tqdm(agents, desc=f"  {name} aggression"):
        aggression_scores.append(compute_aggression(game, agent))
    aggression_scores = np.array(aggression_scores)

    # Discretize into buckets
    bucket_edges = np.linspace(aggression_scores.min(), aggression_scores.max(), num_buckets + 1)
    labels = np.digitize(aggression_scores, bucket_edges[1:-1])  # 0 to num_buckets-1

    # Embed each agent (single trajectory)
    embeddings = []
    for agent in tqdm(agents, desc=f"  {name} embedding"):
        emb = embed_trajectory(
            game, model, agent, config,
            state_min, state_max,
            opponent_pool=opponent_pool, device=device,
        )
        embeddings.append(emb)
    embeddings = np.array(embeddings)

    # Train/val split (80/20)
    n_train = int(0.8 * len(agents))
    train_emb, val_emb = embeddings[:n_train], embeddings[n_train:]
    train_labels, val_labels = labels[:n_train], labels[n_train:]

    # Linear probe
    import torch.nn as nn_module
    import torch.optim as optim_module
    embed_dim = train_emb.shape[1]
    probe = nn_module.Linear(embed_dim, num_buckets).to(device)
    optimizer = optim_module.Adam(probe.parameters(), lr=1e-3)
    criterion = nn_module.CrossEntropyLoss()

    X_train = torch.FloatTensor(train_emb).to(device)
    y_train = torch.LongTensor(train_labels).to(device)
    X_val = torch.FloatTensor(val_emb).to(device)
    y_val = torch.LongTensor(val_labels).to(device)

    probe.train()
    for _ in range(200):
        optimizer.zero_grad()
        loss = criterion(probe(X_train), y_train)
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(X_val).argmax(dim=1).cpu().numpy()
    accuracy = np.mean(preds == val_labels)
    random_baseline = 1.0 / num_buckets

    return {"accuracy": accuracy, "random_baseline": random_baseline, "num_buckets": num_buckets}


def main():
    parser = argparse.ArgumentParser(description="Compare contrastive vs Grover encoders")
    parser.add_argument("--game", type=str, default="kuhn_poker")
    parser.add_argument("--num-pretrain-agents", type=int, default=500)
    parser.add_argument("--num-eval-agents", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-contrastive", action="store_true",
                        help="Skip contrastive encoder (use existing checkpoint)")
    parser.add_argument("--contrastive-checkpoint", type=str, default=None)
    parser.add_argument("--skip-grover", action="store_true",
                        help="Skip Grover training (use existing checkpoint)")
    parser.add_argument("--grover-checkpoint", type=str, default=None)
    parser.add_argument("--tasks", type=str, default="ABD",
                        help="Tasks to run: A, B, D, F (few-shot ID), R (retrieval), S (strategy). e.g. 'FRS'")
    parser.add_argument("--train-traj", type=int, default=5,
                        help="Number of train trajectories per agent for Task D/F")
    parser.add_argument("--agent-pool", type=str, default=None,
                        help="Path to saved agent pool (from generate_agents.py)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device or get_device_string()
    print(f"Device: {device}")
    print(f"Game: {args.game}")

    game = pyspiel.load_game(args.game)
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    if args.agent_pool:
        from generate_agents import load_agents
        print(f"\nLoading agents from {args.agent_pool}...")
        pretrain_agents, eval_agents = load_agents(args.agent_pool, game)
    else:
        # Create pretrain population
        print(f"\nCreating {args.num_pretrain_agents} pretrain agents...")
        pretrain_agents = [
            PPOAgent(num_actions, info_state_size, "cpu", layer_init, 256)
            for _ in range(args.num_pretrain_agents)
        ]

        # Create eval population (separate)
        print(f"Creating {args.num_eval_agents} eval agents...")
        eval_agents = [
            PPOAgent(num_actions, info_state_size, "cpu", layer_init, 256)
            for _ in range(args.num_eval_agents)
        ]

    overrides = {"epochs": args.epochs}

    # --- Train / load contrastive encoder ---
    if args.skip_contrastive and args.contrastive_checkpoint:
        print("\nLoading contrastive encoder from checkpoint...")
        from test_trajectory_encoder import load_trajectory_encoder_from_checkpoint
        c_model, c_adapter, c_config = load_trajectory_encoder_from_checkpoint(
            args.contrastive_checkpoint, game, policies=pretrain_agents, device=device,
        )
    else:
        print("\n" + "="*60)
        print("Training CONTRASTIVE encoder")
        print("="*60)
        c_model, c_adapter, c_config = train_contrastive_encoder(
            game, pretrain_agents, overrides, device,
        )

    # --- Train / load Grover encoder ---
    if args.skip_grover and args.grover_checkpoint:
        print("\nLoading Grover encoder from checkpoint...")
        from grover_implementation import GroverEncoder
        checkpoint = torch.load(args.grover_checkpoint, map_location=device)
        g_config = checkpoint["config"]
        obs_dim = checkpoint["obs_dim"]
        num_actions = checkpoint["num_actions"]
        g_encoder = GroverEncoder(
            obs_dim=obs_dim, act_dim=num_actions,
            hidden_dim=g_config.hidden_dim, embed_dim=g_config.embed_dim,
        ).to(device)
        g_encoder.load_state_dict(checkpoint["model_state_dict"])
        g_adapter = GroverAdapter(g_encoder, game, g_config, policies=pretrain_agents)
        state_min = checkpoint["state_min"]
        state_max = checkpoint["state_max"]
        if state_min is not None:
            state_min = state_min.cpu()
        if state_max is not None:
            state_max = state_max.cpu()
        g_adapter.set_normalization_stats(state_min, state_max)
        g_model = GroverModelWrapper(g_encoder, obs_dim, num_actions)
    else:
        print("\n" + "="*60)
        print("Training GROVER encoder")
        print("="*60)
        g_model, g_adapter, g_config = train_grover(
            game, pretrain_agents, overrides, device,
        )

    tasks = args.tasks.upper()
    results = {
        "game": args.game,
        "seed": args.seed,
        "num_eval_agents": args.num_eval_agents,
    }

    # --- Evaluate: Task D (Agent Identification) ---
    if "D" in tasks:
        contrastive_id = evaluate_agent_identification(
            "Contrastive", game, c_model, c_config,
            c_adapter.state_min, c_adapter.state_max,
            eval_agents, train_traj=args.train_traj, test_traj=5,
            opponent_pool=pretrain_agents, device=device,
        )

        grover_id = evaluate_agent_identification(
            "Grover", game, g_model, g_config,
            g_adapter.state_min, g_adapter.state_max,
            eval_agents, train_traj=args.train_traj, test_traj=5,
            opponent_pool=pretrain_agents, device=device,
        )
        results["contrastive_agent_id"] = contrastive_id
        results["grover_agent_id"] = grover_id

    # --- Evaluate: Few-shot Agent Identification (Option 1) ---
    if "F" in tasks:
        contrastive_fewshot = evaluate_agent_identification(
            "Contrastive (1-shot)", game, c_model, c_config,
            c_adapter.state_min, c_adapter.state_max,
            eval_agents, train_traj=1, test_traj=5,
            opponent_pool=pretrain_agents, device=device,
        )

        grover_fewshot = evaluate_agent_identification(
            "Grover (1-shot)", game, g_model, g_config,
            g_adapter.state_min, g_adapter.state_max,
            eval_agents, train_traj=1, test_traj=5,
            opponent_pool=pretrain_agents, device=device,
        )
        results["contrastive_fewshot"] = contrastive_fewshot
        results["grover_fewshot"] = grover_fewshot

    # --- Evaluate: Policy Retrieval (Option 3) ---
    if "R" in tasks:
        contrastive_retrieval = evaluate_policy_retrieval(
            "Contrastive", game, c_model, c_config,
            c_adapter.state_min, c_adapter.state_max,
            eval_agents, train_traj=5, test_traj=5,
            opponent_pool=pretrain_agents, device=device,
        )

        grover_retrieval = evaluate_policy_retrieval(
            "Grover", game, g_model, g_config,
            g_adapter.state_min, g_adapter.state_max,
            eval_agents, train_traj=5, test_traj=5,
            opponent_pool=pretrain_agents, device=device,
        )
        results["contrastive_retrieval"] = contrastive_retrieval
        results["grover_retrieval"] = grover_retrieval

    # --- Evaluate: Strategy Classification (Option 2) ---
    if "S" in tasks:
        contrastive_strategy = evaluate_strategy_classification(
            "Contrastive", game, c_model, c_config,
            c_adapter.state_min, c_adapter.state_max,
            eval_agents, opponent_pool=pretrain_agents, device=device,
        )

        grover_strategy = evaluate_strategy_classification(
            "Grover", game, g_model, g_config,
            g_adapter.state_min, g_adapter.state_max,
            eval_agents, opponent_pool=pretrain_agents, device=device,
        )
        results["contrastive_strategy"] = contrastive_strategy
        results["grover_strategy"] = grover_strategy

    # --- Evaluate: Task A (Fixed Opponent) ---
    if "A" in tasks or "B" in tasks:
        c_encoder_fn = c_adapter.get_encoder(device=device)
        g_encoder_fn = g_adapter.get_encoder(device=device)

    if "A" in tasks:
        contrastive_payoff = evaluate_payoff_prediction(
            "Contrastive", game, c_encoder_fn, eval_agents, device,
        )
        grover_payoff = evaluate_payoff_prediction(
            "Grover", game, g_encoder_fn, eval_agents, device,
        )
        results["contrastive_payoff"] = contrastive_payoff
        results["grover_payoff"] = grover_payoff

    # --- Evaluate: Task B (Pairwise) ---
    if "B" in tasks:
        contrastive_pairwise = evaluate_pairwise_payoff(
            "Contrastive", game, c_encoder_fn, eval_agents, device,
        )
        grover_pairwise = evaluate_pairwise_payoff(
            "Grover", game, g_encoder_fn, eval_agents, device,
        )
        results["contrastive_pairwise"] = contrastive_pairwise
        results["grover_pairwise"] = grover_pairwise

    # --- Summary ---
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if "D" in tasks:
        print("\nTask D - Agent Identification (Top-1 Accuracy):")
        print(f"  Contrastive:  k-NN={contrastive_id['knn']['top1_acc']:.3f}  "
              f"Linear={contrastive_id['linear_probe']['top1_acc']:.3f}")
        print(f"  Grover:       k-NN={grover_id['knn']['top1_acc']:.3f}  "
              f"Linear={grover_id['linear_probe']['top1_acc']:.3f}")
        print(f"  Random:       {1/args.num_eval_agents:.3f}")

    if "F" in tasks:
        print("\nFew-shot Agent Identification (1-shot, Top-1 Accuracy):")
        print(f"  Contrastive:  k-NN={contrastive_fewshot['knn']['top1_acc']:.3f}  "
              f"Linear={contrastive_fewshot['linear_probe']['top1_acc']:.3f}")
        print(f"  Grover:       k-NN={grover_fewshot['knn']['top1_acc']:.3f}  "
              f"Linear={grover_fewshot['linear_probe']['top1_acc']:.3f}")
        print(f"  Random:       {1/args.num_eval_agents:.3f}")

    if "R" in tasks:
        print("\nPolicy Retrieval:")
        for k in [1, 5, 10]:
            key = f"recall@{k}"
            if key in contrastive_retrieval:
                print(f"  Recall@{k}:  Contrastive={contrastive_retrieval[key]:.3f}  "
                      f"Grover={grover_retrieval[key]:.3f}")

    if "S" in tasks:
        print("\nStrategy Classification (Aggression Buckets):")
        print(f"  Contrastive:  {contrastive_strategy['accuracy']:.3f}")
        print(f"  Grover:       {grover_strategy['accuracy']:.3f}")
        print(f"  Random:       {contrastive_strategy['random_baseline']:.3f}")

    if "A" in tasks:
        print("\nTask A - Fixed Opponent Payoff Prediction (MSE):")
        print(f"  Contrastive:  MSE={contrastive_payoff['mse']:.6f}  "
              f"Improvement={contrastive_payoff['improvement_pct']:.1f}%")
        print(f"  Grover:       MSE={grover_payoff['mse']:.6f}  "
              f"Improvement={grover_payoff['improvement_pct']:.1f}%")

    if "B" in tasks:
        print("\nTask B - Pairwise Payoff Prediction (MSE):")
        print(f"  Contrastive:  MSE={contrastive_pairwise['mse']:.6f}  "
              f"Improvement={contrastive_pairwise['improvement_pct']:.1f}%")
        print(f"  Grover:       MSE={grover_pairwise['mse']:.6f}  "
              f"Improvement={grover_pairwise['improvement_pct']:.1f}%")

    # Save
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"compare_{args.game}_seed{args.seed}.npy"
    np.save(save_path, results, allow_pickle=True)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
