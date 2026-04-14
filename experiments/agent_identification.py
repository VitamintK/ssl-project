"""
Agent Identification Experiment

Given a held-out trajectory, classify which policy generated it.
This is a downstream task from Grover et al. (2018) that tests whether
learned embeddings are discriminative enough to distinguish individual policies.

Approach:
1. Load a pre-trained trajectory encoder
2. For each agent, embed multiple held-out trajectories
3. Train a k-NN or linear classifier on (embedding -> agent ID)
4. Report top-1 and top-5 accuracy on held-out trajectories
"""

import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pyspiel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from utils import PPOAgentPolicy, get_device_string, make_diverse_random_kuhn_poker_layer_init
from trajectory_encoder import (
    TrajectoryEncoder,
    TrajectoryEncoderConfig,
    TrajectoryEncoderAdapter,
    collect_trajectory_transitions,
    normalize_transitions,
)
from test_trajectory_encoder import load_trajectory_encoder_from_checkpoint


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def embed_trajectory(
    game: pyspiel.Game,
    model: TrajectoryEncoder,
    agent: PPOAgent,
    config: TrajectoryEncoderConfig,
    state_min: torch.Tensor,
    state_max: torch.Tensor,
    opponent_pool=None,
    device: str = "cpu",
) -> np.ndarray:
    """Embed a single trajectory from an agent. Returns a numpy embedding vector."""
    model.eval()
    with torch.no_grad():
        transitions = collect_trajectory_transitions(
            game, agent, player_id=0,
            num_transitions=config.num_transitions,
            opponent_pool=opponent_pool,
        )
        if config.normalize and state_min is not None:
            transitions, _, _ = normalize_transitions(transitions, state_min, state_max)

        traj_tensors = []
        for state, action, reward in transitions:
            t = torch.cat([
                state,
                torch.tensor([action], dtype=torch.float32),
                torch.tensor([reward], dtype=torch.float32),
            ])
            traj_tensors.append(t)

        trajectory = torch.stack(traj_tensors).unsqueeze(0).to(device)
        embedding = model(trajectory)
        # Handle VAE models that return tuples (recon_states, recon_actions, recon_rewards, mu, logvar)
        if isinstance(embedding, tuple):
            embedding = model.get_embedding(trajectory)
        return embedding.squeeze(0).cpu().numpy()


def collect_embeddings(
    game, model, agents, config, state_min, state_max,
    num_trajectories_per_agent: int,
    opponent_pool=None,
    device: str = "cpu",
):
    """Collect multiple trajectory embeddings per agent.

    Returns:
        embeddings: np.ndarray of shape (num_agents * num_trajectories_per_agent, embed_dim)
        labels: np.ndarray of agent IDs, shape (num_agents * num_trajectories_per_agent,)
    """
    embeddings = []
    labels = []
    for agent_id, agent in enumerate(tqdm(agents, desc="Embedding agents")):
        for _ in range(num_trajectories_per_agent):
            emb = embed_trajectory(
                game, model, agent, config,
                state_min, state_max,
                opponent_pool=opponent_pool,
                device=device,
            )
            embeddings.append(emb)
            labels.append(agent_id)
    return np.array(embeddings), np.array(labels)


def evaluate_knn(
    train_embeddings, train_labels,
    test_embeddings, test_labels,
    k_values=(1, 3, 5),
):
    """Evaluate k-NN classification accuracy."""
    results = {}
    max_k = max(k_values)
    n_classes = len(np.unique(train_labels))

    knn = KNeighborsClassifier(n_neighbors=max_k, metric="cosine")
    knn.fit(train_embeddings, train_labels)

    # Get probability estimates for top-k accuracy
    probs = knn.predict_proba(test_embeddings)

    for k in k_values:
        if k == 1:
            # Use a separate 1-NN for exact top-1
            knn1 = KNeighborsClassifier(n_neighbors=1, metric="cosine")
            knn1.fit(train_embeddings, train_labels)
            preds = knn1.predict(test_embeddings)
            acc = np.mean(preds == test_labels)
        else:
            acc = top_k_accuracy_score(test_labels, probs, k=min(k, n_classes), labels=knn.classes_)
        results[f"top{k}_acc"] = acc

    return results


def evaluate_linear_probe(
    train_embeddings, train_labels,
    test_embeddings, test_labels,
    num_classes: int,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """Evaluate linear probe classification accuracy."""
    embed_dim = train_embeddings.shape[1]

    model = nn.Linear(embed_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_train = torch.FloatTensor(train_embeddings).to(device)
    y_train = torch.LongTensor(train_labels).to(device)
    X_test = torch.FloatTensor(test_embeddings).to(device)
    y_test = torch.LongTensor(test_labels).to(device)

    # Train
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    top1_acc = np.mean(preds == test_labels)
    n_classes = len(np.unique(np.concatenate([train_labels, test_labels])))
    top5_acc = top_k_accuracy_score(
        test_labels, probs, k=min(5, n_classes),
        labels=list(range(num_classes)),
    )

    return {"top1_acc": top1_acc, "top5_acc": top5_acc}


def main():
    parser = argparse.ArgumentParser(description="Agent Identification Experiment")
    parser.add_argument("--game", type=str, default="kuhn_poker")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trajectory encoder checkpoint")
    parser.add_argument("--num-agents", type=int, default=100,
                        help="Number of agents to use")
    parser.add_argument("--train-trajectories", type=int, default=5,
                        help="Number of trajectories per agent for training")
    parser.add_argument("--test-trajectories", type=int, default=5,
                        help="Number of trajectories per agent for testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    device = args.device or get_device_string()
    print(f"Device: {device}")

    game = pyspiel.load_game(args.game)
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    # Resolve checkpoint path
    if args.checkpoint is None:
        # Try default checkpoints
        project_root = Path(__file__).resolve().parent.parent
        # Try both full name and short name (e.g. kuhn_poker -> kuhn)
        game_short = args.game.replace("_poker", "")
        candidates = [
            project_root / f"checkpoints/trajectory_encoder_{args.game}_random_improved_500.pt",
            project_root / f"checkpoints/trajectory_encoder_{game_short}_random_improved_500.pt",
            project_root / f"checkpoints/trajectory_encoder_{args.game}_random.pt",
            project_root / f"checkpoints/trajectory_encoder_{game_short}_random.pt",
        ]
        for c in candidates:
            if c.exists():
                args.checkpoint = str(c)
                break
        if args.checkpoint is None:
            print(f"No checkpoint found for {args.game}. Tried: {[str(c) for c in candidates]}")
            sys.exit(1)

    print(f"Loading encoder from {args.checkpoint}")
    model, adapter, config = load_trajectory_encoder_from_checkpoint(
        args.checkpoint, game, policies=None, device=device,
    )
    state_min = adapter.state_min
    state_max = adapter.state_max

    # Create random agents
    print(f"Creating {args.num_agents} random agents...")
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)
    agents = [
        PPOAgent(num_actions, info_state_size, "cpu", layer_init, 256)
        for _ in range(args.num_agents)
    ]

    # Collect train embeddings
    print(f"\nCollecting {args.train_trajectories} train trajectories per agent...")
    train_emb, train_labels = collect_embeddings(
        game, model, agents, config, state_min, state_max,
        num_trajectories_per_agent=args.train_trajectories,
        device=device,
    )
    print(f"Train set: {train_emb.shape}")

    # Collect test embeddings (separate rollouts)
    print(f"Collecting {args.test_trajectories} test trajectories per agent...")
    test_emb, test_labels = collect_embeddings(
        game, model, agents, config, state_min, state_max,
        num_trajectories_per_agent=args.test_trajectories,
        device=device,
    )
    print(f"Test set: {test_emb.shape}")

    # Evaluate with k-NN
    print("\n--- k-NN Classification ---")
    knn_results = evaluate_knn(train_emb, train_labels, test_emb, test_labels, k_values=(1, 3, 5))
    for metric, val in knn_results.items():
        print(f"  {metric}: {val:.4f}")

    # Evaluate with linear probe
    print("\n--- Linear Probe Classification ---")
    linear_results = evaluate_linear_probe(
        train_emb, train_labels, test_emb, test_labels,
        num_classes=args.num_agents, device=device,
    )
    for metric, val in linear_results.items():
        print(f"  {metric}: {val:.4f}")

    # Random baseline
    random_top1 = 1.0 / args.num_agents
    random_top5 = min(5.0 / args.num_agents, 1.0)
    print(f"\n--- Random Baseline ---")
    print(f"  top1_acc: {random_top1:.4f}")
    print(f"  top5_acc: {random_top5:.4f}")

    # Save results
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "game": args.game,
        "num_agents": args.num_agents,
        "train_trajectories": args.train_trajectories,
        "test_trajectories": args.test_trajectories,
        "seed": args.seed,
        "knn": knn_results,
        "linear_probe": linear_results,
        "random_baseline": {"top1_acc": random_top1, "top5_acc": random_top5},
    }
    save_path = results_dir / f"agent_id_{args.game}_seed{args.seed}.npy"
    np.save(save_path, results, allow_pickle=True)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
