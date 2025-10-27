import pyspiel

from tqdm import tqdm
from open_spiel.python import policy as policy_lib
from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from utils import diverse_random_kuhn_poker_layer_init
from downstream import PayoffPredictor, set_seed


def test_kuhn_poker_example():
    """
    Test the PayoffPredictor on Kuhn Poker with the specified configuration.
    """

    # Load Kuhn Poker game
    game = pyspiel.load_game("kuhn_poker")
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    # Create opponent policy (UniformRandomPolicy)
    opponent_policy = policy_lib.UniformRandomPolicy(game)

    # Create random PPO agents using diverse initialization
    print("Creating random PPO agents...")
    num_agents = 300
    ppo_agents = []

    for _ in tqdm(range(num_agents)):
        agent = PPOAgent(
            num_actions=num_actions,
            observation_shape=info_state_size,
            device="cpu"
        )

        for layer in agent.actor:
            if hasattr(layer, 'weight'):
                diverse_random_kuhn_poker_layer_init(game, layer)

        ppo_agents.append(agent)

    # Define encoder function
    # Encoder: lambda ppo_agent: ppo_agent.actor[0].weight.flatten()[::10]
    def encoder_fn(ppo_agent):
        """Extract weights from first layer of actor network, flatten and subsample."""
        # For PPOAgent, the actor is directly accessible
        first_layer = ppo_agent.actor[0]  # First linear layer
        weights = first_layer.weight.flatten()[::10]  # Every 10th element
        return weights

    # Create and train predictor
    predictor = PayoffPredictor(
        game=game,
        ppo_agents=ppo_agents,
        opponent_policy=opponent_policy,
        encoder_fn=encoder_fn,
        hidden_dims=[128, 64, 32],
        device="cpu"
    )

    # Train the model
    print("\nTraining predictor model...")
    history = predictor.train(
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-3,
        validation_split=0.2,
        verbose=True
    )

    # Evaluate
    print("\nEvaluating model...")
    metrics = predictor.evaluate()
    print(f"\nEvaluation Results:")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"R2: {metrics['r2']:.6f}")

    # Save the model
    predictor.save("payoff_predictor_kuhn.pth")

    return predictor, history, metrics


if __name__ == "__main__":
    set_seed(42)
    test_kuhn_poker_example()