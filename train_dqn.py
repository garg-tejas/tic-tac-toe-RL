from dqn_ai import DQNTicTacToe
import time
from utils import get_model_path
import os

def main():
    start_time = time.time()
    
    # Create a DQN agent
    agent = DQNTicTacToe()
    
    # Force retraining (even if model file exists)
    print("Training DQN agent...")
    
    # Reset model weights to random initialization
    agent.policy_net.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    agent.update_target_network()
    
    # Clear training statistics
    agent.training_stats = {'wins': [], 'losses': [], 'draws': []}
    
    # Train with curriculum learning
    print("Starting curriculum training...")
    agent.train_curriculum(episodes=50000)  # Train with more episodes
    
    # Save trained model
    save_path = get_model_path("dqn")
    agent.save_model(save_path)
    print(f"Model saved to {save_path}")
    
    # Show training stats
    try:
        agent.plot_training_progress()
    except Exception as e:
        print(f"Could not plot training progress: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Training complete in {elapsed_time:.2f} seconds!")

if __name__ == "__main__":
    main()