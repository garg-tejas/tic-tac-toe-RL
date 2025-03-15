from mcts_ai import MCTSAI
import time
from utils import get_model_path
import torch

def main():
    start_time = time.time()
    
    # Optimize CUDA settings for training
    if torch.cuda.is_available():
        # Set to prevent fragmentation of GPU memory
        torch.backends.cudnn.benchmark = True
    
    # Create an MCTS agent
    agent = MCTSAI(exploration_weight=1.41, time_limit=0.5, temperature=0.1)

    # Force retraining (even if model file exists)
    print("Training MCTS agent...")
    
    # Clear training statistics
    agent.training_stats = {'wins': [], 'losses': [], 'draws': []}
    agent.position_values = {}
    
    # Train with curriculum learning
    print("Starting curriculum training...")
    agent.train_curriculum(episodes=50000)  # Train with more episodes
    
    # Save trained model
    agent.save_model()
    print(f"Model saved")
    
    # Show training stats
    try:
        agent.plot_training_progress()
    except Exception as e:
        print(f"Could not plot training progress: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Training complete in {elapsed_time:.2f} seconds!")

if __name__ == "__main__":
    main()