from q_learning_ai import QLearningTicTacToe
import time

def main():
    start_time = time.time()
    
    # Create a Q-learning agent
    agent = QLearningTicTacToe()
    
    # Force retraining (even if q_table.pkl exists)
    print("Training Q-learning agent...")
    agent.q_table = {}  # Clear existing Q-table
    agent.opponent_q_table = {}  # Clear existing opponent Q-table
    
    # Initialize with heuristics
    agent.initialize_with_heuristics()
    
    # Train with curriculum learning
    agent.train_curriculum(episodes=20000)  # Train with more episodes
    
    # Save trained model
    agent.save_q_table()
    
    # Show training stats
    try:
        agent.plot_training_progress()
    except:
        print("Could not plot training progress")
    
    elapsed_time = time.time() - start_time
    print(f"Training complete in {elapsed_time:.2f} seconds!")

if __name__ == "__main__":
    main()