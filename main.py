
import sys
import json
from src.env import SolitaireEnv

def main():
    # Initialize the environment
    env = SolitaireEnv()
    done = False

    # Initialize dataset to collect state-action pairs
    dataset = []

    # Game loop
    while not done:
        # Render the current state
        print(f"\nReward: {env.reward}")
        env.render()

        # Get user input for action
        try:
            print("\nEnter your action:")
            input_str = input()

            # Split the input string into individual parts and convert them to integers
            action = list(map(int, input_str.split()))
        
            if action[0] == 9:
                break
        except ValueError:
            print("Invalid input! Please enter valid numbers.")
            continue

        # Get the current state (observation)
        current_state = env._get_observation()  # Modify this based on how the state is represented in your environment
        current_state_serializable = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in current_state.items()
        }

        # Add the state and action to the dataset
        dataset.append({'state': current_state_serializable, 'action': action})

        # Take the action
        try:
            obs, reward, done, terminal, info = env.step(action)
            if done:
                print("Congratulations! You have completed the game!\n Your score: ", env.reward)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Save the dataset to a file after the game ends
    with open("gameplay_data1.json", "w") as f:
        json.dump(dataset, f, indent=4)
    print("Dataset saved!")

if __name__ == "__main__":
    main()
