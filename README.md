# Deep Q-Network for Space Invaders

This project demonstrates the implementation and training of a Deep Q-Network (DQN) to play the classic Atari game Space Invaders. The goal is to use reinforcement learning techniques to teach an AI agent to maximize its score by learning from its interaction with the game environment. The project is built using Python, TensorFlow, and Keras.




## Overview

Deep Q-Networks (DQNs) are a groundbreaking innovation in reinforcement learning, combining neural networks with Q-learning to solve complex decision-making tasks. They enable agents to learn optimal actions by approximating the Q-function, which estimates the cumulative future rewards for each possible action in a given state. This technique has been pivotal in achieving human-level performance in various domains, including gaming, robotics, and resource optimization.

The significance of DQNs lies in their ability to process high-dimensional inputs, such as images, directly from raw pixels. For example, DeepMind's DQN achieved remarkable success in mastering Atari games by learning strategies that surpass human expertise. Real-world applications of DQNs include autonomous vehicle navigation, personalized recommendations, and inventory management systems.

In this project, the DQN architecture is adapted to Space Invaders, a classic Atari game, demonstrating how reinforcement learning can teach an AI agent to achieve high scores through iterative improvement and interaction with the game environment.
The DQN is a neural network-based reinforcement learning algorithm. It approximates the Q-function, which predicts the future rewards for taking specific actions in given states. The project architecture is inspired by DeepMind's original DQN paper and adapted to work with the Space Invaders environment.



## Key Features

**Environment**

**Game**: Space Invaders from the Atari games suite.

**Preprocessing**:

Grayscale frame conversion to reduce complexity by eliminating color information, which is not critical for gameplay. This helps the model focus on essential spatial and temporal features.

Downscaling frames to 84x84 pixels reduces computational load while retaining enough detail for effective decision-making.

Stacking 4 consecutive frames captures temporal dynamics, such as movement direction and speed, enabling the agent to make more informed predictions about future states.

**Game**: Space Invaders from the Atari games suite.

**Preprocessing**:

Grayscale frame conversion to reduce complexity.

Downscaling frames to 84x84 pixels.

Stacking 4 consecutive frames to capture temporal information.




## Neural Network

The Q-network architecture consists of:

1. Convolutional layers to extract features from input frames.

2. A flattening layer to convert 2D feature maps into a 1D vector.

3. Fully connected layers to model high-level reasoning.

4. An output layer to predict Q-values for all possible actions.

Below is a visual representation of the Q-network architecture, illustrating how input frames are processed through the layers:


The Q-network architecture consists of:

1. Convolutional layers to extract features from input frames.

2. A flattening layer to convert 2D feature maps into a 1D vector.

3. Fully connected layers to model high-level reasoning.

4. An output layer to predict Q-values for all possible actions.


## Hyperparameters

- Learning rate: 0.00025

- Discount factor (γ): 0.99

- Batch size: 32

- Exploration rate (ε): Linearly decayed from 1.0 to 0.1 over 1,000,000 frames.

- Replay buffer size: 100,000 experiences.

- Target network update frequency: Every 10,000 frames.


## Training

1. Optimizer: Adam with gradient clipping. This ensures the gradients remain stable and prevents exploding gradients, which can destabilize training.

2. Loss function: Huber loss to handle outliers in Q-value updates. Huber loss is less sensitive to large errors compared to Mean Squared Error, making it ideal for reinforcement learning.

3. Experience Replay: Stores past experiences for sampling during training, reducing correlation between updates. This improves sample efficiency and stabilizes learning by breaking the temporal correlation between consecutive experiences.

4. Epsilon-Greedy Policy: Balances exploration and exploitation by taking random actions during initial frames and gradually shifting to model-based decisions. This ensures the agent explores the environment sufficiently before focusing on optimizing known strategies.

## Implementation Process

1. **Environment Setup**

We used the Gymnasium library to load and preprocess the Space Invaders environment. Frame stacking and preprocessing were handled using AtariPreprocessing and FrameStack wrappers.

env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode="rgb_array")
env = AtariPreprocessing(env)
env = FrameStack(env, 4)

2. **Neural Network Design**

The Q-network architecture was implemented in Keras using convolutional and fully connected layers.

def create_q_model():
    return keras.Sequential([
        keras.Input(shape=(84, 84, 4)),
        layers.Conv2D(32, kernel_size=8, strides=4, activation="relu"),
        layers.Conv2D(64, kernel_size=4, strides=2, activation="relu"),
        layers.Conv2D(64, kernel_size=3, strides=1, activation="relu"),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(num_actions, activation="linear")
    ])

3. **##Training Loop**

The main training loop includes the following steps:

- Reset the environment: Start each episode by resetting the game state.

- Epsilon-greedy action selection: Select actions using the exploration-exploitation trade-off.

- Update replay buffer: Store experiences (state, action, reward, next state, done).

- Sample experiences: Train the model using a random batch of experiences from the replay buffer.

- Compute loss: Use the Bellman equation to compute Q-value updates.

- Optimize the model: Apply gradients to update model weights.

- Update target network: Periodically synchronize weights with the main network.

4. ## Monitoring Performance**

**The agent’s performance is tracked by**:

- Recording the reward per episode.

- Calculating a running average reward over the last 100 episodes.

- Saving the model when significant milestones are reached.


# Results

## Key Observations

The agent’s performance improved steadily, as indicated by increasing running reward and episodic rewards.

Best scores achieved include 1115 points within 7510 episodes, showcasing the model’s ability to learn effective strategies.

Despite fluctuations in reward, the model demonstrated consistent progress over time.



# Challenges

- Balancing exploration and exploitation required fine-tuning of epsilon decay.

- Training stability was sensitive to the choice of hyperparameters and experience replay size.

Sparse rewards in Space Invaders made it challenging to learn optimal policies.

# How to Use

## Prerequisites

1. Install required libraries:

pip install tensorflow keras gymnasium ale-py

2. Ensure the Atari ROMs are set up for the environment.


# Run the Model

1. Train the model from scratch:

python train_space_invaders.py

2. Resume training from a saved model:

model = keras.models.load_model("path_to_saved_model.keras")

3. Watch the agent play:

env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode="human")

Future Work

Extend Training: Train with more episodes and varied hyperparameters.

Fine-Tuning: Optimize exploration decay and reward scaling.

Additional Features: Introduce double DQN or dueling network architectures to improve performance.

Generalization: Test the trained model on other Atari games to evaluate its robustness.



# Conclusion

This project successfully implemented and trained a Deep Q-Network to play Space Invaders, showcasing the potential of reinforcement learning in mastering complex tasks. The journey highlighted the importance of meticulous preprocessing, hyperparameter tuning, and efficient experience replay mechanisms for achieving optimal results.

