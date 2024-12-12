# Space Invaders Deep Q-Network (DQN)

## Overview

This project demonstrates a Deep Q-Network (DQN) implementation for mastering the classic Atari game Space Invaders using reinforcement learning. By leveraging TensorFlow and Keras, the AI agent learns to play the game autonomously, showcasing the power of neural networks in solving complex decision-making tasks.

## 🎮 Features

- **Advanced Preprocessing**: 
  - Grayscale frame conversion
  - Frame downscaling to 84x84 pixels
  - 4-frame stacking for temporal context

- **Neural Network Architecture**:
  - Convolutional layers for feature extraction
  - Fully connected layers for high-level reasoning
  - Q-value prediction for game actions

- **Intelligent Learning Techniques**:
  - Epsilon-greedy exploration
  - Experience replay
  - Target network stabilization
  - Huber loss optimization

## 🛠 Requirements

- Python 3.8+
- TensorFlow 2.x
- Keras
- Gymnasium
- Ale-py
- NumPy
- Matplotlib (for visualization)

## 📦 Installation

```bash
pip install tensorflow keras gymnasium[atari] ale-py numpy matplotlib
```

## 🚀 Usage

### Training the Model

```bash
# Train from scratch
python train_space_invaders.py

# Resume training from a saved model
python train_space_invaders.py --load_model path_to_saved_model.keras
```

### Watching the Agent Play

```bash
python play_space_invaders.py
```

## 📊 Performance Metrics

### Reward Progression

![Reward Progression](reward_progression.png)

The graph illustrates the agent's learning journey:
- **Blue Line**: Raw episode rewards
- **Red Line**: 50-episode running average

### Key Performance Highlights

- **Best Score**: 1115 points
- **Learning Episodes**: Approximately 7510
- **Notable Milestone**: Achieved stable performance with running reward > 800

## 🧠 Learning Process

1. **Exploration Phase**: Random actions with decaying exploration rate
2. **Experience Replay**: Sampling and learning from past experiences
3. **Q-Value Optimization**: Continuous improvement of action-value estimations
4. **Target Network Synchronization**: Periodic weight updates for stability

## 🔬 Hyperparameters

- **Learning Rate**: 0.00025
- **Discount Factor (γ)**: 0.99
- **Batch Size**: 32
- **Exploration Rate (ε)**: Linear decay from 1.0 to 0.1
- **Replay Buffer**: 100,000 experiences

## 🚧 Future Improvements

- [ ] Implement Double DQN
- [ ] Experiment with dueling network architectures
- [ ] Cross-game generalization testing
- [ ] Advanced reward scaling techniques

## 📝 License

[Specify your license here, e.g., MIT License]

## 🙏 Acknowledgements

- DeepMind's original DQN paper
- OpenAI Gymnasium
- Atari Learning Environment

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](your-github-repo-issues-link).