# Autonomous Driving Agent: Reinforcement Learning in CarRacing-v3

## Overview
This project explores decision-making in Reinforcement Learning (RL) by training an autonomous agent to navigate a 2D racing track. The agent starts with zero knowledge of the environment and learns a driving policy entirely through trial and error, processing raw pixel data to make steering, acceleration, and braking decisions.

This project was built using `gymnasium` for the environment and `stable-baselines3` for the RL framework.

## Sources used: 
1. youtube : Nicholas Renotte's Rl learing course: the course is 5 years old so a lot of thigs have changed , like he uses openAi gym where as now we have to ue gymnasium , models of versions have changed .
2. 
2. Gymnasium documentation : great resource , the video was good as a a starting point , but the doc helped a lot .

3. Gemini : obviously :) 




## 1. The Learning Process
The project demonstrates the learning pipeline in two distinct phases:
* **Untrained Agent (Random Policy):** The script first runs a baseline test using `env.action_space.sample()`. these are just random actions done by the car . 

* **Trained Agent (PPO):** The agent is then trained for 10,000 timesteps using Proximal Policy Optimization (PPO). After training, the model's weights are saved, reloaded, and evaluated. While 10,000 steps is a very short training window for this environment, the agent begins to show early signs of learning (e.g., favoring the gas pedal to avoid time penalties), moving away from purely stochastic behavior.
* a timestep of 500000 would have given a lot better  results ,but idts my laptop can handle that 


## 2. State (Observation Space)
The state provided to the agent is a **96x96 RGB pixel image** of the environment (`Box(0, 255, (96, 96, 3), uint8)`). The agent essentially "sees" the track from a top-down bird's-eye view. Because the state consists of visual pixel data rather than simple numerical coordinates, a **Convolutional Neural Network (CNN)** was required to process the observations.

## 3. Actions (Action Space)
The agent operates in a **Continuous Action Space** (`Box([-1. 0. 0.], [1. 1. 1.], (3,), float32)`). It outputs an array of three continuous floating-point numbers per frame:
1. **Steering:** Ranges from `-1.0` (Full Left) to `1.0` (Full Right).
2. **Gas:** Ranges from `0.0` (No Gas) to `1.0` (Full Gas).
3. **Brake:** Ranges from `0.0` (No Brake) to `1.0` (Full Brake).

This allows for smooth, analog-style driving rather than rigid grid-based movement.

## 4. Reward Function Design
The environment utilizes a dense reward function designed to encourage speed and accuracy:
* **Track Completion:** The agent receives a positive reward for every new track tile it visits (roughly `+1000 / N`, where N is the total number of tiles).
* **Time Penalty:** The agent receives a `-0.1` penalty for every frame that passes.
* **Reasoning:** This balance forces the agent to not just stay on the track, but to complete it as quickly as possible. If there was no time penalty, the agent might learn to drive infinitely slow to avoid crashing. If the penalty is too high, the agent might intentionally drive off the track to end the episode early and stop the negative point bleed.

## 5. Failed Behavior & Analysis
**The Failure:** Because the model was only trained for 10,000 timesteps (an extremely short duration for visual continuous-control tasks), the agent fails to successfully navigate sharp corners. It frequently accelerates off the track into the grass, spins in circles, or gets stuck at the track boundaries. 
**The "Why":** The agent has started optimizing for the time-penalty (-0.1 per frame) by holding the gas, but it hasn't experienced enough episodes to map the visual cue of an approaching corner to the physical requirement of braking. It lacks an understanding of momentum and friction. 

## 6. Improvements Made for Performance
To achieve a stable and efficient training pipeline, several architectural improvements were made:
* **Vectorized Environments:** Wrapped the environment in a `DummyVecEnv`. This standardizes the data flow for Stable-Baselines3, effectively handling the transition between standard Gymnasium 5-tuple returns and SB3's internal processing.
* **Headless Training:** Separated the training environment from the evaluation environment. The model trains on a "headless" instance (no rendering), saving immense amounts of RAM and CPU overhead, and only triggers PyGame rendering during the final evaluation phase.

## 7. Architectural Decisions & Rejected Methods
* **Why Stable-Baselines3 (SB3)?** I chose to use an established framework rather than writing PyTorch gradient math from scratch. This allowed me to focus heavily on core RL concepts (tuning the reward loop, managing environments, and handling continuous spaces) rather than debugging low-level neural network architecture.
* **Why PPO over DQN?** DQN (Deep Q-Network) is designed for discrete action spaces (e.g., Up, Down, Left, Right). Because driving requires continuous analog inputs, PPO (Proximal Policy Optimization) was selected, as it is the industry standard for continuous control tasks.
* **Why `CnnPolicy` over `MlpPolicy`?** An MLP (Multi-Layer Perceptron) expects a 1D array of numbers. Since our state is a 3D image array (96, 96, 3), an MLP would fail to capture the spatial relationships of the track. `CnnPolicy` was chosen to properly extract visual features.
* **Why 2D instead of 3D (e.g., Trackmania)?** Due to hardware constraints  a 2D environment was chosen to ensure training could run efficiently on a CPU while still fulfilling all the theoretical requirements of the task.

## How to Run
1. Ensure you have the required dependencies installed:
   run the given command in terminal , than proceed with running the script .
   pip install "gymnasium[box2d]" stable-baselines3
   
