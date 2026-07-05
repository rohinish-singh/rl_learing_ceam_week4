# Autonomous Driving Agent: Highway Traffic Navigation

## Overview
This project explores decision-making in Reinforcement Learning (RL) by training an autonomous agent to navigate a multi-lane highway environment (`highway-fast-v0`). 

## Changes made for the free given week : 
1 .added randomization of the map , changes "paint colors" of the grass and road so the AI doesn't memorize pixels; it doesn't put physical objects in your way.
2. learnt about a new model : highway-fast-v0 

This project was built using `gymnasium`, `highway-env`, and `stable-baselines3`.

## 1. The Learning Process
The project demonstrates the learning pipeline in two distinct phases:
* **Untrained Agent (Random Policy):** The script first runs a baseline test using random actions. The car behaves erratically, switching lanes without looking and crashing into other vehicles almost immediately, resulting in early episode termination and negative scores.
* **Trained Agent (PPO):** The agent is then trained for 100,000 timesteps using Proximal Policy Optimization (PPO). After training, the agent's weights are saved, reloaded, and evaluated. The trained agent successfully learns to weave through traffic, maintain high speeds, and avoid collisions. 

## 2. State (Observation Space)
Unlike visual-based environments that output raw pixels, this environment outputs a **Kinematic Observation Space**. 
* The state is represented as a 2D matrix (typically `5x5`) containing the exact physical data of the agent's car (ego vehicle) and the nearest neighboring vehicles.
* The matrix includes features like: `[presence, x, y, vx, vy]` (whether a car is there, its X/Y coordinates, and its X/Y velocity).
* Because the state is a structured array of numbers rather than a high-dimensional image, an **MLP (Multi-Layer Perceptron)** policy was used instead of a CNN. This drastically improved training speed and sample efficiency.

## 3. Actions (Action Space)
The agent operates in a **Discrete Meta-Action Space**. Instead of controlling raw analog pedals and steering angles, the agent makes high-level decisions. The 5 discrete actions are:
* `0`: Change lane to the left
* `1`: Idle (maintain current speed and lane)
* `2`: Change lane to the right
* `3`: Faster (accelerate)
* `4`: Slower (decelerate)

## 4. Reward Function Design & Obstacles
The environment utilizes a reward function explicitly designed to balance speed with safety against dynamic obstacles:
* **High-Speed Reward:** The agent receives positive reinforcement for maintaining a high forward velocity (`target_speed`).
* **Collision Penalty:** Crashing into another vehicle (an obstacle) yields a massive negative penalty and instantly terminates the episode.
* **Reasoning:** This forces the agent to balance *Exploitation* (driving as fast as possible to maximize points) with *Exploration/Safety* (learning to slow down or change lanes when a slower car is blocking the path).

## 5. Failed Behavior & Analysis
**The Failure:** During early training, the agent often exhibits a behavior where it accelerates to maximum speed, but instead of changing lanes to avoid a slower car ahead, it simply slams into the back of it.
**The "Why":** The positive reward signal for driving fast is immediate and dense (it gets points every single frame it drives fast). The penalty for crashing is sparse (it only happens once at the very end). The agent briefly learned a greedy policy: "If I drive fast, I get points now, even if I die later." It takes tens of thousands of episodes for the model's value function to accurately predict the delayed penalty of a crash and learn that braking or changing lanes yields a higher net score over time.

## 6. Architectural Decisions & Frameworks
* **Why `highway-env`?** The baseline `CarRacing` environment is a static track with no dynamic objects. `highway-env` was chosen to introduce moving obstacles and test the agent's ability to predict the trajectory of other cars.
* **Why `MlpPolicy`?** Because the environment provides exact numerical coordinates rather than pixel data, using a Convolutional Neural Network (`CnnPolicy`) is impossible. Switching to `MlpPolicy` allowed the model to train 10x faster on a CPU.
* **Why Stable-Baselines3 (PPO)?** PPO is highly stable and prevents the agent from making disastrously large updates to its policy network. SB3 abstracts the complex math, allowing for a focus on environment design and reward tuning.

## How to Run
1. Ensure you have the required dependencies installed:
   run in terminal:
   pip install gymnasium highway-env stable-baselines3