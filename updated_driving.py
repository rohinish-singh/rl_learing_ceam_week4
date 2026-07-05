import os
import gymnasium as gym
import highway_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO


# SETUP AND TEST THE ENVIRONMENT (RANDOM)


env = gym.make("highway-fast-v0", render_mode="human")
episodes = 3
for episode in range(1, episodes + 1):
    obs, info = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = truncated or terminated
        score += reward
        env.render()

print(f"Random Agent Total Reward: {score:.2f}")
env.close()


# 2. BACKGROUND TRAINING

log_path = "log_path_driving"
os.makedirs(log_path, exist_ok=True)

# Headless training env (No render_mode so it trains instantly in the background)
train_env = gym.make("highway-fast-v0")
train_env = DummyVecEnv([lambda: train_env])


model = PPO(policy="MlpPolicy", verbose=1, env=train_env, tensorboard_log=log_path)


model.learn(total_timesteps=100000)

train_env.close()


# 3. SAVE THE MODEL

sv_dir = "saved_models"
os.makedirs(sv_dir, exist_ok=True)
model_path = os.path.join(sv_dir, "highway_driving_model")

model.save(model_path)


# 4. EVALUATION & VISUALIZATION

del model

eval_env = gym.make("highway-fast-v0", render_mode="human")
eval_env = DummyVecEnv([lambda: eval_env])

model = PPO.load(model_path, env=eval_env)

print("Starting Evaluation...")
evaluate_policy(model, eval_env, n_eval_episodes=5, render=True)


# 5. FINAL TEST

print("Running final test track loop...")
obs = eval_env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = eval_env.step(action)
    eval_env.render()

    if dones.any():
        print("Test run finished!")
        break

eval_env.close()