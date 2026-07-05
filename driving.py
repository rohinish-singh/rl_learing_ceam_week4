
import os


from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import gymnasium as gym
from sympy.polys.domains import domain

# setup and test the envoirmnt
#env = gym.make("CarRacing-v3" ) -- earlier version , had to read the doc for this
env = gym.make("CarRacing-v3" , domain_randomize=True )
episodes = 5
for episode in range(1, episodes + 1):
    obs , info= env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        n_state, reward, terminated, truncated, info = env.step(action)
        done = truncated or terminated
        score += reward


print(f"total rewaed {score}")


env.close()


# train the model using ppo
log_path = "log_path_driving"
os.makedirs(log_path , exist_ok=True)

#eval_env = gym.make("CarRacing-v3", render_mode="human")-- earlier version , had to read the doc for this
eval_env = gym.make("CarRacing-v3", render_mode="human", domain_randomize=True)
eval_env = DummyVecEnv([lambda: eval_env])



model = PPO(policy= "CnnPolicy" , verbose= 1 , env= eval_env , tensorboard_log=log_path)
model.learn(total_timesteps=10000)







# Save the model
sv_dir = "saved_models"
os.makedirs(sv_dir, exist_ok=True)
model_path = os.path.join(sv_dir, "my_driving_model")

model.save(model_path)

# Evaluation
del model

model = PPO.load(model_path, env=eval_env)

evaluate_policy(model , eval_env ,n_eval_episodes=10 ,render=True)


#test

obs = eval_env.reset() # when we vectorise pass obs only and not info
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = eval_env.step(action) # we use done and not terminated and trunicated as stbl vec still uses done
    eval_env.render("human")
    if dones.any():
        print("Test run finished!")
        break
eval_env.close()