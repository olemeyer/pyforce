from pyforce.env import wrap_openai_gym
from pyforce.nn import default_network_components
from pyforce.agents import PPOAgent
import gym
import torch

device="cuda:0" if torch.cuda.is_available() else "cpu"

env=wrap_openai_gym(gym.make("LunarLanderContinuous-v2"))

observation_processor,hidden_layers,action_mapper=default_network_components(env)

agent=PPOAgent(
    observation_processor,
    hidden_layers,
    action_mapper,
    save_path="./evals/ppo_example",
    value_lr=5e-4,
    policy_lr=5e-4
).to(device)

agent.train(env,episodes=1000,train_freq=2048,eval_freq=50,render=True, batch_size=128,gamma=.99,tau=.95,clip=.2,n_steps=32,entropy_coef=.01)