from pyforce.env import wrap_openai_gym
from pyforce.nn import default_network_components
from pyforce.agents import A2CAgent
import gym
import torch

device="cuda:0" if torch.cuda.is_available() else "cpu"

env=wrap_openai_gym(gym.make("LunarLanderContinuous-v2"))

observation_processor, hidden_layers, action_mapper=default_network_components(env)

agent=A2CAgent(
    observation_processor,
    hidden_layers,
    action_mapper,
    save_path="./evals/a2c_example",
    value_lr=1e-3,
    policy_lr=1e-3
).to(device)

agent.train(env,episodes=1000,train_freq=256,eval_freq=50,render=True,gamma=.99,entropy_coef=.01)