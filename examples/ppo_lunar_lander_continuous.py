from pyforce.env import DictEnv, ActionSpaceScaler, TorchEnv
from pyforce.nn.observation import ObservationProcessor
from pyforce.nn.hidden import HiddenLayers
from pyforce.nn.action import ActionMapper
from pyforce.agents import PPOAgent
import gym
import torch

device="cuda:0" if torch.cuda.is_available() else "cpu"

env=gym.make("LunarLanderContinuous-v2")
env=DictEnv(env)
env=ActionSpaceScaler(env)
env=TorchEnv(env).to(device)

observation_processor=ObservationProcessor(env)
hidden_layers=HiddenLayers(observation_processor.n_output)
action_mapper=ActionMapper(env,hidden_layers.n_output)

agent=PPOAgent(
    observation_processor,
    hidden_layers,
    action_mapper,
    save_path="./evals/ppo_example",
    value_lr=5e-4,
    policy_lr=5e-4
).to(device)

agent.train(env,episodes=1000,train_freq=2048,eval_freq=50,render=True, batch_size=128,gamma=.99,tau=.95,clip=.2,n_steps=32,entropy_coef=.01)