from pyforce.env import DictEnv, ActionSpaceScaler, TorchEnv
from pyforce.nn.observation import ObservationProcessor
from pyforce.nn.hidden import HiddenLayers
from pyforce.nn.action import ActionMapper
from pyforce.agents import A2CAgent
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

agent=A2CAgent(
    observation_processor,
    hidden_layers,
    action_mapper,
    save_path="./evals/a2c_example",
    value_lr=1e-3,
    policy_lr=1e-3
).to(device)

agent.train(env,episodes=1000,train_freq=256,eval_freq=50,render=True,gamma=.99,entropy_coef=.01)