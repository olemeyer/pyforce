from pyforce.env import wrap_openai_gym
from pyforce.nn import default_network_components
from pyforce.agents import TD3Agent
import gym
import torch

device="cuda:0" if torch.cuda.is_available() else "cpu"

env=wrap_openai_gym(gym.make("LunarLanderContinuous-v2"))


agent=TD3Agent(
    env,
    save_path="./evals/td3_example",
    critic_lr=1e-3,
    actor_lr=1e-3
).to(device)

agent.train(env,100000,train_freq=1,batch_size=100,policy_noise=0.,policy_noise_clip=.5,gamma=.99, policy_freq=2, tau=0.005,warmup_steps=10000,buffer_size=50000, exp_noise=.1,eval_freq=1, render=True, eval_episodes=1)