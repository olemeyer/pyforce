import gym
import numpy as np
from .base import EnvWrapper
import torch

#Converts states, actions, rewards and terminal flags to and from Pytorch variables.
class TorchEnv(EnvWrapper):
  def __init__(self,env,device="cpu"):
    super().__init__(env)
    self.device=device

  def reset(self):
    state=self.env.reset()
    state=self.to_tensor(state)
    return state

  def random_action(self):
    action=self.action_space.sample()
    action={k:self.to_tensor(action[k]) for k in action}
    return action

  def to_tensor(self,x):
    if isinstance(x,dict):
      x={k:self.to_tensor(x[k]) for k in x}
      return x
    if isinstance(x,(float,int,bool,np.bool_)):
      x=[x]
    return torch.FloatTensor(x).unsqueeze(0).to(self.device)

  def to_numpy(self,x):
    if torch.is_tensor(x):
      x=x.detach().cpu().numpy()[0]
    return x

  def to_int_if_discrete(self,x,k):
    if isinstance(self.action_space[k],gym.spaces.Discrete):
      if len(x.shape)>0:
        x=x[0]
      return int(x)
    return x

  def step(self,action):
    action={k:self.to_numpy(action[k]) for k in action}
    action={k:self.to_int_if_discrete(action[k],k) for k in action}
    state,reward,done,info=self.env.step(action)
    return self.to_tensor(state),self.to_tensor(reward),self.to_tensor(done),info

  def to(self,device):
    self.device=device
    return self