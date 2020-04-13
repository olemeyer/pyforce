import torch
from torch import nn
import gym
import numpy as np

class ObservationProcessor(nn.Module):
  def __init__(self,env):
    super().__init__()
    for k in env.observation_space.spaces:
      space=env.observation_space.spaces[k]
      setattr(self,"_{}".format(k),nn.Linear(space.shape[-1],space.shape[-1]))

    #calc output size
    state=env.reset()
    state={k:state[k].cpu() for k in state}
    self.n_output=self(state).shape[-1]


  def forward(self,x):
    keys=sorted([k for k in x])
    x=[getattr(self,"_{}".format(k))(x[k]) for k in keys]
    x=torch.cat(x,-1)
    return x