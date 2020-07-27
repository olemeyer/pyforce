from torch import nn
import torch
import gym
import numpy as np

class ActionProcessor(nn.Module):
  def __init__(self,env):
    super().__init__()

    self.n_output=0
    for k in env.action_space.spaces:
      space=env.action_space.spaces[k]
      setattr(self,"_{}".format(k),nn.Linear(space.shape[-1],space.shape[-1]))
      self.n_output+=np.prod(space.shape)


  def forward(self,x):
    keys=sorted([k for k in x])
    x=[getattr(self,"_{}".format(k))(x[k]) for k in keys]
    x=torch.cat(x,-1)
    return x