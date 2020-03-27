import torch
from torch import nn
import gym
import numpy as np

class HiddenLayers(nn.Module):
  def __init__(self,n_input):
    super().__init__()
    self.n_hidden=n_input*8
    self.n_output=self.n_hidden
    self.hidden=nn.Sequential(
        nn.LayerNorm(n_input),
        nn.Linear(n_input,self.n_hidden),
        nn.ELU(),
        nn.LayerNorm(self.n_hidden),
        nn.Linear(self.n_hidden,self.n_hidden),
        nn.ELU(),
        nn.LayerNorm(self.n_hidden),
        nn.Linear(self.n_hidden,self.n_output),
        nn.ELU()
    )

  def forward(self,x):
    return self.hidden(x)