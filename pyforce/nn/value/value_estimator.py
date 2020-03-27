from torch import nn
import torch

class ValueEstimator(nn.Module):
  def __init__(self,n_input):
    super().__init__()
    self.v_out=nn.Linear(n_input,1)
  def forward(self,x):
    return self.v_out(x)