import torch
from torch import nn
import gym
import numpy as np

class DistributionDict(object):
  def __init__(self,distributions):
    self.distributions=distributions

  def sample(self):
    x={k:self.distributions[k].sample() for k in self.distributions}
    #x={k:x[k]*2-1 if isinstance(self.distributions[k],BetaDistributionAction) else x[k] for k in x}
    return x

  def best(self):
    x={k:self.distributions[k].mean if isinstance(self.distributions[k],torch.distributions.Beta) else torch.argmax(self.distributions[k].probs).reshape(-1) for k in self.distributions}
    #x={k:x[k]*2-1 if isinstance(self.distributions[k],torch.distributions.Beta) else x[k] for k in x}
    return x

  def entropy(self):
    x={k:self.distributions[k].entropy().unsqueeze(0) for k in self.distributions}
    x=[x[k] for k in x]
    x=torch.cat(x,0) # action x batch x logprob
    if len(x.shape)==3:
      x=x.permute(1,2,0)
      x=x.mean(-1)
    else:
      x=x.permute(1,0)
    return x

  def log_prob(self,x):
    #x={k:(x[k]+1)/2 for k in x}
    x={k:self.distributions[k].log_prob(x[k]).unsqueeze(0) for k in self.distributions}
    x=[x[k] for k in x]
    x=torch.cat(x,0) # action x batch x logprob or action x batch
    if len(x.shape)==3:
      x=x.permute(1,2,0)
      x=x.mean(-1)
    else:
      x=x.permute(1,0)
    return x

class CategoricalDistributionAction(nn.Module):
  def __init__(self,n_input,n_action):
    super().__init__()

    self.out_p=nn.Sequential(
        nn.Linear(n_input,n_action),
        nn.Softmax(dim=-1)
    )

  def forward(self,x):
    x_p=self.out_p(x)
    return torch.distributions.Categorical(x_p)

class BetaDistributionAction(nn.Module):
  def __init__(self,n_input,n_action):
      super().__init__()
      
      self.out_a=nn.Sequential(
          nn.Linear(n_input,n_action),
          nn.Softplus()
      )
      self.out_b=nn.Sequential(
          nn.Linear(n_input,n_action),
          nn.Softplus()
      )

  def forward(self,x):
    x_a=self.out_a(x)+1
    x_b=self.out_b(x)+1 

    return torch.distributions.Beta(x_a,x_b)

class DeterministicContinuousAction(nn.Module):
    def __init__(self,n_input,n_action):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_input,n_action),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.net(x)


class ActionMapper(nn.Module):
  def __init__(self,env,n_input,deterministic=False):
    super().__init__()
    self.deterministic=deterministic
    keys=[]
    for k in env.action_space.spaces:
      space=env.action_space.spaces[k]
      
      key="_{}".format(k)
      if isinstance(space,gym.spaces.Discrete):
        n_action=space.n
        if deterministic:
          raise NotImplementedError()
        else:
          setattr(self,key,CategoricalDistributionAction(n_input,n_action))
      else:
        n_action=space.shape[-1]
        if deterministic:
          setattr(self,key,DeterministicContinuousAction(n_input,n_action))
        else:
          setattr(self,key,BetaDistributionAction(n_input,n_action))
        
  def forward(self,x):
    x_={}
    for k in self._modules:
      x_[k[1:]]=getattr(self,k)(x)
    
    if self.deterministic:
      return x_
    return DistributionDict(x_)

