import gym
import numpy as np
from .base import EnvWrapper

#Scales the Action Space to a uniform range from 0 to 1
class _ActionScaler:
  def __init__(self,action_space):
    if isinstance(action_space,gym.spaces.Box):
      lb=action_space.low
      ub=action_space.high
      self.scale=ub-lb
      self.bias=lb
    
    if isinstance(action_space,gym.spaces.Box):
      ub=np.ones(action_space.shape)
      lb=np.zeros(action_space.shape)
      self.action_space=gym.spaces.Box(lb,ub)
    else:
      self.action_space=action_space
  
  def __call__(self,x):
    if isinstance(self.action_space,gym.spaces.Box):
      
      x=(x*self.scale)+self.bias
      return x
    else:
      return x
    

class ActionSpaceScaler(EnvWrapper):
  def __init__(self,env):
    super().__init__(env)
    self.scaler={k:_ActionScaler(env.action_space.spaces[k]) for k in env.action_space.spaces}
    self.action_space=gym.spaces.Dict({
        k:self.scaler[k].action_space for k in self.scaler
    })

  def step(self,action):
    action={k:self.scaler[k](action[k]) for k in self.scaler}
    return self.env.step(action)