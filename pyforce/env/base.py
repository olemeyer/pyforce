import gym
import numpy as np

#Base class for all environment wrappers. 
#Implements forwarding of all essential methods of the OpenAI Gyms (reset,step and render). 
#Furthermore the original environment is kept and can be accessed from the wrapper.
class EnvWrapper(gym.Env):
  def __init__(self,env):
    self.env=env
    self.action_space=env.action_space
    self.observation_space=env.observation_space

  def reset(self,*args,**kwargs):
    return self.env.reset(*args,**kwargs)

  def render(self,*args,**kwargs):
    return self.env.render(*args,**kwargs)

  def step(self,*args,**kwargs):
    return self.env.step(*args,**kwargs)