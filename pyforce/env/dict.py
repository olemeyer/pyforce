import gym
import numpy as np
from .base import EnvWrapper

#This wrapper ensures that the Observation Space and Action Space is of type gym.spaces.Dict. 
#This allows state and action spaces of any complexity to be handeled in uniform manner.
class DictEnv(EnvWrapper):
  
  def __init__(self,env):
    super().__init__(env)
    self.replace_obs=not isinstance(env.observation_space,gym.spaces.Dict)
    self.replace_action=not isinstance(env.action_space,gym.spaces.Dict)

    if self.replace_obs:
      self.observation_space=gym.spaces.Dict({
          "state":env.observation_space
      })

    if self.replace_action:
      self.action_space=gym.spaces.Dict({
          "action":env.action_space
      })

  def _state(self,state):
    if self.replace_obs:
        return {
            "state":state
        }
    return state

  def reset(self,*args,**kwargs):
    state=self.env.reset(*args,**kwargs)
    return self._state(state)

  def _action(self,action):
    if self.replace_action:
      action=action["action"]

    return action

  def step(self,action):
    action=self._action(action)
    state,reward,done,info=self.env.step(action)
    return self._state(state),reward,done,info