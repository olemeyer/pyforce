import torch
import gym
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from ..memory import Memory
from tqdm.auto import tqdm

class BaseAgent(nn.Module):
  def __init__(self,save_path=None):
    super().__init__()
    self.memory=Memory()
    self.eval_memory=Memory()
    self.env_steps=nn.Parameter(torch.zeros(1)[0],requires_grad=False)
    self.env_episodes=nn.Parameter(torch.zeros(1)[0],requires_grad=False)
    self.eval_env_steps=nn.Parameter(torch.zeros(1)[0],requires_grad=False)
    self.eval_env_episodes=nn.Parameter(torch.zeros(1)[0],requires_grad=False)
    self.writer=None

    if save_path is not None:
      self.load(save_path)

  def load(self,save_path):
    self.writer=SummaryWriter(save_path, flush_secs=10)

  def write_scalar(self,tag,value,step=None):
    if self.writer is not None:
      step=self.env_steps if step is None else step
      self.writer.add_scalar(tag,value,step)
  
  def train(self,env,episodes=1000,eval_freq=None,eval_env=None,**kwargs):
    for episode in tqdm(range(episodes)):
      done=False
      state=env.reset()
      while not done:
        action,action_info=self.get_action(state,False,kwargs)
        next_state,reward,done,_=env.step(action)
        self.memory.append(state=state,action=action,next_state=next_state,reward=reward,done=done,**action_info)
        state=next_state
        done=done.max()==1
        self.env_steps+=1
        self.after_step(done,False,kwargs)
        
      self.env_episodes+=1

      if eval_freq is not None and self.env_episodes%eval_freq==0:
        self.eval(env if eval_env is None else eval_env,episodes=1,**kwargs)

  def eval(self,env,eval_episodes=10,render=False,**kwargs):
    for episode in range(eval_episodes):
      done=False
      state=env.reset()
      while not done:
        if render:
          env.render()
        action,action_info=self.get_action(state,True,kwargs)
        next_state,reward,done,_=env.step(action)
        self.eval_memory.append(state=state,action=action,next_state=next_state,reward=reward,done=done,**action_info)
        state=next_state
        done=done.max()==1
        self.eval_env_steps+=1
        self.after_step(done,True,kwargs)
      
      self.eval_env_episodes+=1
        

  def get_action(self,state,eval,args):
    raise NotImplementedError()

  def after_step(self,done,eval,args):
    raise NotImplementedError()