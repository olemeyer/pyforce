import torch

def discount(values,dones,gamma,next_values=None):
  discounted=torch.zeros_like(values).to(values.device)
  r=torch.zeros(1).to(values.device)
  if next_values is not None:
    r=r+next_values[-1]
  for i in reversed(range(len(discounted))):
    r=values[i]+(1-dones[i])*gamma*r
    discounted[i]=r

  return discounted

def generalized_advantage(gamma,tau,rewards,dones,values,next_values):
  td=temporal_difference(gamma,rewards,dones,values,next_values)
  advantage=discount(td,dones,gamma*tau)
  return advantage

def temporal_difference(gamma, rewards, dones, values, next_values):
  return rewards+(1-dones)*gamma*next_values-values