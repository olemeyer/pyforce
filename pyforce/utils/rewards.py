import torch

def discount(x,dones,gamma,last_next=0.):
  discounted=torch.zeros_like(x).to(x.device)
  r=torch.zeros(1).to(x.device)+last_next
  for i in reversed(range(len(discounted))):
    r=x[i]+(1-dones[i])*gamma*r
    discounted[i]=r

  return discounted

def generalized_advantage(gamma,tau,rewards,dones,values,last_next_value):
  next_values = torch.cat((values[1:], last_next_value.unsqueeze(0)))
  td=temporal_difference(gamma,rewards,dones,values,next_values)
  advantage=discount(td,dones,gamma*tau)
  return advantage

def temporal_difference(gamma, rewards, dones, values, next_values):
  return rewards+(1-dones)*gamma*next_values-values