import numpy as np
import torch

class Memory():
  def __init__(self,device="cpu",data=None,ordered=True):
    self.device=device
    self.data=data if data is not None else {}
    self.ordered=ordered

  def clear(self):
    self.data={}

  def __len__(self):
    keys=self.keys()
    if len(keys)==0:
      return 0
    else:
      k=keys[0]
      if isinstance(self.data[k],dict):
        keys2=[k for k in self.data[k]]
        return self.data[k][keys2[0]].shape[0]
      else:
        return self.data[k].shape[0]

  def keys(self):
    keys=[k for k in self.data]
    return keys

  def append(self,**data):
    for k in data:
      if not isinstance(data[k],dict):
        data[k]={"_value":data[k]}
      new_data={i:data[k][i] for i in data[k]}
      if k in self.data:
        existing_data=self.data[k]
        new_data={i:torch.cat([existing_data[i],new_data[i]]) for i in new_data}
      self.data[k]=new_data

  def to(self,device):
    self.device=device
    for k in self.data:
      self.data[k]={i:self.data[k].to(self.device) for i in self.data[k]}
    return self

  def sample(self,n):
    k=list(self.data.keys())[0]
    max_i=len(self.data[k])
    idx=np.random.choice(max_i,n,replace=n>max_i)
    data={}
    for k in self.data:
      data[k]={i:self.data[k][i][idx] for i in self.data[k]}
    return Memory(device=self.device,data=data,ordered=False)

  def __getattr__(self,name):
    if name not in self.data:
      return []
    if len(self.data[name])==1 and "_value" in self.data[name]:
      return self.data[name]["_value"]
    return self.data[name]
