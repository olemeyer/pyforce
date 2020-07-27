from torch import nn
import torch
from ..observation import ObservationProcessor
from ..action import ActionProcessor

class QCritic(nn.Module):
    def __init__(self,env,feature_size):
        super().__init__()
        self.obs_processor=ObservationProcessor(env)
        self.act_processor=ActionProcessor(env)
        self.net=nn.Sequential(
            nn.Linear(self.obs_processor.n_output+self.act_processor.n_output,feature_size),
            nn.SELU(),
            nn.Linear(feature_size,feature_size),
            nn.SELU(),
            nn.Linear(feature_size,1)
        )

    def forward(self,x):
        x_obs,x_act=x
        x_obs=self.obs_processor(x_obs)
        x_act=self.act_processor(x_act)
        x=torch.cat([x_obs,x_act],-1)
        return self.net(x)

class DoubleQCritic(nn.Module):
    def __init__(self,env,feature_size):
        super().__init__()
        self.q1=QCritic(env,feature_size)
        self.q2=QCritic(env,feature_size)

    def forward(self,x):
        return self.q1(x),self.q2(x)