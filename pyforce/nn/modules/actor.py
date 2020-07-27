from torch import nn
from ..observation import ObservationProcessor
from ..action import ActionMapper

class DeterministicActor(nn.Module):
    def __init__(self,env,feature_size):
        super().__init__()
        obs_processor=ObservationProcessor(env)
        self.net=nn.Sequential(
            obs_processor,
            nn.Linear(obs_processor.n_output,feature_size),
            nn.SELU(),
            nn.Linear(feature_size,feature_size),
            nn.SELU(),
            ActionMapper(env,feature_size, deterministic=True)
        )

    def forward(self,x):
        return self.net(x)