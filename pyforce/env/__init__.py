from .action_scaler import ActionSpaceScaler
from .dict import DictEnv
from .torch import TorchEnv

def wrap_openai_gym(env):
    return TorchEnv(ActionSpaceScaler(DictEnv(env)))