from .observation import ObservationProcessor
from .hidden import HiddenLayers
from .action import ActionMapper

def default_network_components(env):
    obs_processor=ObservationProcessor(env)
    hidden_layers=HiddenLayers(obs_processor.n_output)
    action_mapper=ActionMapper(env,hidden_layers.n_output)
    return obs_processor, hidden_layers, action_mapper