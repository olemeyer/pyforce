  
<p  align="center"><img  src="https://docs.google.com/drawings/d/e/2PACX-1vQTqWkqIzSlT3zldytD8L0kj6MZVpE_5ZslrDAvMhLEG-anipK2UPJuHm3ImGhVVTVYiZrsbTlKf3Yo/pub?w=756&h=265"  height="200px"  /></p>

  

[![Status](https://img.shields.io/badge/status-active-success.svg)]()

  
  

  

</div>

  

  

---

  

  

#Todo

  

  

## 🧐 About <a name = "about"></a>

  

  

#Todo

  

  

## 🏁 Getting Started <a name = "getting_started"></a>

  

  

    pip install pyforce-rl

  
  

## 🎈 Usage <a name="usage"></a>

  

```python
from pyforce.env import DictEnv, ActionSpaceScaler, TorchEnv
from pyforce.nn.observation import ObservationProcessor
from pyforce.nn.hidden import HiddenLayers
from pyforce.nn.action import ActionMapper
from pyforce.agents import PPOAgent
import gym
import torch

device="cuda:0" if torch.cuda.is_available() else "cpu"

env=gym.make("LunarLanderContinuous-v2")
env=DictEnv(env)
env=ActionSpaceScaler(env)
env=TorchEnv(env).to(device)

observation_processor=ObservationProcessor(env)
hidden_layers=HiddenLayers(observation_processor.n_output)
action_mapper=ActionMapper(env,hidden_layers.n_output)

agent=PPOAgent(
	observation_processor,
	hidden_layers,
	action_mapper,
	save_path="./evals/ppo_example",
	value_lr=5e-4,
	policy_lr=5e-4
).to(device)

agent.train(env,episodes=1000,train_freq=2048,eval_freq=50,render=True, batch_size=128,gamma=.99,tau=.95,clip=.2,n_steps=32,entropy_coef=.01)
```
  

  

## 🚀 Implement custom RL Agents <a name = "deployment"></a>



```python
from pyforce.agents.base import BaseAgent
from torch import nn
  
class  MyAgent(BaseAgent):

def  __init__(self,observationprocessor,hiddenlayers,actionmapper,save_path=None):

	super().__init__(save_path)

	self.policy_net = nn.Sequential(observationprocessor, hiddenlayers, actionmapper)
	self.value_net = ...

def  forward(self, state):
	return  self.policy_net(state)

def  get_action(self, state, eval, args):
	#return action + possible additional information to be stored in the memory
	return  self(state).sample(), {} 

def  after_step(self, done, eval, args):
	if  not  eval:
		if  self.env_steps % args["train_freq"] == 0 and len(self.memory) > 0:
			#do training

	if done and eval:
		#do evaluation
```
  

## ⛏️ Built Using <a name = "built_using"></a>

  

  

-  [PyTorch](https://pytorch.com/) - ML Framework

  

-  [OpenAI Gym](https://gym.openai.com/) - Environment API

  

-  [NumPy](https://numpy.org/) - Numerical calculations outside PyTorch

  

<!--

  

## ✍️ Authors <a name = "authors"></a>

  

  

-  [@olemeyer](https://github.com/olemeyer)

  

  

See also the list of [contributors](https://github.com/kylelobo/The-Documentation-Compendium/contributors) who participated in this project.

  
  
  

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

  

  

-  [Cherry-RL](http://cherry-rl.net/) & [Keras-RL](https://keras-rl.readthedocs.io/en/latest/) for Inspiration

-->
