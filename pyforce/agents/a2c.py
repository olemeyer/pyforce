import torch
import gym
import numpy as np
from torch import nn
from .base import BaseAgent
from ..nn.value import ValueEstimator
import copy
import torch.nn.functional as F
from ..utils import discount


class A2CAgent(BaseAgent):
    def __init__(self,
                 observationprocessor,
                 hiddenlayers,
                 actionmapper,
                 policy_lr=1e-4,
                 value_lr=1e-4,
                 save_path=None):
        super().__init__(save_path)

        self.value_net = nn.Sequential(copy.deepcopy(observationprocessor),
                                       copy.deepcopy(hiddenlayers),
                                       ValueEstimator(hiddenlayers.n_output))

        self.policy_net = nn.Sequential(observationprocessor, hiddenlayers,
                                        actionmapper)

        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(),
                                                lr=value_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                                 lr=policy_lr)

    def forward(self, state):
        return self.policy_net(state)

    def get_action(self, state, eval, args):
        return self(state).sample(), {}

    def after_step(self, done, eval, args):
        if not eval:
            if self.env_steps % args["train_freq"] == 0 and len(
                    self.memory) > 0:

                self.write_scalar("batch/reward_mean",
                                  self.memory.reward.detach().mean().numpy())

                next_state_value = self.value_net({
                    k: self.memory.next_state[k][-1]
                    for k in self.memory.next_state
                })
                returns = discount(self.memory.reward, self.memory.done,
                                   args["gamma"], next_state_value)

                value = self.value_net(self.memory.state)
                value_loss = F.mse_loss(value, returns.detach())
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

                advantage = returns - value
                dist = self(self.memory.state)
                log_probs = dist.log_prob(self.memory.action)
                entropy = dist.entropy()

                policy_loss = -(log_probs * advantage.detach() +
                                args["entropy_coef"] * entropy).mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                
                self.write_scalar("loss/value",value_loss.detach().cpu().numpy())
                self.write_scalar("loss/policy",policy_loss.detach().cpu().numpy())
                self.memory.clear()

        if done and eval:
            self.write_scalar("eval/reward_mean",
                              self.eval_memory.reward.detach().mean().cpu().numpy(),step=self.eval_env_episodes)
            self.write_scalar("eval/reward_sum",
                              self.eval_memory.reward.detach().sum().cpu().numpy(),step=self.eval_env_episodes)
            self.eval_memory.clear()
