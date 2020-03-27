import torch
import gym
import numpy as np
from torch import nn
from .base import BaseAgent
from ..nn.value import ValueEstimator
import copy
import torch.nn.functional as F
from ..utils import discount, generalized_advantage
from ..agents import A2CAgent

class PPOAgent(A2CAgent):
    def after_step(self,done,eval,args):
        if not eval:
            if self.env_steps%args["train_freq"]==0 and len(self.memory)>0:

                self.write_scalar("batch/reward_mean",self.memory.reward.detach().mean().cpu().numpy())

                with torch.no_grad():
                    next_state_value=self.value_net({k:self.memory.next_state[k][-1] for k in self.memory.next_state})
                    returns=discount(self.memory.reward,self.memory.done,args["gamma"],next_state_value)

                    values=self.value_net(self.memory.state)

                    advantages=generalized_advantage(args["gamma"],args["tau"],self.memory.reward,self.memory.done,values,next_state_value)
                    advantages=(advantages-advantages.mean())/(advantages.std()+1e-6)
                    old_log_probs=self.policy_net(self.memory.state).log_prob(self.memory.action)

                for _ in range(args["n_steps"]):
                    idx=np.random.choice(advantages.shape[0],args["batch_size"])

                    batch_states={k:self.memory.state[k][idx] for k in self.memory.state}
                    batch_actions={k:self.memory.action[k][idx] for k in self.memory.action}
                    batch_returns=returns[idx]
                    batch_advantages=advantages[idx]
                    batch_old_log_probs=old_log_probs[idx]

                    batch_values=self.value_net(batch_states)
                    value_loss=F.mse_loss(batch_values,batch_returns)
                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    self.value_optimizer.step()

                    new_dist=self.policy_net(batch_states)
                    new_log_probs=new_dist.log_prob(batch_actions)
                    new_entropy=new_dist.entropy()

                    ratios=torch.exp(new_log_probs-batch_old_log_probs.detach())
                    obj=ratios*batch_advantages
                    obj_clip=torch.clamp(ratios,1-args["clip"],1+args["clip"])*batch_advantages
                    policy_loss=-(torch.min(obj,obj_clip)+args["entropy_coef"]*new_entropy).mean()
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
