
import torch
import gym
import numpy as np
from torch import nn
from .base import BaseAgent
import copy
import torch.nn.functional as F
from ..nn.modules import DoubleQCritic, DeterministicActor
import copy

class TD3Agent(BaseAgent):
    def __init__(self,env,save_path=None, feature_size=256, critic_lr=1e-3, actor_lr=1e-3):
        super().__init__(save_path=save_path)

        self.critic=DoubleQCritic(env,feature_size)
        self.critic_target=copy.deepcopy(self.critic)

        self.actor=DeterministicActor(env,feature_size)
        self.actor_target=copy.deepcopy(self.actor)

        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)

        self.train_steps=0

    def forward(self,x):
        return self.actor(x)

    def get_action(self, x, eval, args):

        x=self(x)
        if not eval:
            if self.env_steps<args["warmup_steps"]:
                x={k:torch.rand_like(x[k]) for k in x}
            else:
                x={
                    k:(
                        x[k]+torch.randn_like(x[k])*args["exp_noise"]
                    ).clamp(0,1) for k in x
                }


        x={k:x[k].detach() for k in x}
        return x,{}

    def after_step(self, done, eval, args):
        if not eval:
            if self.env_steps % args["train_freq"] == 0 and len(self.memory) > args["warmup_steps"]:

                idx=np.random.choice(len(self.memory),args["batch_size"])

                batch_states={k:self.memory.state[k][idx] for k in self.memory.state}
                batch_actions={k:self.memory.action[k][idx] for k in self.memory.action}
                batch_next_states={k:self.memory.next_state[k][idx] for k in self.memory.next_state}
                batch_reward=self.memory.reward[idx]
                batch_done=self.memory.done[idx]
                
                with torch.no_grad():

                    # Select action according to policy and add clipped noise
                    noise={
                        k:(
                            torch.randn_like(batch_actions[k])*args["policy_noise"]
                            ).clamp(-args["policy_noise_clip"],args["policy_noise_clip"]) for k in batch_actions
                    }
                    
                    batch_next_action=self.actor_target(batch_next_states)
                    batch_next_action={
                        k:(batch_next_action[k]+noise[k]).clamp(0,1) for k in batch_next_action
                    }

                    # Compute the target Q value
                    target_Q1, target_Q2 = self.critic_target([batch_next_states, batch_next_action])
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_Q = batch_reward + (1-batch_done) * args["gamma"] * target_Q

                # Get current Q estimates
                current_Q1, current_Q2 = self.critic([batch_states, batch_actions])

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

                # Optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                self.write_scalar("loss/critic",critic_loss.detach().cpu())

                # Delayed policy updates
                if self.train_steps% args["policy_freq"] == 0:

                    # Compute actor losse
                    actor_loss = -self.critic.q1([batch_states, self.actor(batch_states)]).mean()
                    
                    # Optimize the actor 
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    self.write_scalar("loss/policy",actor_loss.detach().cpu())

                    # Update the frozen target models
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(args["tau"] * param.data + (1 - args["tau"]) * target_param.data)

                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(args["tau"]* param.data + (1 - args["tau"]) * target_param.data)

                self.train_steps+=1
                self.write_scalar("reward/batch",self.memory.reward[-args["train_freq"]:].mean().detach().cpu())

                for k in self.memory.data:
                    self.memory.data[k]={i:self.memory.data[k][i][-args["buffer_size"]:] for i in self.memory.data[k]}


        elif eval and done and len(self.memory) > args["warmup_steps"]:
            self.write_scalar("eval/reward_mean",
                              self.eval_memory.reward.detach().mean().cpu(),step=self.eval_env_episodes)
            self.write_scalar("eval/reward_sum",
                              self.eval_memory.reward.detach().sum().cpu(),step=self.eval_env_episodes)
            self.eval_memory.clear()