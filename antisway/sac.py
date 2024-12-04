import torch
import torch.nn.functional as F

from agents import GenericACAgent

from return_handler import return_handler

class SACAgent(GenericACAgent):
    def update_actor(self, obs):        
        
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)  # use .sum(-1, keepdim=True)?
        Q1, Q2 = self.critic(obs, action)

        q_vals = torch.min(Q1,Q2)

        actor_loss = (self.alpha * log_prob - q_vals).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the temperature parameter toachieve entropy close to the target entropy
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach())
        alpha_loss = alpha_loss.mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return actor_loss.item(), -log_prob.mean().item(), alpha_loss.item()
    
    def update_critic(self, obs, action, reward, next_obs, not_done_no_max):
        
        next_dist = self.actor(next_obs)
        next_action = next_dist.rsample()
        next_log_prob = next_dist.log_prob(next_action).sum(-1, keepdim=True)  # use .sum(-1, keepdim=True)?

        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_val = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
        target_Q = reward + (not_done_no_max * self.discount * target_val)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()