import torch
import torch.nn.functional as F

from agents import GenericACAgent


class SACAgent(GenericACAgent):
    def update_actor(self, obs):        
        #========== TODO: start ==========
        # Sample actions and the log_prob of the actions from the actor given obs. Hint: This is the same as AC agent.
        # Get the two Q values from the double Q function critic and take the minimum value. Then calculate the actor loss which
        # is defined by self.alpha * log_prob - actor_Q. Make sure that gradient does not flow through the alpha paramater. 
        
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)  # use .sum(-1, keepdim=True)?
        Q1, Q2 = self.critic(obs, action)

        q_vals = torch.min(Q1,Q2)

        actor_loss = (self.alpha * log_prob - q_vals).mean()

        #========== TODO: end ==========
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
        #========== TODO: start ==========
        # Train the double Q function:
        # Hint step 1: Sample the next_action and log_prob of the next action using the self.actor and the next_obs. Use the code 
        # below in update_actor as a reference on how to do this
        
        next_dist = self.actor(next_obs)
        next_action = next_dist.rsample()
        next_log_prob = next_dist.log_prob(next_action).sum(-1, keepdim=True)  # use .sum(-1, keepdim=True)?

        # Hint step 2: Sample the two target Q values from the critic_target using next_obs and the sampled next_action. 
        # Calculate the target value by taking the min of the values and then subtracting self.alpha * log_prob
        # The target Q is the reward + (not_done_no_max * discount * target_value)

        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_val = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
        target_Q = reward + (not_done_no_max * self.discount * target_val)
        
        # Hint step 3:
        # Sample the current Q1 and Q2 values of the current state using the critic, and regress onto the target Q.
        # The loss is mse(Q1, targetQ) + mse(Q2 + target Q)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        #========== TODO: end ==========
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()