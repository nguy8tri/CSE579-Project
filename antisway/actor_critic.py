import torch
from agents import GenericACAgent

class ActorCriticAgent(GenericACAgent):
    def update_actor(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        #========== TODO: start ==========
        # Implement the actor update
        # Compute the Q values of the action using self.critic(obs, action). In this case it is a single instead of
        # double Q function so you do not need to take a minimum.
        # The policy loss is the mean over the negative Q values i.e we want to maximize the Q values
        
        actor_Q = self.critic(obs, action)  # Action Value Function
        actor_loss = -actor_Q.mean()
        
        #========== TODO: end ==========
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item(), 0, 0
    
    def update_critic(self, obs, action, reward, next_obs, not_done_no_max):
        #========== TODO: start ==========
        # Train the single Q function:
        # Hint: Step 1: Compute current Q predictions using the obs and action and self.critic()
        # Hint: Step 2: Compute q targets using reward + critic_target * not_done_no_max for next_obs and
        # next_actions actions sampled from the current policy. Use torch.no_grad() for this step to disable
        # gradient flow to the critic_target and the actor.
        # Hint: Step 3: Compute Bellman error as mean squared error between q_predictions and q_targets
        criterion = torch.nn.MSELoss(reduction='mean')

        q_pred = self.critic(obs,action)

        with torch.no_grad():
            next_dist = self.actor(next_obs)
            next_act = next_dist.rsample()   
            q_next = self.critic_target(next_obs, next_act)
            q_target = reward + (q_next*not_done_no_max*self.discount)
        
        critic_loss = criterion(q_pred, q_target)

        #========== TODO: end ==========

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.item()
