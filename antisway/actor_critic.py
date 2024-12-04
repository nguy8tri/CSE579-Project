import torch
from agents import GenericACAgent

class ActorCriticAgent(GenericACAgent):
    def update_actor(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        actor_Q = self.critic(obs, action)  # Action Value Function
        actor_loss = -actor_Q.mean()
        
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item(), 0, 0
    
    def update_critic(self, obs, action, reward, next_obs, not_done_no_max):
       
        criterion = torch.nn.MSELoss(reduction='mean')

        q_pred = self.critic(obs,action)

        with torch.no_grad():
            next_dist = self.actor(next_obs)
            next_act = next_dist.rsample()   
            q_next = self.critic_target(next_obs, next_act)
            q_target = reward + (q_next*not_done_no_max*self.discount)
        
        critic_loss = criterion(q_pred, q_target)

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.item()
