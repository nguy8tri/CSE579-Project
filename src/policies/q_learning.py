import torch
from torch.nn.functional import mse_loss
from common.policy_defs import GenericACAgent


class ActorCriticAgent(GenericACAgent):
    def update_actor(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()

        q_values = self.critic(obs, action)
        actor_loss = -torch.mean(q_values)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item(), 0, 0

    def update_critic(self, obs, action, reward, next_obs, not_done_no_max):
        q_hat = self.critic(obs, action)
        q = None
        with torch.no_grad():
            next_action = self.actor(next_obs).rsample()
            next_q = self.critic_target(next_obs, next_action)
            q = reward + not_done_no_max * self.discount * next_q
        critic_loss = mse_loss(q_hat, q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.item()
