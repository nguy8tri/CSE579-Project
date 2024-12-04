import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from ..common.policy_defs import GenericACAgent
from ..utils.layers import eval_mode
from ..utils.rollout import rollout


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


class SACAgent(GenericACAgent):
    def update_actor(self, obs):
        # ========== TODO: start ==========
        # Sample actions and the log_prob of the actions from the actor given obs. Hint: This is the same as AC agent.
        # Get the two Q values from the double Q function critic and take the minimum value. Then calculate the actor loss which
        # is defined by self.alpha * log_prob - actor_Q. Make sure that gradient does not flow through the alpha paramater.

        dist = self.actor(obs)
        acs = dist.rsample()
        l_probs = dist.log_prob(acs)
        q_vals = torch.min(*self.critic(obs, acs))

        actor_loss = -torch.mean(q_vals - self.alpha.detach() * l_probs)

        # ========== TODO: end ==========
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the temperature parameter toachieve entropy close to the target entropy
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = self.alpha * (-l_probs - self.target_entropy).detach()
        alpha_loss = alpha_loss.mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return actor_loss.item(), -l_probs.mean().item(), alpha_loss.item()

    def update_critic(self, obs, action, reward, next_obs, not_done_no_max):
        # ========== TODO: start ==========
        # Train the double Q function:
        # Hint step 1: Sample the next_action and log_prob of the next action using the self.actor and the next_obs. Use the code
        # below in update_actor as a reference on how to do this

        # Hint step 2: Sample the two target Q values from the critic_target using next_obs and the sampled next_action.
        # Calculate the target value by taking the min of the values and then subtracting self.alpha * log_prob
        # The target Q is the reward + (not_done_no_max * discount * target_value)

        # Hint step 3:
        # Sample the current Q1 and Q2 values of the current state using the critic, and regress onto the target Q.
        # The loss is mse(Q1, targetQ) + mse(Q2 + target Q)

        next_ac_dist = self.actor(next_obs)
        next_ac = next_ac_dist.rsample()
        next_l_prob = next_ac_dist.log_prob(next_ac).sum(-1, keepdim=True)
        q = None
        with torch.no_grad():
            v = (
                torch.min(*self.critic_target(next_obs, next_ac))
                - self.alpha.detach() * next_l_prob
            )
            q = reward + (not_done_no_max * self.discount * v)
        q_1_hat, q_2_hat = self.critic(obs, action)
        critic_loss = F.mse_loss(q_1_hat, q) + F.mse_loss(q_2_hat, q)

        # ========== TODO: end ==========
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()


class ActorCriticTrainer:
    def __init__(
        self,
        agent,
        env,
        replay_buffer,
        num_train_steps=100_000,
        num_seed_steps=5000,
        eval_frequency=5000,
        num_eval_episodes=10,
    ):
        self.agent = agent
        self.env = env
        self.replay_buffer = replay_buffer
        self.num_train_steps = num_train_steps
        self.num_seed_steps = num_seed_steps
        self.eval_frequency = eval_frequency
        self.num_eval_episodes = num_eval_episodes

    def train_agent(self):
        """
        Generic training loop for an agent. It runs num_seed_steps of random exploration and then does the training loop.
        In the training loop, it samples an action, takes an environment step, and then updates the agent with a sampled batch
        from the replay buffer. It also evaluates the agent periodically.
        """
        episode, episode_reward, done = 0, 0, True
        step = 0
        since_last_eval = 0
        actor_loss = []
        critic_loss = []
        batch_reward = []
        while step < self.num_train_steps:
            if done:
                # evaluate agent periodically
                if since_last_eval > self.eval_frequency:
                    # Note the step here will fluctuate as it waits until terminaton to evaluate, but this is ok.
                    self.evaluate_agent(step)
                    since_last_eval = 0

                obs = self.env.reset()[0]
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
            # sample action for data collection
            if step < self.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
                since_last_eval += 1

            # run training update
            if step >= self.num_seed_steps:
                result = self.agent.update(self.replay_buffer)
                # add the policy and critic loss to the list
                actor_loss.append(result[2])
                critic_loss.append(result[1])
                batch_reward.append(result[0])
                if result is not None:
                    # tuple is train_batch_reward, critic_loss, actor_loss, actor_entropy, alpha_loss
                    if step % 5000 == 0:
                        # round these tensors to 4 decimal places
                        result = tuple(map(lambda x: round(x, 4), result))
                        print(
                            f"step {step}, batch_r {result[0]}, critic_l {result[1]}, actor_l {result[2]}"
                            f", actor_ent {result[3]}, alpha_l {result[4]}"
                        )
                        for param in self.agent.actor.parameters():
                            print(param[0])
                            break

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # allow infinite bootstrap
            done = float(done)
            done_no_max = (
                0 if episode_step + 1 == self.env.spec.max_episode_steps else done
            )
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            step += 1
        return actor_loss, critic_loss, batch_reward

    def evaluate_agent(self, step, verbose=False):
        average_episode_reward = 0
        av_ep_ln = 0
        for _ in range(self.num_eval_episodes):
            result = rollout(self.env, self.agent)
            ep_ln = len(result["rewards"])
            episode_reward = np.mean(result["observations"][:, -3])
            average_episode_reward += episode_reward
            av_ep_ln += ep_ln
            if verbose:
                print(f"eval episode reward {episode_reward}, episode length {ep_ln}")
        average_episode_reward /= self.num_eval_episodes
        av_ep_ln /= self.num_eval_episodes
        print(
            f"eval step {step}, average episode reward {average_episode_reward}, average episode length {av_ep_ln}"
        )
