import matplotlib.pyplot as plt
import numpy as np

# Global lists for storing loss values
policy_losses = []
critic_losses = []
batch_rewards_pg = []
batch_rewards = []
velocities = []
angles = []

def return_handler(velocity = None, angle = None, policy_loss = None, critic_loss = None, batch_reward=None, batch_reward_pg=None, plot=False):
    # Record data if requested
    if policy_loss is not None:
        policy_losses.extend(policy_loss)
    
    if critic_loss is not None:
        critic_losses.extend(critic_loss)

    if batch_reward is not None:
        batch_rewards.extend(batch_reward)

    if batch_reward_pg is not None:
        batch_rewards_pg.extend(batch_reward_pg)

    if velocity is not None:
        velocities.append(velocity)
    
    if angle is not None:
        angles.append(angle)

    # Plot data if requested
    if plot:
        plt.figure(figsize=(10, 12))
        if velocities:
            # Plot
            plt.subplot(2, 1, 1)
            plt.plot(velocities, label='Velocity') 
            plt.xlabel('Training Steps')
            plt.ylabel('Cart Velocity')
            plt.title('Velocity over Time')
            plt.legend()
            ax = plt.gca()
            ax.set_xlim([0, len(velocities)])

        if angles:
            # Plot
            plt.subplot(2, 1, 2)
            plt.plot(angles, label='Angle')
            plt.xlabel('Training Steps')
            plt.ylabel('Pole Angle')
            plt.title('Pole Angle over Time')
            plt.legend()
            ax = plt.gca()
            ax.set_xlim([0, len(angles)])
        


        plt.figure(figsize=(10, 12))
        if batch_rewards:
            # Plot
            plt.subplot(3, 1, 1)
            plt.plot(batch_rewards, label='Batch Rewards')
            plt.xlabel('Training Steps')
            plt.ylabel('Rewards')
            plt.title('Batch Rewards over Time')
            plt.legend()
            ax = plt.gca()
            ax.set_xlim([0, len(batch_rewards)])
        elif batch_rewards_pg:
            plt.subplot(1, 1, 1)
            plt.plot(batch_rewards_pg, label='Batch Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Rewards')
            plt.title('Batch Rewards over Time')
            plt.legend()
            ax = plt.gca()
            ax.set_xlim([0, len(batch_rewards_pg)])

        if policy_losses:
            # Plot 
            plt.subplot(3, 1, 2)
            plt.plot(policy_losses, label='Actor Losses')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Actor Loss over Time')
            plt.legend()
            ax = plt.gca()
            ax.set_xlim([0, len(policy_losses)])

        if critic_losses:
            # Plot
            plt.subplot(3, 1, 3)
            plt.plot(critic_losses, label='Critic Losses')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Critic Loss over Time')
            plt.legend()
            ax = plt.gca()
            ax.set_xlim([0, len(critic_losses)])


        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.35)
        
        plt.show()