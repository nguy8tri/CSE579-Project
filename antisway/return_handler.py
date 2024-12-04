import matplotlib.pyplot as plt
import numpy as np

# Global lists for storing loss values

# Training Returns
policy_losses = []
critic_losses = []
batch_rewards_pg = []
batch_rewards = []

# Simulation Returns
velocities_rl = []
angles_rl = []
actions_rl = []

velocities_ct = []
angles_ct = []
actions_ct = []

targets = []
ct_offset_array = np.zeros(250)


def return_handler(policy_loss = None, 
                   critic_loss = None, 
                   batch_reward=None, 
                   batch_reward_pg=None, 
                   action_rl = None, 
                   velocity_rl = None, 
                   angle_rl = None,
                   action_ct = None, 
                   velocity_ct = None, 
                   angle_ct = None, 
                   target = None,   
                   plot = False):

    # Record data if requested

    # Training Returns
    if policy_loss is not None: policy_losses.extend(policy_loss)
    if critic_loss is not None: critic_losses.extend(critic_loss)
    if batch_reward is not None: batch_rewards.extend(batch_reward)
    if batch_reward_pg is not None: batch_rewards_pg.extend(batch_reward_pg)
        
    # Simulation Returns RL
    if velocity_rl is not None: velocities_rl.append(velocity_rl)
    if angle_rl is not None: angles_rl.append(angle_rl)
    if action_rl is not None: actions_rl.append(action_rl)
        
    # Simulation Returns CT
    if velocity_ct is not None: velocities_ct.extend(velocity_ct)
    if angle_ct is not None: angles_ct.extend(angle_ct)
    if action_ct is not None: actions_ct.extend(action_ct)

    if target is not None: targets.extend(target)

    # Plot data if requested
    if plot:
        if velocities_rl or angles_rl:
            plt.figure(figsize=(10, 12))
            
        if velocities_rl:
            plt.subplot(2, 1, 1)
            if velocities_ct: 
                plt.plot(targets[249:], label='Target Velocity')
                plt.plot(velocities_ct[249:], label='Velocity CT', linestyle='dashed')
                velocities_rl[0:0] = ct_offset_array
            plt.plot(velocities_rl, label='Velocity RL') 
            plt.xlabel('Step (ms)')
            plt.ylabel('Cart Velocity (m/s)')
            plt.title('Velocity over Time')
            plt.legend()
            plt.grid(True)
            ax = plt.gca()
            ax.set_xlim([0, len(velocities_rl)])

        if angles_rl:
            plt.subplot(2, 1, 2)
            if angles_ct: 
                plt.plot(angles_ct[249:], label='Angle CT', linestyle='dashed')
                angles_rl[0:0] = ct_offset_array
            plt.plot(angles_rl, label='Angle RL')
            plt.xlabel('Step (ms)')
            plt.ylabel('Pole Angle (rad)')
            plt.title('Pole Angle vs Time')
            plt.legend()
            plt.grid(True)
            ax = plt.gca()
            ax.set_xlim([0, len(angles_rl)])

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.35)

        '''if actions_rl:
            plt.figure(figsize=(10, 8))
            if actions_ct:
                plt.plot(actions_ct, label='Action CT', linestyle='dashed')
                actions_rl[0:0] = ct_offset
            plt.plot(actions_rl, label='Action')
            plt.xlabel('Step (ms)')
            plt.ylabel('Force (N)')
            plt.title('Motor Force (Before Gearing) vs Time')
            plt.legend()
            plt.grid(True)
            ax = plt.gca()
            ax.set_xlim([0, len(actions_rl)])

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.35)'''



        if  batch_rewards or batch_rewards_pg or policy_losses or critic_losses:
            plt.figure(figsize=(10, 12))
        if batch_rewards:
            plt.subplot(3, 1, 1)
            plt.plot(batch_rewards, label='Batch Rewards')
            plt.xlabel('Training Steps')
            plt.ylabel('Rewards')
            plt.title('Batch Rewards over Time')
            plt.legend()
            plt.grid(True)
            ax = plt.gca()
            ax.set_xlim([0, len(batch_rewards)])

        elif batch_rewards_pg:
            plt.subplot(1, 1, 1)
            plt.plot(batch_rewards_pg, label='Batch Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Rewards')
            plt.title('Batch Rewards over Time')
            plt.legend()
            plt.grid(True)
            ax = plt.gca()
            ax.set_xlim([0, len(batch_rewards_pg)])

        if policy_losses:
            plt.subplot(3, 1, 2)
            plt.plot(policy_losses, label='Actor Losses')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Actor Loss over Time')
            plt.legend()
            plt.grid(True)
            ax = plt.gca()
            ax.set_xlim([0, len(policy_losses)])

        if critic_losses:
            plt.subplot(3, 1, 3)
            plt.plot(critic_losses, label='Critic Losses')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Critic Loss over Time')
            plt.legend()
            plt.grid(True)
            ax = plt.gca()
            ax.set_xlim([0, len(critic_losses)])


        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.35)
        
        plt.show()