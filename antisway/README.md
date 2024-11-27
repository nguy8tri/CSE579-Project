# This is a modified version of homework-2

## Setup and Installation
    This works with the homework 1 and 2 conda environments: cse579a1

## Environment Parameters
    The custom gym environments can be found in custom_envs/envs/
    Parameters for the cart pendulum including the reward function can be chanaged there. 
    The environments xml file (cart_pendulum.xml) including mass, friction, 
    and damping values can be found within custom_envs/envs/assets

## Training
    python main.py  --task pg --env cartpendulum
    python main.py  --task actor_critic --env cartpendulum
    python main.py  --task sac --env cartpendulum
       
    
## Evaluation
    python main.py  --task pg --env cartpendulum --test
    python main.py  --task actor_critic --test --env cartpendulum
    python main.py  --task sac --test --env cartpendulum
    

