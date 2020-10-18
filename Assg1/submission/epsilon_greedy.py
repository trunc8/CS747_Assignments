#trunc8 did this

import numpy as np
import random
import math

def pull_arm(reward_means, arm):
    '''
    Bernoulli distribution with uniform probability
    '''
    return (random.uniform(0,1) < reward_means[arm])

def soln(reward_means, epsilon, horizon, num_of_arms):
    '''
    If multiple arms have highest empirical mean, use the index returned by argmax
    One cycle of round robin exploration. Thus initialise pulls so far with 1's for each arm.
    '''
    empirical_means = np.empty(num_of_arms)
    arm_pulls = np.ones(num_of_arms)
    REW = 0
    REG = 0
    for t in range(1, horizon+1):
        # One cycle of Round Robin
        if t <= num_of_arms:
            arm = t-1
            run_reward = pull_arm(reward_means, arm)
            REW += run_reward
            empirical_means[arm] = run_reward
            continue

        # Remaining is normal epsilon greedy
        if random.uniform(0,1) < epsilon: # Exploration
            arm = random.randint(0, num_of_arms-1)
        else: # Exploitation
            arm = np.argmax(empirical_means)
        num_of_pulls = arm_pulls[arm]
        run_reward = pull_arm(reward_means, arm)
        REW += run_reward
        empirical_means[arm] = (empirical_means[arm]*num_of_pulls
            + run_reward)/(num_of_pulls + 1)
        arm_pulls[arm] = num_of_pulls + 1
    REG = horizon*max(reward_means) - REW
    return REG