#trunc8 did this

import numpy as np
import random
import math

def pull_arm(reward_means, arm):
    '''
    Bernoulli distribution with uniform probability
    '''
    return (random.uniform(0,1) < reward_means[arm])

def soln(reward_means, horizon, num_of_arms):
    '''
    Initialise pulls so far with 1's for each arm
    If multiple arms have highest empirical mean, use the index returned by argmax
    One cycle of Round Robin exploration
    '''
    empirical_means = np.empty(num_of_arms)
    arm_ucb = np.empty(num_of_arms)
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

        for a in range(num_of_arms):
            arm_ucb[a] = empirical_means[a] + math.sqrt(2*math.log(t)/arm_pulls[a])
        arm = np.argmax(arm_ucb)
        num_of_pulls = arm_pulls[arm]
        run_reward = pull_arm(reward_means, arm)
        REW += run_reward
        empirical_means[arm] = (empirical_means[arm]*num_of_pulls
            + run_reward)/(num_of_pulls + 1)
        arm_pulls[arm] = num_of_pulls + 1
    REG = horizon*max(reward_means) - REW
    return REG