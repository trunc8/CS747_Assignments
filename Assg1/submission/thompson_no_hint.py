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
    If multiple arms have highest empirical mean, use the index returned by argmax
    No Round Robin exploration
    '''
    beta_samples = np.empty(num_of_arms)
    arm_successes = np.zeros(num_of_arms)
    arm_fails = np.zeros(num_of_arms)
    REW = 0
    REG = 0
    for t in range(1, horizon+1):
        for a in range(num_of_arms):
            beta_samples[a] = random.betavariate(arm_successes[a]+1, arm_fails[a]+1)
        arm = np.argmax(beta_samples)
        if pull_arm(reward_means, arm) == 1:
            arm_successes[a] += 1
            REW += 1
        else:
            arm_fails[a] += 1
    REG = horizon*max(reward_means) - REW
    return REG
