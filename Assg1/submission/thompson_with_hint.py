#trunc8 did this

import numpy as np
import random
import math

def pull_arm(reward_means, arm):
    '''
    Bernoulli distribution with uniform probability
    '''
    return (random.uniform(0,1) < reward_means[arm])

def soln(reward_means, horizon, num_of_arms, hint_ls, randomSeed):
    '''
    Here the parameter reward_means is in sorted order to be used to gain some hint.
    If multiple arms have highest empirical mean, use the index returned by argmax
    No Round Robin exploration
    '''
    np.random.seed(randomSeed)
    hint = np.min(hint_ls)
    # hint = 0
    beta_samples = np.empty(num_of_arms)
    arm_successes = np.zeros(num_of_arms)
    arm_fails = np.zeros(num_of_arms)
    REW = 0
    REG = 0
    for t in range(1, horizon+1):
        for a in range(num_of_arms):
            if t > horizon*0.75 and arm_successes[a]/(arm_successes[a]+arm_fails[a]+1) < hint and random.uniform(0,1) < 0.5:
              beta_samples[a] = 0
              continue
            beta_samples[a] = np.random.beta(arm_successes[a]+1, arm_fails[a]+1)
        arm = np.argmax(beta_samples)
        if pull_arm(reward_means, arm) == 1:
            arm_successes[arm] += 1
            REW += 1
        else:
            arm_fails[arm] += 1
    REG = horizon*max(reward_means) - REW
    return REG
