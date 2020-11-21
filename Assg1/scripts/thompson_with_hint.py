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
    
    # Each column label symbolises hint_ls in that order
    # Each row label symbolises an arm (order is different and unknown from hint_ls)
    confidence = np.empty([num_of_arms, num_of_arms])
    confidence.fill(1.0/num_of_arms)

    REW = 0
    REG = 0
    for t in range(1, horizon+1):

        # Utilising hint to gain confidence since we now know the discrete values it could be
        arm = np.argmax(confidence[:,-1])

        if pull_arm(reward_means, arm) == 1:
            REW += 1
            for i in range(num_of_arms):
                confidence[arm,i] = hint_ls[i]*confidence[arm,i]
        else:
            for i in range(num_of_arms):
                confidence[arm,i] = (1 - hint_ls[i])*confidence[arm,i]
        normalising_sum = sum(confidence[arm,:])
        confidence[arm,:] = confidence[arm,:]/normalising_sum

    REG = horizon*max(reward_means) - REW
    return REG
