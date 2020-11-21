#trunc8 did this

import numpy as np
import random
import math

def pull_arm(reward_means, arm):
    '''
    Bernoulli distribution with uniform probability
    '''
    return (random.uniform(0,1) < reward_means[arm])

def f(p,q):
    if q == 0 and p == 0:
        return 0
    if q == 0 or q == 1:
        print("invalid q encountered")
        return math.nan
    return p*math.log(q) + (1-p)*math.log(1-q)
    
def find_q_in_klucb(p, u, t):
    '''
    Rearranged inequality in q s.t. q terms are on one side
    Other side turns out to be f(p,t) = K2
    q belongs to interval [p,1]
    '''
    if p == 1: # q is sandwiched in interval
        return 1
    c = 3
    K1 = math.log(t) + c*math.log( math.log(t) )
    K2 = 0
    if p == 0:
        K2 = -K1/u
    else:
        K2 = p*math.log(p) + (1-p)*math.log(1-p) - K1/u
    lower = p
    upper = p
    q = p
    power = 1
    while (1):
        if (q+math.pow(2,-power) >= 1):
            power += 1
            continue
        else:
            q += math.pow(2,-power)
        # print('before', end=' ')
        # val = f(p,q)
        # print('after')
        # if val==-1:
            # print(f"what happened, p:{p} q:{q}")
        if (f(p,q) < K2):
            upper = q
            break
    accuracy = 1e-3
    # print(f"\tPositive: {f(p,lower)-K2:3f}\tNegative: {f(p,upper)-K2:3f}\tlower:{lower:3f}, upper:{upper:3f}")
    
    # Finding q by binary search
    while (1):
        mid = (lower+upper)/2
        if abs( f(p,mid)-K2 ) < accuracy:
            # print( f(p,mid) - K2)
            q = mid
            break
        elif f(p,mid)-K2 < 0:
            upper = mid
        else:
            lower = mid

    return q

def soln(reward_means, horizon, num_of_arms):
    '''
    Initialise pulls so far with 1's for each arm
    If multiple arms have highest empirical mean, use the index returned by argmax
    One cycle of Round Robin exploration
    '''
    empirical_means = np.empty(num_of_arms)
    arm_kl_ucb = np.empty(num_of_arms)
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
            arm_kl_ucb[a] = find_q_in_klucb(empirical_means[a], arm_pulls[a], t)
            # I don't have to worry about t=1 as Round Robin will ensure
            # that doesn't happen
        arm = np.argmax(arm_kl_ucb)
        num_of_pulls = arm_pulls[arm]
        run_reward = pull_arm(reward_means, arm)
        REW += run_reward
        empirical_means[arm] = (empirical_means[arm]*num_of_pulls
            + run_reward)/(num_of_pulls + 1)
        arm_pulls[arm] = num_of_pulls + 1
    REG = horizon*max(reward_means) - REW
    return REG