#!/usr/bin/env python

#trunc8 did this

import matplotlib.pyplot as plt
import sys
import argparse
import epsilon_greedy, ucb, kl_ucb, \
    thompson_no_hint, thompson_with_hint

import numpy as np
import random
import math

def main():
    parser = argparse.ArgumentParser() # Initiate the parser
    parser.add_argument("--instance")
    parser.add_argument("--algorithm")
    parser.add_argument("--randomSeed")
    parser.add_argument("--epsilon")
    parser.add_argument("--horizon")
    args = parser.parse_args() # Read args from terminal
    args.epsilon = float(args.epsilon)
    args.horizon = int(args.horizon)
    args.randomSeed = int(args.randomSeed)

    random.seed(args.randomSeed)

    reward_means = []

    # File I/O
    instance_file = open(args.instance, "r")
    for line in instance_file:
        reward_means.append(float(line))
    instance_file.close()
    # File I/O ends

    num_of_arms = len(reward_means)
    REG = 0
    if args.algorithm == "epsilon-greedy":
        REG = epsilon_greedy.soln(reward_means, args.epsilon, args.horizon, num_of_arms)
    elif args.algorithm == "ucb":
        REG = ucb.soln(reward_means, args.horizon, num_of_arms)
    elif args.algorithm == "kl-ucb":
        REG = kl_ucb.soln(reward_means, args.horizon, num_of_arms)
    elif args.algorithm == "thompson-sampling":
        REG = thompson_no_hint.soln(reward_means, args.horizon, num_of_arms, args.randomSeed)
    elif args.algorithm == "thompson-sampling-with-hint":
        hint_ls = np.sort(reward_means)
        REG = thompson_with_hint.soln(reward_means, args.horizon, num_of_arms, hint_ls)
    print(f"{args.instance}, {args.algorithm}, {args.randomSeed}, {args.epsilon}, {args.horizon}, {REG:.3f}")

if __name__=='__main__':
    main()
