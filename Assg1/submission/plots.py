#!/usr/bin/env python

# trunc8 did this

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n")
args = parser.parse_args()
f = open(f"outputDataT{args.n}.txt", "r")

instances = ["../instances/i-1.txt", "../instances/i-2.txt", "../instances/i-3.txt"]
if args.n == '1':
    algorithms = ["epsilon-greedy", "ucb", "kl-ucb", "thompson-sampling"]
elif args.n == '2':
    algorithms = ["thompson-sampling", "thompson-sampling-with-hint"]
horizons = [100, 400, 1600, 6400, 25600, 102400]

Expt = {}
for instance in instances:
    Expt[instance] = {}
    for algorithm in algorithms:
        Expt[instance][algorithm] = {}
        for horizon in horizons:
            Expt[instance][algorithm][horizon] = 0
for line in f:
    instance, algorithm, _, _, horizon, REG = line.split(", ")
    Expt[instance][algorithm][int(horizon)] += float(REG)/50

for instance in instances:
    # plt.figure()
    for algorithm in algorithms:
        lists = sorted(Expt[instance][algorithm].items())
        x, y = zip(*lists)
        plt.plot(x, y, label = algorithm)
    plt.xscale("log")
    plt.xlabel("Horizon")
    plt.ylabel("Cumulative Expected Regret")
    plt.title(f"{instance}")
    plt.legend()
    # plt.show()
    plt.savefig(f"../plots/T{args.n}_instance_{instance[15]}.png")
    # instance[15] contains 1,2,3
    plt.close()

f.close()
