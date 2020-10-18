#!/usr/bin/env python

# trunc8 did this

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n")
args = parser.parse_args()
f = open(f"epsilons.txt", "r")

instances = ["../instances/i-1.txt", "../instances/i-2.txt", "../instances/i-3.txt"]
algorithm = "epsilon-greedy"
horizon = 102400
epsilons = [0.0002, 0.00002]

Expt = {}
for instance in instances:
    Expt[instance] = {}
    for epsilon in epsilons:
        Expt[instance][epsilon] = 0
for line in f:
    instance, algorithm, _, epsilon, horizon, REG = line.split(", ")
    Expt[instance][float(epsilon)] += float(REG)/50

for instance in instances:
    for epsilon in epsilons:
        print(f"Instance: {instance}\tEpsilon: {epsilon}\tREG: {Expt[instance][epsilon]}")

f.close()
