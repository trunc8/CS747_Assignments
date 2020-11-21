#!/usr/bin/env python

# trunc8 did this
import time
import subprocess
import os
import argparse

from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument("-n")
args = parser.parse_args()

start = time.time()
prev = start
file = "epsilons.txt"
if os.path.exists(file):
    os.remove(file) # To prevent appending

f = open(file,"w")

instances = ["../instances/i-1.txt", "../instances/i-2.txt", "../instances/i-3.txt"]

algorithm = "epsilon-greedy"

commands = []

# horizons = [100, 400, 1600, 6400, 25600, 102400]
horizon = 102400
randomSeeds = range(50)
epsilons = [0.0002, 0.00002]
for instance in instances:
    for epsilon in epsilons:
            for randomSeed in randomSeeds:
                cmd = f"python bandit.py --instance {instance} --algorithm {algorithm} --randomSeed {randomSeed} --epsilon {epsilon} --horizon {horizon}"
                commands.append(cmd)

n = 40
pool = Pool(n) # n concurrent commands at a time
for i, returncode in enumerate(pool.imap(partial(call, shell=True, stdout=f), commands)):
    if returncode != 0:
       print("%d command failed: %d" % (i, returncode))

    print(f"Time taken: {time.time() - start}")

f.close()
