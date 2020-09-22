#!/usr/bin/env python

# trunc8 did this
import time
import subprocess
import matplotlib.pyplot as plt

start = time.time()
f = open("outputDataT1.txt","w")

instances = ["../instances/i-1.txt", "../instances/i-2.txt", "../instances/i-3.txt"]
algorithms = ["epsilon-greedy", "ucb", "kl-ucb", "thompson-sampling"]
algorithm = algorithms[1]
randomSeeds = range(50)
epsilon = 0.02
horizons = [100, 400, 1600, 6400, 25600, 102400]
for instance in instances:
    for horizon in horizons:
        for randomSeed in randomSeeds:
            cmd = f"python bandit.py --instance {instance} --algorithm {algorithm} --randomSeed {randomSeed} --epsilon {epsilon} --horizon {horizon}"
            p = subprocess.Popen(cmd.split(), shell=False, stdout=f)
            p.wait()
        print(f"Time elapsed:{time.time()-start:.2f}s instance:{instance} horizon:{horizon}")
f.close()
# cmd = "python bandit.py --instance ../instances/i-1.txt --algorithm epsilon-greedy --randomSeed 0 --epsilon 0.02 --horizon 100"
# subprocess.run(cmd)
