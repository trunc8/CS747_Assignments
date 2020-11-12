#!/usr/bin/env python

# trunc8 did this

import numpy as np
import matplotlib.pyplot as plt
import random
import logging
import sys
import sarsa_zero

# top left corner of gridworld is (0,0)
# x increases towards right
# y increases towards bottom

wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
x_max = 9  # Note: Counting from 0
y_max = 6
start_state = [3, 0]
goal_state = [3, 7]
action_names = {0: "Right",
  1: "Down",
  2: "Left",
  3: "Up"
  }
alpha = 0.5
gamma = 1 # Undiscounted episodic task


def main():
  four_actions = {0: [0,1],
    1: [1,0],
    2: [0,-1],
    3: [-1,0]
    } # Order: Right, Down, Left, Up
  king_actions = {0: [0,1],
    1: [1,1],
    2: [1,0],
    3: [1,-1],
    4: [0,-1],
    5: [-1,-1],
    6: [-1,0],
    7: [-1,1]
    } # Order: E, SE, S, SW, W, NW, N, NE
  max_time_step = 8000
  num_of_runs = 10
  
  # Sarsa(0) with four actions
  episode_count_four_actions = np.zeros(max_time_step+1)
  actions_list = four_actions
  for i in range(num_of_runs):
    episode_count_four_actions += (1.0/num_of_runs)*sarsa_zero.runSarsaZero(
      actions_list, max_time_step = max_time_step)
  plt.plot(episode_count_four_actions)
  plt.title("SARSA(0) with Four Actions")
  plt.show()

  # Sarsa(0) with king's actions
  episode_count_king_actions = np.zeros(max_time_step+1)
  actions_list = king_actions
  for i in range(num_of_runs):
    episode_count_king_actions += (1.0/num_of_runs)*sarsa_zero.runSarsaZero(
      actions_list, max_time_step = max_time_step)
  plt.plot(episode_count_king_actions)
  plt.title("SARSA(0) with King's Actions")
  plt.show()

  # Sarsa(0) with stochastic wind and king's actions
  episode_count_stoc_king_actions = np.zeros(max_time_step+1)
  actions_list = king_actions
  for i in range(num_of_runs):
    episode_count_stoc_king_actions += (1.0/num_of_runs)*sarsa_zero.runSarsaZero(
      actions_list, stochasticity = True, max_time_step = max_time_step)
  plt.plot(episode_count_stoc_king_actions)
  plt.title("SARSA(0) with King's Actions and Stochastic Wind")
  plt.show()

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')
  logging.getLogger('matplotlib').setLevel(logging.WARNING)
  # Below is a toggle switch for logging messages
  logging.disable(sys.maxsize)
  random.seed(0)
  np.random.seed(0)
  main()

