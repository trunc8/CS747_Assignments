#!/usr/bin/env python

# trunc8 did this

import numpy as np
import matplotlib.pyplot as plt
import random
import sarsa_zero, expected_sarsa, q_learning

# top left corner of gridworld is (0,0)
# x increases towards right
# y increases towards bottom

# wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
# x_max = 9  # Note: Counting from 0
# y_max = 6
# start_state = [3, 0]
# goal_state = [3, 7]
# alpha = 0.5
# gamma = 1 # Undiscounted episodic task


def main():
  four_actions = {0: [0,1],
    1: [1,0],
    2: [0,-1],
    3: [-1,0]
    } # Order: E, S, W, N
  king_actions = {0: [0,1],
    1: [1,1],
    2: [1,0],
    3: [1,-1],
    4: [0,-1],
    5: [-1,-1],
    6: [-1,0],
    7: [-1,1]
    } # Order: E, SE, S, SW, W, NW, N, NE
  max_time_step = 8000 # Time horizon to count episodes
  num_of_runs = 10 # Number of runs to average over
  
  # Sarsa(0) with four actions
  episode_count_four_actions = np.zeros(max_time_step+1)
  actions_list = four_actions
  for i in range(num_of_runs):
    episode_count_four_actions += (1.0/num_of_runs)*sarsa_zero.runSarsaZero(
      actions_list, max_time_step = max_time_step)
  plt.plot(episode_count_four_actions)
  plt.title("SARSA(0) with Four Actions")
  plt.xlabel("Time steps")
  plt.ylabel("Episodes")
  # plt.show()
  plt.savefig("sarsa_zero_four_actions.png")
  plt.clf()

  # Sarsa(0) with king's actions
  episode_count_king_actions = np.zeros(max_time_step+1)
  actions_list = king_actions
  for i in range(num_of_runs):
    episode_count_king_actions += (1.0/num_of_runs)*sarsa_zero.runSarsaZero(
      actions_list, max_time_step = max_time_step)
  plt.plot(episode_count_king_actions)
  plt.title("SARSA(0) with King's Actions")
  plt.xlabel("Time steps")
  plt.ylabel("Episodes")
  # plt.show()
  plt.savefig("sarsa_zero_king_actions.png")
  plt.clf()

  # Sarsa(0) with stochastic wind and king's actions
  episode_count_stoc_king_actions = np.zeros(max_time_step+1)
  actions_list = king_actions
  for i in range(num_of_runs):
    episode_count_stoc_king_actions += (1.0/num_of_runs)*sarsa_zero.runSarsaZero(
      actions_list, stochasticity = True, max_time_step = max_time_step)
  plt.plot(episode_count_stoc_king_actions)
  plt.title("SARSA(0) with King's Actions and Stochastic Wind")
  plt.xlabel("Time steps")
  plt.ylabel("Episodes")
  # plt.show()
  plt.savefig("sarsa_zero_stochastic_king_actions.png")
  plt.clf()

  # Sarsa(0), Expected Sarsa, Q-learning with four actions
  episode_count_sarsa_zero = np.zeros(max_time_step+1)
  episode_count_expected_sarsa = np.zeros(max_time_step+1)
  episode_count_q_learning = np.zeros(max_time_step+1)
  actions_list = four_actions
  for i in range(num_of_runs):
    episode_count_sarsa_zero += (1.0/num_of_runs)*sarsa_zero.runSarsaZero(
      actions_list, max_time_step = max_time_step)
    episode_count_expected_sarsa += (1.0/num_of_runs)*expected_sarsa.runExpectedSarsa(
      actions_list, max_time_step = max_time_step)
    episode_count_q_learning += (1.0/num_of_runs)*q_learning.runQLearning(
      actions_list, max_time_step = max_time_step)
    
  plt.plot(episode_count_sarsa_zero, label="SARSA(0)")
  plt.plot(episode_count_expected_sarsa, label="Expected SARSA")
  plt.plot(episode_count_q_learning, label="Q Learning")

  plt.legend()
  plt.title("Comparison of TD control algorithms")
  plt.xlabel("Time steps")
  plt.ylabel("Episodes")
  # plt.show()
  plt.savefig("comparison_td_algorithms.png")
  plt.clf()

if __name__ == '__main__':
  random.seed(0)
  np.random.seed(0)
  main()

