#!/usr/bin/env python

# trunc8 did this

import numpy as np
import random
import logging
import sys
import windy_gridworld

def runSarsaZero(actions_list, 
    stochasticity = False, 
    wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0], 
    x_max = 9, 
    y_max = 6, 
    start_state = [3, 0], 
    goal_state = [3, 7], 
    max_time_step = 8000, 
    epsilon = 0.1,
    alpha = 0.5, 
    gamma = 1
    ):

  # Initializing all variables
  episodes = np.zeros(max_time_step+1)
  time_step = 0
  prev_timestep = 0
  Q = np.zeros([y_max+1, x_max+1, len(actions_list)]) 
  action = 0
  while(time_step < max_time_step):
    # START OF A NEW EPISODE
    curr_state = start_state
    if (random.uniform(0, 1.0) < epsilon):
      action = random.randint(0,len(actions_list)-1)
    else:
      # actions_from_curr_state = Q[curr_state[0], curr_state[1], :]
      # next_action = np.random.choice(np.flatnonzero(
      #     np.isclose(actions_from_curr_state, actions_from_curr_state.max())))
      action = np.argmax(Q[curr_state[0], curr_state[1], :])

    while(True):
      # 
      time_step += 1
      if (time_step > max_time_step):
        break
      next_state, reward = windy_gridworld.obtainNextStateAndReward(curr_state, 
        action, actions_list, stochasticity)
      next_action = 0
      if (random.uniform(0, 1.0) < epsilon):
        next_action = random.randint(0,len(actions_list)-1)
      else:
        # actions_from_next_state = Q[next_state[0], next_state[1], :]
        # next_action = np.random.choice(np.flatnonzero(
        #   actions_from_next_state==actions_from_next_state.max()))
        next_action = np.argmax(Q[next_state[0], next_state[1], :])
      
      target = reward + gamma*Q[next_state[0], next_state[1], next_action]
      Q[curr_state[0], curr_state[1], action] += alpha*(target - Q[curr_state[0], curr_state[1], action])
      
      #Debug
      if (time_step < 50):
        logging.debug(f"\t\t\tQ_updated: {Q[curr_state[0], curr_state[1]]}")
      
      curr_state = next_state[:]
      action = next_action
      if (next_state == goal_state):
        episodes[time_step] = episodes[time_step-1] + 1
        logging.debug(f"Length of episode: {time_step - prev_timestep}")
        prev_timestep = time_step
        break
      else:
        episodes[time_step] = episodes[time_step-1]
  return episodes