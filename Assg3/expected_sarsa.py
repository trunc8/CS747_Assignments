#!/usr/bin/env python

# trunc8 did this

import numpy as np
import random
import windy_gridworld

def runExpectedSarsa(actions_list, 
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
      # This loop exits when goal state is reached or when max time step is reached
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

      action_probabilities = np.full(len(actions_list), epsilon/len(actions_list))
      actions_from_next_state = Q[next_state[0], next_state[1], :]
      max_indices = np.flatnonzero(actions_from_next_state==actions_from_next_state.max())
      action_probabilities[max_indices] += (1-epsilon)/len(max_indices)
      
      target = reward + gamma*np.sum(np.dot(action_probabilities, Q[next_state[0], next_state[1], :]))
      Q[curr_state[0], curr_state[1], action] += alpha*(target - Q[curr_state[0], curr_state[1], action])
      
      curr_state = next_state[:]
      action = next_action
      if (next_state == goal_state):
        episodes[time_step] = episodes[time_step-1] + 1
        prev_timestep = time_step
        break
      else:
        episodes[time_step] = episodes[time_step-1]
  return episodes