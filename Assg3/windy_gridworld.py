#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import random
import logging
import sys

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
four_actions = {0: [0,1],
  1: [1,0],
  2: [0,-1],
  3: [-1,0]
  } # Order: Right, Down, Left, Up
king_actions = {0: [0,1],}
alpha = 0.5
gamma = 1 # Undiscounted episodic task
max_time_step = 8000

random.seed(0)
np.random.seed(0)

def obtainNextStateAndReward(curr_state, action, actions_list, wind_strength, x_max, y_max, goal_state):
  y, x = curr_state # Assuming current state is valid
  wind_effect = wind_strength[x]
  reward = -1
  
  x_new = x + actions_list[action][1]
  x_new = min(x_new, x_max) # Action chosen may be unrealistic
  x_new = max(x_new, 0)
  
  y_new = y + actions_list[action][0] - wind_effect
  y_new = min(y_new, y_max)
  y_new = max(y_new, 0)

  if (x_new == goal_state[1] and y_new == goal_state[0]):
    return goal_state, -1

  return [y_new, x_new], reward

def runSarsaZero():
  epsilon = 0.1
  episodes = np.zeros(max_time_step+1)
  time_step = 0
  episodes[0] = 0
  actions_list = four_actions
  Q = np.zeros([y_max+1, x_max+1, len(actions_list)])
  prev_timestep = 0
  while(time_step < max_time_step):
    action = 0
    curr_state = start_state
    if (random.uniform(0, 1.0) < epsilon):
      action = random.randint(0,len(actions_list)-1)
    else:
      # actions_from_curr_state = Q[curr_state[0], curr_state[1], :]
      # next_action = np.random.choice(np.flatnonzero(
      #     np.isclose(actions_from_curr_state, actions_from_curr_state.max())))
      action = np.argmax(Q[curr_state[0], curr_state[1], :])

    while(curr_state != goal_state):
      time_step += 1
      if (time_step > max_time_step):
        break
      if (time_step < 50):
        logging.debug(f"t={time_step}\tCurr_state: {curr_state}\t{action_names[action]}")
        logging.debug(f"\t\t\tQ: {Q[curr_state[0], curr_state[1]]}")
      next_state, reward = obtainNextStateAndReward(curr_state, 
        action, actions_list, wind_strength, x_max, y_max, goal_state)
      next_action = 0
      if (random.uniform(0, 1.0) < epsilon):
        next_action = random.randint(0,len(actions_list)-1)
      else:
        # actions_from_next_state = Q[next_state[0], next_state[1], :]
        # next_action = np.random.choice(np.flatnonzero(
        #   actions_from_next_state==actions_from_next_state.max()))
        next_action = np.argmax(Q[next_state[0], next_state[1], :])
      Q[curr_state[0], curr_state[1], action] += alpha*(reward + 
        gamma*Q[next_state[0], next_state[1], next_action] - Q[curr_state[0], curr_state[1], action])
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

def plotEpisodes():
  episodes = np.zeros(max_time_step+1)
  num_of_runs = 10
  for i in range(num_of_runs):
    print(f"Round {i}")
    episodes += (1.0/num_of_runs)*runSarsaZero()
  plt.plot(episodes)
  plt.show()

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
# Below is a toggle switch for logging messages
logging.disable(sys.maxsize)
plotEpisodes()
# print(obtainNextStateAndReward(curr_state, 2, actions_list, wind_strength, x_max, y_max, goal_state))