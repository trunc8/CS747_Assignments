#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import random

# lower left corner of gridworld is (0,0)
# x increases towards right
# y increases towards top

wind_strength = [0,0,0,1,1,1,2,2,1,0]
x_max = 9 # Note: Counting from 0
y_max = 6
start_state = [0,3]
goal_state = [7,3]
four_actions = {0: [0,1],
  1: [0,-1],
  2: [1,0],
  3: [-1,0]
  } # Order: Up, Down, Right, Left
alpha = 0.4
epsilon = 0.05
gamma = 1 # Undiscounted episodic task
max_time_step = 12000

random.seed(0)

def obtainNextStateAndReward(curr_state, action, actions_list, wind_strength, x_max, y_max, goal_state):
  x, y = curr_state # Assuming current state is valid
  wind_effect = 0
  reward = -0.5
  
  x_new = x + actions_list[action][0]
  if (x_new > x_max or x_new < 0): # Action chosen may be unrealistic
    x_new = x
    reward = -1
  else:
    wind_effect = wind_strength[x_new]
  
  y_new = y + actions_list[action][1] + wind_effect
  if (y_new > y_max or y_new < 0):
    reward = -1
    y_new = y

  if (x_new == goal_state[0] and y_new == goal_state[1]):
    return goal_state, 10

  return [x_new, y_new], reward

def runSarsaZero():
  episodes = np.zeros(max_time_step+1)
  time_step = 0
  episodes[0] = 0
  actions_list = four_actions
  Q = np.zeros([x_max+1, y_max+1, len(actions_list)])
  while(time_step < max_time_step):
    action = 0
    curr_state = start_state
    if (random.random() < epsilon):
      action = random.randint(0,len(actions_list)-1)
    else:
      action = np.argmax(Q[curr_state[0], curr_state[1], :])

    while(curr_state != goal_state):
      if (episodes[time_step] == 140):
        print(curr_state)
      time_step += 1
      if (time_step > max_time_step):
        break
      next_state, reward = obtainNextStateAndReward(curr_state, 
        action, actions_list, wind_strength, x_max, y_max, goal_state)
      next_action = 0
      if (random.random() < epsilon):
        next_action = random.randint(0,len(actions_list)-1)
      else:
        next_action = np.argmax(Q[curr_state[0], curr_state[1], :])
      Q[curr_state[0], curr_state[1], action] += alpha*(reward + 
        gamma*Q[next_state[0], next_state[1], next_action] - Q[curr_state[0], curr_state[1], action])
      curr_state = next_state[:]
      action = next_action
      if (next_state == goal_state):
        episodes[time_step] = episodes[time_step-1] + 1
        break
      else:
        episodes[time_step] = episodes[time_step-1]
  return episodes

def plotEpisodes():
  episodes = np.zeros(max_time_step+1)
  for i in range(10):
    episodes += 0.1*runSarsaZero()
  plt.plot(episodes)
  plt.show()

plotEpisodes()
# print(obtainNextStateAndReward(curr_state, 2, actions_list, wind_strength, x_max, y_max, goal_state))