#!/usr/bin/env python

# trunc8 did this
import random

def obtainNextStateAndReward(curr_state, 
    action, 
    actions_list, 
    stochasticity = False,
    wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0], 
    x_max = 9, 
    y_max = 6, 
    goal_state = [3, 7], 
    ):
  y, x = curr_state # Assuming current state is valid
  wind_effect = wind_strength[x]
  if stochasticity == True:
    wind_effect = wind_effect + random.randint(-1,1)
  reward = -1
  
  # Allowing maximum movement
  x_new = x + actions_list[action][1]
  y_new = y + actions_list[action][0] - wind_effect

  # But preventing from escaping grid
  x_new = min(x_new, x_max)
  x_new = max(x_new, 0)
  y_new = min(y_new, y_max)
  y_new = max(y_new, 0)

  if (x_new == goal_state[1] and y_new == goal_state[0]):
    return goal_state, -1

  return [y_new, x_new], reward