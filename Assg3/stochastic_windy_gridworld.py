#!/usr/bin/env python

# trunc8 did this
import random

def obtainNextStateAndReward(curr_state, action, actions_list, wind_strength, x_max, y_max, goal_state):
  y, x = curr_state # Assuming current state is valid
  wind_effect = wind_strength[x] + random.randint(-1,1)
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