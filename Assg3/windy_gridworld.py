#!/usr/bin/env python
import numpy as np

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

def obtainNextStateAndReward(curr_state, action, actions_list, wind_strength, x_max, y_max, goal_state):
	x, y = curr_state # Assuming current state is valid
	wind_effect = 0
	
	x_new = x + actions_list[action][0]
	if (x_new > x_max or x_new < 0): # Action chosen may be unrealistic
		x_new = x
	else:
		wind_effect = wind_strength[x_new]
	
	y_new = y + actions_list[action][1] + wind_effect
	if (y_new > y_max or y_new < 0):
		y_new = y

	if (x_new == goal_state[0] and y_new == goal_state[1]):
		return [-1,-1], 1

	return [x_new, y_new], -1


curr_state = [3,0]
print(obtainNextStateAndReward(curr_state, 2, four_actions, wind_strength, x_max, y_max, goal_state))