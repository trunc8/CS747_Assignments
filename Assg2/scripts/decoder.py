#!/usr/bin/env python

# trunc8 did this

import logging
import sys

import argparse
import numpy as np

def main():
  logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')
  # Below is a toggle switch for logging messages
  logging.disable(sys.maxsize)

  parser = argparse.ArgumentParser(description="Input path to grid path")
  parser.add_argument("--grid", help="grid file path")
  parser.add_argument("--value_policy", help="value and policy file path")
  args = parser.parse_args()

  grid_file = open(args.grid, "r")
  grid = []
  for line in grid_file:
  	grid.append(line.split())
  grid_file.close()
  grid_side = len(grid[0])

  mdptype ="episodic"
  gamma = 1
  actions = [0,1,2,3] # N, E, W, S respectively
  A = len(actions)

  start = 0
  end = []
  state_no = 0
  state_no_of_grid = np.zeros([grid_side, grid_side], dtype=int)
  for row in range(1, grid_side-1): # Across rows
    for col in range(1, grid_side-1): # Along a row
      if grid[row][col] != '1':
        if grid[row][col] == '2':
          start = state_no
        elif grid[row][col] == '3':
          end.append(state_no)
        state_no_of_grid[row][col] = state_no
        state_no += 1
  S = state_no

  R = np.zeros((S, A, S), dtype=np.float64)
  T = np.zeros((S, A, S), dtype=np.float64)
  for row in range(1, grid_side-1): # Across rows
    for col in range(1, grid_side-1): # Along a row
      if grid[row][col] != '1' and grid[row][col] != '3':
        source = state_no_of_grid[row][col]
        if grid[row-1][col] != '1': # N or 0
          dest = state_no_of_grid[row-1][col]
          R[source][0][dest] = -1
          T[source][0][dest] = 1
        if grid[row][col+1] != '1': # E or 1
          dest = state_no_of_grid[row][col+1]
          R[source][1][dest] = -1
          T[source][1][dest] = 1
        if grid[row][col-1] != '1': # W or 2
          dest = state_no_of_grid[row][col-1]
          R[source][2][dest] = -1
          T[source][2][dest] = 1
        if grid[row+1][col] != '1': # S or 3
          dest = state_no_of_grid[row+1][col]
          R[source][3][dest] = -1
          T[source][3][dest] = 1

  value_policy_file = open(args.value_policy, "r")
  s = 0
  Policy = np.zeros(S, dtype=np.int8)
  for line in value_policy_file:
    data = line.split()
    Policy[s] = int(data[1])
    logging.debug(f"policy[{s}]: {data[1]}")
    s += 1

  s = start
  logging.debug(f"Start state: {s}")
  logging.debug(f"policy: {Policy}")
  while s not in end:
    if(Policy[s]==0):
      print("N", end=' ')
    elif(Policy[s]==1):
      print("E", end=' ')
    elif(Policy[s]==2):
      print("W", end=' ')
    elif(Policy[s]==3):
      print("S", end=' ')
    # sys.exit(0)
    for s_dash in range(S):
      if T[s][Policy[s]][s_dash] == 1:
        s = s_dash
        break

if __name__ == '__main__':
	main()