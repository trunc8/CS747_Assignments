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
  args = parser.parse_args()

  grid_file = open(args.grid, "r")
  grid = []
  for line in grid_file:
  	grid.append([int(s) for s in line.split()])
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
      if grid[row][col] != 1:
        if grid[row][col] == 2:
          start = state_no
        elif grid[row][col] == 3:
          end.append(state_no)
        state_no_of_grid[row][col] = state_no
        state_no += 1
  S = state_no

  R = np.zeros((S, A, S), dtype=np.float64)
  T = np.zeros((S, A, S), dtype=np.float64)
  for row in range(1, grid_side-1): # Across rows
    for col in range(1, grid_side-1): # Along a row
      if grid[row][col] not in [1,3]:
        source = state_no_of_grid[row][col]
        if grid[row-1][col] not in [1,3]: # N or 0
          dest = state_no_of_grid[row-1][col]
          R[source][0][dest] = -1
          T[source][0][dest] = 1
        elif grid[row-1][col] == 1:
          R[source][0][source] = -2
          T[source][0][source] = 1
        if grid[row][col+1] not in [1,3]: # E or 1
          dest = state_no_of_grid[row][col+1]
          R[source][1][dest] = -1
          T[source][1][dest] = 1
        elif grid[row][col+1] == 1:
          R[source][1][source] = -2
          T[source][1][source] = 1
        if grid[row][col-1] not in [1,3]: # W or 2
          dest = state_no_of_grid[row][col-1]
          R[source][2][dest] = -1
          T[source][2][dest] = 1
        elif grid[row][col-1] == 1:
          R[source][2][source] = -2
          T[source][2][source] = 1
        if grid[row+1][col] not in [1,3]: # S or 3
          dest = state_no_of_grid[row+1][col]
          R[source][3][dest] = -1
          T[source][3][dest] = 1
        elif grid[row+1][col] == 1:
          R[source][3][source] = -2
          T[source][3][source] = 1
      if grid[row][col] == 3:
        dest = state_no_of_grid[row][col]
        if grid[row-1][col] not in [1,3]: # N or 0
          source = state_no_of_grid[row-1][col]
          R[source][0][dest] = 0
          T[source][0][dest] = 1
        if grid[row][col+1] not in [1,3]: # E or 1
          source = state_no_of_grid[row][col+1]
          R[source][1][dest] = 0
          T[source][1][dest] = 1
        if grid[row][col-1] not in [1,3]: # W or 2
          source = state_no_of_grid[row][col-1]
          R[source][2][dest] = 0
          T[source][2][dest] = 1
        if grid[row+1][col] not in [1,3]: # S or 3
          source = state_no_of_grid[row+1][col]
          R[source][3][dest] = 0
          T[source][3][dest] = 1

  print(f"numStates {S}")
  print(f"numActions {A}")
  print(f"start {start}")
  print("end", end='')
  if len(end) == 0:
    print("-1")
  else:
    for ed in end:
      print(f" {ed}", end='')
    print()
  for s in range(S):
    for a in range(A):
      for s_dash in range(S):
        if T[s][a][s_dash] == 1:
          print(f"transition {s} {a} {s_dash} {R[s][a][s_dash]} {T[s][a][s_dash]}")
  print(f"mdptype {mdptype}")
  print(f"discount {gamma}")        


if __name__ == '__main__':
	main()