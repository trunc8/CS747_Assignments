#!/usr/bin/env python

# trunc8 did this

import numpy as np

def soln(S, A, R, T, gamma):
  V = np.zeros(S)
  Policy = np.zeros(S)

  eps = 1e-15

  timestep = 1
  V_next = np.zeros(S)
  while True:
    for s in range(S):
      maxm_value = float('-inf')
      # Maximize value over all actions
      for a in range(A):
        value = 0
        for s_dash in T[s][a]:
          value += T[s][a][s_dash]*(R[s][a][s_dash] + gamma*V[s_dash])
        maxm_value = max(maxm_value, value)
      V_next[s] = maxm_value

    if np.linalg.norm(V_next - V) < eps:
      break # Success
    V[:] = V_next
    timestep += 1
    if timestep > 1e15:
      print("Too many timesteps. Please check")
      break
  # Unless we had a timeout, V is now V*
  
  # Finding policy*
  for s in range(S):
    maxm_action_value = float('-inf')
    maxm_action = 0
    for a in range(A):
      action_value = 0
      for s_dash in T[s][a]:
        action_value += T[s][a][s_dash]*(R[s][a][s_dash] + gamma*V[s_dash])
      if action_value > maxm_action_value:
        maxm_action_value = action_value
        maxm_action = a
    Policy[s] = maxm_action

  return V, Policy
