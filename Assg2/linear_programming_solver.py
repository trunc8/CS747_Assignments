#!/usr/bin/env python

# trunc8 did this

import numpy as np
import pulp
from pulp import LpVariable, LpProblem, LpMaximize

def soln(S, A, R, T, gamma):
  V = np.zeros(S)
  Policy = np.zeros(S)

  C_list = {}
  V_list = []
  for s in range(S):
    V_list.append(LpVariable(f"V{s}"))

  prob = LpProblem("MDP_Planning", LpMaximize)
  prob += -pulp.lpSum(V_list)
  for s in range(S):
    C_list[s] = {}
    for a in range(A):
      # To ensure that the lists aren't empty in any case
      C_list[s][a] = [0]
      for s_dash in T[s][a]:
        C_list[s][a].append(T[s][a][s_dash]*(R[s][a][s_dash] + gamma*V_list[s_dash]))
      prob += pulp.lpSum(C_list[s][a]) <= V_list[s]
  
  status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

  for s in range(S):
    V[s] = pulp.value(V_list[s])

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
