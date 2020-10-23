#!/usr/bin/env python

# trunc8 did this

import numpy as np
import pulp
from pulp import LpVariable, LpProblem, LpMaximize
import logging

def soln(S, A, R, T, gamma):
  V = np.zeros(S)

  V_list = []
  for s in range(S):
    V_list.append(LpVariable(f"V{s}"))
  V_list = np.array(V_list)
  
  C_list = np.sum(np.multiply(T, R+gamma*V_list), axis=2)
  prob = LpProblem("MDP_Planning", LpMaximize)
  prob += -pulp.lpSum(V_list)
  for s in range(S):
    for a in range(A):
      prob += pulp.lpSum(C_list[s][a]) <= V_list[s]
  
  status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
  if (pulp.LpStatus[status] == 'Optimal'):
    logging.debug("Linear Programming [OK]")
  else:
    logging.error("Linear Programming [FAILED]")

  for s in range(S):
    V[s] = pulp.value(V_list[s])

  # Finding policy*
  Policy = np.argmax(np.sum(np.multiply(T, R + gamma*V), axis=2), axis=1)

  return V, Policy
