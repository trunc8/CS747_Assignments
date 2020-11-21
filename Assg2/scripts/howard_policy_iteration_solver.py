#!/usr/bin/env python

# trunc8 did this

import numpy as np
import pulp
from pulp import LpVariable, LpProblem, LpMaximize
import logging

def policyEvaluation(S, R, T, gamma, Policy):
  V = np.zeros(S)
  V_list = []
  for s in range(S):
    V_list.append(LpVariable(f"V{s}"))
  V_list = np.array(V_list)
  
  C_list = [np.dot(T[s, Policy[s], :], R[s, Policy[s], :] + gamma*V_list) for s in range(S)]
  prob = LpProblem("MDP_Planning", LpMaximize)
  prob += -pulp.lpSum(V_list)
  for s in range(S):
      prob += pulp.lpSum(C_list[s]) <= V_list[s]
  
  status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
  if (pulp.LpStatus[status] == 'Optimal'):
    logging.debug("Linear Equation Solving [OK]")
  else:
    logging.error("Linear Equation Solving [FAILED]")

  for s in range(S):
    V[s] = pulp.value(V_list[s])
  return V

  # rhs = np.array([np.dot(T[s, Policy[s], :], R[s, Policy[s], :]) for s in range(S)])
  # lhs_1 = np.array([T[s, Policy[s], :] for s in range(S)])
  # lhs = np.ones([S,S]) - gamma*lhs_1
  # V = np.matmul(np.linalg.pinv(lhs), rhs)
  # return V

def soln(S, A, R, T, gamma):
  logging.debug("Howard Policy Iteration started")
  V = np.zeros(S)
  Policy = np.argmax(np.sum(np.multiply(T, R + gamma*V), axis=2), axis=1)

  has_improvable_states = True
  eps = 1e-5

  while has_improvable_states:
    has_improvable_states = False
    V = policyEvaluation(S, R, T, gamma, Policy)
    Q = np.sum(np.multiply(T, R + gamma*V), axis=2)
    best_action = np.argmax(Q, axis=1)
    logging.debug(f"Value function: {V}")
    for s in range(S):
      if Q[s, best_action[s]] - V[s] > eps:
        has_improvable_states = True
        Policy[s] = best_action[s]

  return V, Policy
