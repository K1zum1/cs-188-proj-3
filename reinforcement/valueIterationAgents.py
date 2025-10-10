# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for i in range(self.iterations):
            new_vals = self.values.copy()
            for state in self.mdp.getStates(): #just use defined methods and update values
                computeAct = self.computeActionFromValues(state)
                if computeAct is not None:
                    new_vals[state] = self.computeQValueFromValues(state, computeAct)
            self.values = new_vals
        return self.values
            


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.

        .getTransitionStatesAndProbs(state, action): 

        Returns list of (nextState, prob) pairs
        representing the states reachable
        from 'state' by taking 'action' along
        with their transition probabilities.
        """
        "*** YOUR CODE HERE ***"
        q_val = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for next_state, prob in transitions: #formula in spec 
            inner = self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state]
            q_val += prob * inner
        return q_val


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        best = None #action
        best_val = float('-inf')
        actions = self.mdp.getPossibleActions(state) 
        if self.mdp.isTerminal(state): 
            return None
        for action in actions: #loop through actions and find the best one
            q_val = self.computeQValueFromValues(state, action)
            if q_val >= best_val: #only change if its equal or better
                best_val = q_val
                best = action
        return best

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def maxFinderHelper(self, s, actions):
        max_q = float('-inf')
        for action in actions:
            q_val = self.computeQValueFromValues(s, action)
            if q_val > max_q:
                max_q = q_val
        return max_q

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        """
        - Compute predecessors of all states.
        - Initialize an empty priority queue.
        - For each non-terminal state s, do: iterate over states in the order returned by self.mdp.getStates()
        - Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s (this represents what the value should be); call this number diff. Do NOT update self.values[s] in this step.
        - Push s into the priority queue with priority -diff (note that this is negative). We use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
        """      
        predescesor = {} #avoid duplicates 
        priority = util.PriorityQueue()

        for state in self.mdp.getStates(): 
            for action in self.mdp.getPossibleActions(state):
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0: 
                        if next_state not in predescesor:
                            predescesor[next_state] = set()
                        predescesor[next_state].add(state) #look into later
            

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                if actions:
                    max_q = self.maxFinderHelper(state, actions)
                    diff = abs(self.values[state] - max_q)
                    if diff > self.theta:
                        priority.update(state, -diff) #neg for min heap

        """
            - For iteration in 0, 1, 2, ..., self.iterations - 1, do:
            - If the priority queue is empty, then terminate.
            - Pop a state s off the priority queue.
            - Update the value of s (if it is not a terminal state) in self.values.
        """
        for i in range(self.iterations):
            if priority.isEmpty():
                return
            s = priority.pop() #current state
            if not self.mdp.isTerminal(s):
                actions = self.mdp.getPossibleActions(s)
                #update the value of s
                if actions:
                    max_q = float('-inf')
                    #find the largest value -- uh is this necessary?
                    for action in actions:
                        q_val = self.computeQValueFromValues(s, action)
                        if q_val > max_q:
                            max_q = q_val
                    self.values[s] = max_q 
                    # Find the abs of the difference between the current value of p in self.values and the highest Q-value across all possible actions from p 
                    # (this represents what the value should be); call this number diff. Do NOT update self.values[p] in this step.
                    # If diff > theta, push p into the priority queue with priority -diff (note that this is negative),
                    # as long as it does not already exist in the priority queue with equal or lower priority. 
                    # As before, we use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a
                    # higher error.
                    for key in predescesor.get(s):
                        actions = self.mdp.getPossibleActions(key)
                        if actions:
                            max_q = self.maxFinderHelper(key, actions)
                            diff = abs(self.values[key] - max_q)
                            if diff > self.theta and priority: #priority.  > -diff:
                                priority.update(key, -diff)
        return self.values




        

                        
        



