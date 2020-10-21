# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

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

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #while still iterations
          #for each state
            #for action in each state
              #get Q(state,action)
            #store largest (state,action) in Counter

        for i in range(self.iterations):
          newValues = self.values.copy() #WTF WHY THIS TOOK HOURS
          for state in mdp.getStates():
            v = [float("-inf")]
            if not mdp.isTerminal(state):
              for action in mdp.getPossibleActions(state):
                v += [self.computeQValueFromValues(state,action)]
              newValues[state] = max(v)
          self.values = newValues
        # print self.values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Q(s,a) = Sum_s'( T(s,a,s') + [R(s,a,s') + yV*(s')])
        sat = self.mdp.getTransitionStatesAndProbs(state,action) #list of Sprime And T pairs
        return sum([ pair[1]*(self.mdp.getReward(state, action, pair[0]) + self.discount*self.values[pair[0]]) 
                    for pair in sat])
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE *t**"
        # V*(s) = max_a Q(s,a)
        # Create (qvalue, action) then get max
        noLegal = [(float("-inf"), None)]
        return max(noLegal + [(self.computeQValueFromValues(state,action),action) for action in self.mdp.getPossibleActions(state)],
                    key=lambda x: x[0])[1]

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
