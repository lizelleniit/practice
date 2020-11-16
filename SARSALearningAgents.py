# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import numpy as np
import random,util,math

class SARSALearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        # print args
        "*** YOUR CODE HERE ***"
        #self.qvalues = util.Counter()
        self.Q={'TERMINAL_STATE': np.array([0])}

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #return self.qvalues[(state,action)]
        print('state',state)
        print('action',action)
        actions=self.getLegalActions(state)
        print("actions",actions)
        ai=actions.index(action)
        if state in self.Q:
            print("staaate",state,"; ai: ",ai,"; action: ",action)
            return self.Q[state][ai]
        else:
            return 0.0
        util.raiseNotDefined()



    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # UGH why no same? return max([0.0] + [self.getQValue(state,action) for action in self.getLegalActions(state)])
        # OH. No consider negatives

        # temp = util.Counter()
        # for action in self.getLegalActions(state): 
        #   temp[action] = self.getQValue(state, action)
        # print temp.argMax(), temp[temp.argMax()]
        # return temp[temp.argMax()]
        

        #values = [self.getQValue(state,action) for action in self.getLegalActions(state)]
        #return max(values) if values else 0.0
        actions=self.getLegalActions(state)
        if not actions:
            # this means there are no legal actions
            best_Q=0.0
        else:
            Qfors=np.zeros(len(actions))
            for ai,a in enumerate(actions):
                Qfors[ai]=self.getQValue(state,a)
            best_Q=np.max(Qfors)
        
        return best_Q
        util.raiseNotDefined()


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #bestPair = (float("-inf"), None)
        #for action in self.getLegalActions(state):
        #  if self.getQValue(state,action) >  bestPair[0]:
        #    bestPair = (self.getQValue(state,action), action)
        #return bestPair[1]
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions=self.getLegalActions(state)
        
        if not legalActions:
            # this means there are no legal actions
            return None
        else:
            Qfors=np.zeros(len(legalActions))
            for ai,a in enumerate(legalActions):
                
                Qfors[ai]=self.getQValue(state,a)
            best_ai=np.random.choice(np.where(Qfors==Qfors.max())[0])
       
            return legalActions[best_ai]
        util.raiseNotDefined()


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        ## Pick Action
        #legalActions = self.getLegalActions(state)
        #action = None
        #"*** YOUR CODE HERE ***"
        
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        
        if np.random.random()<self.epsilon:
            if not legalActions:
                print('no legal actions')
                action = None
            else:
                # return a random action
                action = np.random.choice(legalActions)
        else: # return the best action
            action = self.computeActionFromQValues(state)
                
        

        return action
        util.raiseNotDefined()
    def index_in_list(a_list, index):

        return (index < len(a_list))

    def update(self, state, action, nextState, nextAction, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #sample = reward + self.discount * self.getValue(nextState)
        #self.qvalues[(state,action)] = (1 - self.alpha) * self.qvalues[(state,action)] + self.alpha * sample
        legalActions=self.getLegalActions(state)
        ai=legalActions.index(action)
        legalActionsNext=self.getLegalActions(nextState)
        if legalActionsNext==None or nextAction==None:
            aiNext=0
        else:
            aiNext=legalActionsNext.index(nextAction)
        # check whether we have a key for "state"
        if not state in self.Q:
            self.Q[state]=np.zeros(len(legalActions))
        if not nextState in self.Q:
            #print('nextState ',nextState)
            self.Q[nextState]=np.zeros(len(legalActionsNext))
        
        self.Q[state][ai]+=self.alpha*(reward + self.discount*self.Q[nextState][aiNext]-self.Q[state][ai])
        #print("reward ",reward)
        
        #print("New Q for state ",state," and action ",legalActions[ai],": ",self.Q[state][ai])
        
        return nextState, reward

        util.raiseNotDefined()


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)
        
    def getValue(self, state):
        return self.computeValueFromQValues(state)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class nStepSARSALearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        # print args
        "*** YOUR CODE HERE ***"
        #self.qvalues = util.Counter()
        self.Q={'TERMINAL_STATE': np.array([0])}

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #return self.qvalues[(state,action)]
        print('state',state)
        print('action',action)
        actions=self.getLegalActions(state)
        print("actions",actions)
        ai=actions.index(action)
        if state in self.Q:
            print("staaate",state,"; ai: ",ai,"; action: ",action)
            return self.Q[state][ai]
        else:
            return 0.0
        util.raiseNotDefined()



    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # UGH why no same? return max([0.0] + [self.getQValue(state,action) for action in self.getLegalActions(state)])
        # OH. No consider negatives

        # temp = util.Counter()
        # for action in self.getLegalActions(state): 
        #   temp[action] = self.getQValue(state, action)
        # print temp.argMax(), temp[temp.argMax()]
        # return temp[temp.argMax()]
        

        #values = [self.getQValue(state,action) for action in self.getLegalActions(state)]
        #return max(values) if values else 0.0
        actions=self.getLegalActions(state)
        if not actions:
            # this means there are no legal actions
            best_Q=0.0
        else:
            Qfors=np.zeros(len(actions))
            for ai,a in enumerate(actions):
                Qfors[ai]=self.getQValue(state,a)
            best_Q=np.max(Qfors)
        
        return best_Q
        util.raiseNotDefined()


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #bestPair = (float("-inf"), None)
        #for action in self.getLegalActions(state):
        #  if self.getQValue(state,action) >  bestPair[0]:
        #    bestPair = (self.getQValue(state,action), action)
        #return bestPair[1]
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions=self.getLegalActions(state)
        
        if not legalActions:
            # this means there are no legal actions
            return None
        else:
            Qfors=np.zeros(len(legalActions))
            for ai,a in enumerate(legalActions):
                
                Qfors[ai]=self.getQValue(state,a)
            best_ai=np.random.choice(np.where(Qfors==Qfors.max())[0])
       
            return legalActions[best_ai]
        util.raiseNotDefined()


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        ## Pick Action
        #legalActions = self.getLegalActions(state)
        #action = None
        #"*** YOUR CODE HERE ***"
        
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        
        if np.random.random()<self.epsilon:
            if not legalActions:
                print('no legal actions')
                action = None
            else:
                # return a random action
                action = np.random.choice(legalActions)
        else: # return the best action
            action = self.computeActionFromQValues(state)
                
        

        return action
        util.raiseNotDefined()
    def index_in_list(a_list, index):

        return (index < len(a_list))

    def update(self, state, action, nextState, nextAction, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #sample = reward + self.discount * self.getValue(nextState)
        #self.qvalues[(state,action)] = (1 - self.alpha) * self.qvalues[(state,action)] + self.alpha * sample
        legalActions=self.getLegalActions(state)
        ai=legalActions.index(action)
        legalActionsNext=self.getLegalActions(nextState)
        if legalActionsNext==None or nextAction==None:
            aiNext=0
        else:
            aiNext=legalActionsNext.index(nextAction)
        # check whether we have a key for "state"
        if not state in self.Q:
            self.Q[state]=np.zeros(len(legalActions))
        if not nextState in self.Q:
            #print('nextState ',nextState)
            self.Q[nextState]=np.zeros(len(legalActionsNext))
        
        self.Q[state][ai]+=self.alpha*(reward + self.discount*self.Q[nextState][aiNext]-self.Q[state][ai])
        #print("reward ",reward)
        
        #print("New Q for state ",state," and action ",legalActions[ai],": ",self.Q[state][ai])
        
        return nextState, reward

        util.raiseNotDefined()


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)
        
    def getValue(self, state):
        return self.computeValueFromQValues(state)
