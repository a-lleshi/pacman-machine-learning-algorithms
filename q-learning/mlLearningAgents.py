# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        "*** YOUR CODE HERE ***"
        self.state = state
        self.walls = state.getWalls()
        self.food = state.getFood()
        self.pacman_position = state.getPacmanPosition()
        self.ghost_positions = state.getGhostPositions()
    
    # The __hash__ method computes a hash value for the object. This hash value is used to uniquely
    # identify the object when it is used as a key in a dictionary or a member of a set.
    def __hash__(self):
        # Combine the hash values of the walls, food, pacman_position, and ghost_positions
        # to create a unique hash value for this GameStateFeatures object.
        return hash((self.walls, self.food, self.pacman_position, tuple(self.ghost_positions)))

    # The __eq__ method checks if two GameStateFeatures objects are equal by comparing their attributes.
    def __eq__(self, other):
        # If the other object is None, they are not equal.
        if other is None:
            return False

        # Compare the walls, food, pacman_position, and ghost_positions attributes of both objects.
        # If all attributes are equal, the two GameStateFeatures objects are considered equal.
        return (self.walls == other.walls and
                self.food == other.food and
                self.pacman_position == other.pacman_position and
                self.ghost_positions == other.ghost_positions)

class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.1,
                 epsilon: float = 0.1,
                 gamma: float = 0.6,
                 maxAttempts: int = 100,
                 numTraining: int = 50):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        
        # Initialize the count of games played (episodesSoFar) to 0
        self.episodesSoFar = 0

        # Initialize the Q-values (qValues) using a Counter (a dictionary with default value 0)
        self.qValues = util.Counter()

        # Initialize the visit counts (visitCounts) for state-action pairs using a Counter (a dictionary with default value 0)
        self.visitCounts = util.Counter()

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        # Encourage eating food pellets
        foodReward = (startState.getNumFood() - endState.getNumFood()) * 100

        # Discourage staying too close to the ghosts
        ghostDistances = [util.manhattanDistance(endState.getPacmanPosition(), ghostPos)
                        for ghostPos in endState.getGhostPositions()]
        minGhostDistance = min(ghostDistances) if ghostDistances else 0
        ghostReward = -10 if minGhostDistance < 2 else 0

        # Encourage taking shorter paths
        stepCost = -5

        if endState.isWin():
            return foodReward + ghostReward + stepCost + 500
        elif endState.isLose():
            return foodReward + ghostReward + stepCost - 1000
        else:
            return foodReward + ghostReward + stepCost

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        return self.qValues[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        # Get the legal actions for the Pacman in the given state
        legalActions = state.state.getLegalPacmanActions()
        
        # Remove the STOP action from the list of legal actions, if present
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)

        # If there are no legal actions, return a Q-value of 0
        if not legalActions:
            return 0

        # Calculate the Q-values for each action and return the maximum Q-value
        return max([self.getQValue(state, action) for action in legalActions])

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        # Get the current Q-value for the given state and action
        currentQValue = self.getQValue(state, action)

        # Get the maximum Q-value for the next state
        nextQValue = self.maxQValue(nextState)

        # Calculate the updated Q-value using the Q-learning update rule
        updatedQValue = currentQValue + self.alpha * (reward + self.gamma * nextQValue - currentQValue)

        # Store the updated Q-value in the qValues dictionary
        self.qValues[(state, action)] = updatedQValue

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        self.visitCounts[(state, action)] += 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.visitCounts[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        # If the action has been taken less than maxAttempts times, prioritize exploration by returning a high value (infinity)
        if counts < self.maxAttempts:
            return float('inf')
        
        # If the action has been taken maxAttempts times or more, return the utility (expected Q-value) for exploitation
        return utility

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # Get the legal actions available to Pacman in the current state
        legalActions = state.getLegalPacmanActions()

        # Remove the 'STOP' action if it is in the list of legal actions
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)

        # Create a GameStateFeatures object from the current state
        stateFeatures = GameStateFeatures(state)

        # Choose the action to take according to epsilon-greedy strategy
        if util.flipCoin(self.epsilon):
            # With probability epsilon, choose a random action
            action = random.choice(legalActions)
        else:
            # With probability 1 - epsilon, choose the action with the highest Q-value
            action = max(legalActions, key=lambda a: self.getQValue(stateFeatures, a))

        # Generate the next state based on the chosen action
        nextState = state.generatePacmanSuccessor(action)

        # If the next state is valid (not a terminal state)
        if nextState is not None:
            # Create a GameStateFeatures object from the next state
            nextStateFeatures = GameStateFeatures(nextState)

            # Compute the reward for the transition from state to nextState
            reward = self.computeReward(state, nextState)

            # Update the Q-value for the current state-action pair using the reward and next state
            self.learn(stateFeatures, action, reward, nextStateFeatures)

            # Update the visit count for the current state-action pair
            self.updateCount(stateFeatures, action)

        # Return the chosen action
        return action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
