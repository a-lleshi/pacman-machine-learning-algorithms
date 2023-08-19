# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
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

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import copy
import util

# Grid class partially re-used from mapAgents.py from week 5 Practical on KEATS
# 
class Grid:       

    def __init__(self, width, height):
        self.width = width
        self.height = height
        subgrid = []
        for i in range(self.height):
            row=[]
            for j in range(self.width):
                row.append(0)
            subgrid.append(row)
        self.grid = subgrid

    # Print the grid out.
    def display(self):       
        for i in range(self.height):
            for j in range(self.width):
                # print grid elements with no newline
                print (self.grid[i][j]),
            # A new line after each line of the grid
            print 
        # A line after the grid
        print
        
    def setValue(self, x, y, value):
        self.grid[y][x] = value

    def getValue(self, x, y):
        return self.grid[y][x]

    def getHeight(self):
        return self.height

    def getWidth(self):
        return self.width


class MDPAgent(Agent):
    
   
    def __init__(self):
        
        #initialise reward values to 0
        self.food_reward = 0
        self.empty_cell_reward = 0
        self.capsule_reward = 0
        self.ghost_reward = 0
        self.eatable_ghost_reward = 0
        self.discount_factor = 0
        
    def registerInitialState(self, state):
        #  print "Running registerInitialState!"
         # Make a map of the right size
         self.createGridMaps(state)
         self.addWallsToMap(state)
        
         MDPAgent.setInitialCellValues(self, state)

    def setInitialCellValues(self, state): 
        # print("Setting initial cell values")
        self.refreshConsumablesOnMaps(state)
        self.refreshGhostsOnMaps(state)
                 
    def final(self, state):
        print ("Game finished!")

    def createGridMaps(self,state):
        corners = api.corners(state)

        height = self.getLayoutHeight(corners)
        width  = self.getLayoutWidth(corners)
        self.map = Grid(width, height)
        self.utility_map = Grid(width, height)

        #set the reward values for different layouts 
        if self.map.getWidth() > 7:
         
            self.food_reward = 5
            self.empty_cell_reward = -0.03
            self.capsule_reward = 7
            self.ghost_reward = -30
            self.eatable_ghost_reward = 3
            self.discount_factor = 0.7

        if self.map.getWidth() <= 7:
         
            self.food_reward = 5
            self.empty_cell_reward = -0.03
            self.ghost_reward = -30
            self.discount_factor = 0.9

        
    def getLayoutHeight(self, corners):
        height = -1
        for i in range(len(corners)):
            if corners[i][1] > height:
                height = corners[i][1]
        return height + 1

    def getLayoutWidth(self, corners):
        width = -1
        for i in range(len(corners)):
            if corners[i][0] > width:
                width = corners[i][0]
        return width + 1

   
    def addWallsToMap(self, state):
        walls = api.walls(state)
        for i in range(len(walls)):
            self.map.setValue(walls[i][0], walls[i][1], '#')
            
           
    def refreshConsumablesOnMaps(self, state):
        
        # print "refreshConsumablesOnMaps"

        for i in range(self.map.getWidth()):
            for j in range(self.map.getHeight()):
                if self.map.getValue(i, j) != '#':
                    self.map.setValue(i, j, ' ')
                    self.utility_map.setValue(i,j,self.empty_cell_reward)
                    
        food = api.food(state)
        for i in range(len(food)):
            # print "Food at: ", food[i]
            self.map.setValue(food[i][0], food[i][1], '*')
            self.utility_map.setValue(food[i][0], food[i][1], self.food_reward)
            
        capsules = api.capsules(state)
        for i in range(len(capsules)):
            # print "Capsule at: ", capsules[i]
            self.map.setValue(capsules[i][0], capsules[i][1], 'o')
            self.utility_map.setValue(capsules[i][0], capsules[i][1], self.capsule_reward)
            
            
    def refreshGhostsOnMaps(self, state):
        # print "refreshGhostsOnMaps"
        ghosts = api.ghostStates(state)
        pacman = api.whereAmI(state)
          
        for i in range(len(ghosts)):
            
            if ghosts[i][1] == 1:
                # print "Eatable ghost at: ", ghosts[i][0]
                self.map.setValue(int(ghosts[i][0][0]),int(ghosts[i][0][1]), 'F')
                self.utility_map.setValue(int(ghosts[i][0][0]),int(ghosts[i][0][1]), self.eatable_ghost_reward)
            else:
                # print "Ghost at: ", ghosts[i][0]
                self.map.setValue(int(ghosts[i][0][0]),int(ghosts[i][0][1]), 'G')
                
                #create penalty for cells that are close to ghosts
                for j in range(2):
                    if ghosts[i][0][1] + j < self.map.getHeight():
                        self.utility_map.setValue(int(ghosts[i][0][0]),int(ghosts[i][0][1])+j, self.ghost_reward/(j+1))
                    if ghosts[i][0][1] - j >= 0 :
                        self.utility_map.setValue(int(ghosts[i][0][0]),int(ghosts[i][0][1])-j, self.ghost_reward/(j+1))
                    if  ghosts[i][0][0] + j < self.map.getWidth():
                        self.utility_map.setValue(int(ghosts[i][0][0])+j,int(ghosts[i][0][1]), self.ghost_reward/(j+1))
                    if  ghosts[i][0][0] - j >= 0:
                        self.utility_map.setValue(int(ghosts[i][0][0])-j,int(ghosts[i][0][1]), self.ghost_reward/(j+1))
                
          
    def ValueIteration(self,state):              
               
        if self.map.getWidth() > 7:
            # print "Medium" 
            num_of_iterations = 150
        if self.map.getWidth() <= 7:
            # print "Small"
            num_of_iterations = 80
        for iterations in range(num_of_iterations):
            copy_util_map = copy.copy(self.utility_map)

            for i in range(self.map.getWidth()-1):
                for j in range(self.map.getHeight()-1):
                    
                    if self.map.getValue(i,j) != "#" and self.map.getValue(i,j) != "*" and self.map.getValue(i,j) != "G" and self.map.getValue(i,j) != "F" and  self.map.getValue(i,j) != "o":
                        
                        west = MDPAgent.getNeighborUtility(self, copy_util_map.getValue(i,j), (i-1,j),copy_util_map)    
                 
                        east = MDPAgent.getNeighborUtility(self, copy_util_map.getValue(i,j),(i+1,j),copy_util_map)
                                     
                        north = MDPAgent.getNeighborUtility(self, copy_util_map.getValue(i,j), (i,j+1),copy_util_map)
                    
                        south = MDPAgent.getNeighborUtility(self, copy_util_map.getValue(i,j), (i,j-1),copy_util_map)
                    
                        west_utility = api.directionProb * west + (south * 0.1) + (north * 0.1)

                        east_utility = api.directionProb * east + (south * 0.1) + (north * 0.1)

                        south_utility = api.directionProb * south + (east * 0.1) + (west * 0.1)

                        north_utility = api.directionProb * north + (east * 0.1) + (west * 0.1)
                        
                        max_utility_direction = max([(west_utility), (east_utility), (south_utility), (north_utility)])           
                        
                        self.utility_map.setValue(i, j, self.empty_cell_reward + self.discount_factor * max_utility_direction) 
   
    #get direction utility 
    def getNeighborUtility(self, current_exp, neighbor_coordinates,copy_util_map):
        
        expected_utility = 0 

        if self.map.getValue(neighbor_coordinates[0],neighbor_coordinates[1]) == "#" : # if it is next to the wall
            expected_utility = current_exp  #expected utility value does not change, because it bumps into the wall and does not move
        else: 
            expected_utility = copy_util_map.getValue(neighbor_coordinates[0],neighbor_coordinates[1]) # retrieves neighboring expected utility from the coordinates
        
        return expected_utility

    #get optimal move from value iteration
    def getOptimalMove(self, state):
        
        #refresh new positions of ghosts and consumables every move
        self.refreshConsumablesOnMaps(state)
        self.refreshGhostsOnMaps(state)

        MDPAgent.ValueIteration(self,state)

        legal = api.legalActions(state)

        #remove stop because it is not accounted for in the utility map, does not have utility
        legal.remove(Directions.STOP)
        pacman = api.whereAmI(state)

        available_moves = []
        move_utility = [] 

        #iterate through legal moves and get the utility of each move to return max utility move
        if len(legal) > 0:
            for i in range(len(legal)):  
            
                if legal[i] == "West":

                    move_utility.append(self.utility_map.getValue(pacman[0]-1,pacman[1]))
                    available_moves.append(legal[i])
            
                elif legal[i] == "East":
                   
                    move_utility.append(self.utility_map.getValue(pacman[0]+1,pacman[1]))
                    available_moves.append(legal[i])
                      
                    
                elif legal[i] == "North":
                       
                    move_utility.append(self.utility_map.getValue(pacman[0],pacman[1]+1))
                    available_moves.append(legal[i])
                    

                elif legal[i] == "South":
                   
                    move_utility.append(self.utility_map.getValue(pacman[0],pacman[1]-1))
                    available_moves.append(legal[i])
                     
       
        return available_moves[move_utility.index(max(move_utility))] 
    
    def getAction(self, state):
        
        optimal = self.getOptimalMove(state)
        return api.makeMove(optimal,optimal) # pacman moves to the space with the maximum expected utility
      
