
import Tetris
import Piece
import numpy as np
import random
import copy
import math
import time

#################################################################################
#
# Agent.py contains the template for an artificial intelligence agent for Tetris
#
#################################################################################


class TetrisAgent:

    # Defines the Agent
    def __init__(self, gridWidth, gridHeight, policy):
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.pieces = Piece.loadStandardSet(gridWidth)
        self.TetrisGame = Tetris.Tetris(self.gridWidth, self.gridHeight, self.pieces)
        self.history = []
        self.sars_data = []
        
        self.state = None
        self.current_sars = []
        self.policy = policy
        
        self.in_start_state = True
        self.in_end_state = False
        
        self.dists = []
        self.current_dist = []
        
        self.prev_score = 0
        
    # Resets the environment
    def reset(self):
        self.TetrisGame.reset()
        self.history = []
        self.sars_data = []
        self.state = None
        self.current_sars = []
        
        self.in_start_state = True
        self.in_end_state = False
        

    # Runs a game to completion
    def run(self, rand):
        action = 0
        dist = self.get_dist()
        #print(dist / np.sum(dist))
        state = self.generate_state(self.TetrisGame)
        self.state = state
        while True:
            self.current_sars = []

            # Add state to history
            self.history.append(self.state) #
            
            # Add s to sars'
            self.current_sars.append(self.state) #
            # Choose a
            if (rand):
                action = self.rand_action_from_dist(dist)
            else:
                action = self.makeActions()
                
            # Add a to sars'
            self.current_sars.append(action)
            
            # Move the game forward one step
            value = self.TetrisGame.step(action)
            state = self.generate_state(self.TetrisGame)
            self.state = state
            self.in_start_state = False
            # Add r to sars'
            r = self.reward() + value
            #print(r)
            #print(r)
            self.current_sars.append(r)
            #Add s' to sars'
            self.current_sars.append(self.state) #
            self.sars_data.append(self.current_sars)
            if (value < 0 or len(self.history) == 1000):
                self.in_end_state = True
                if (self.TetrisGame.score > 0.0 and not (True in [(self.dists[i] == dist).all() for i in range(len(self.dists))])):
                    self.dists.append(dist)
                    self.current_dist = copy.deepcopy(dist)
                break
                
    def get_dist(self):
        r = random.random()
        if (len(self.dists)>0 and r < 0.5):
            index = random.randint(0, len(self.dists) - 1)
            return self.dists[index]
            #return self.dists[0]
        else:
            return np.power(np.array([random.random() for i in self.action_space()]), 3)
        
    # Returns an array giving a binary conversion of the grid, the current piece, and the next piece
    def generate_state(self, game):
        # binary_grid is gridHeight x gridWidth numpy array, 1 if there is a block, 0 otherwise
        binary_grid = np.array([[1 if game.grid[j][i].isOn else 0 for i in range(1, self.gridWidth+1)] for j in range(1, self.gridHeight+1)])
        piece = self.TetrisGame.currentPiece
        next_piece = self.TetrisGame.nextPiece
        return [binary_grid, copy.deepcopy(piece), copy.deepcopy(next_piece)]
        
    # Computes an action
    def makeActions(self):
        return self.action_from_state()
        
    # Returns the action space
    def action_space(self):
        return range(5)
        #return 0
        
    ########################
    # Reward function derived from state
    # Evaluated as follows:
    #
    # reward = a * aggregate height
    #   + b * lines completed
    #   + c * number of holes
    #   + d * bumpiness
    ########################
    def reward(self):
        
        reward = 0.0
        a = -0.310066
        b = 0.960666
        c = -0.45663
        d = -0.384483
        e = 0.2
        altered_grid = self.alter_grid()
        num_lines_completed = self.complete_lines(altered_grid)
        complete_lines = self.TetrisGame.delta_score + num_lines_completed
        ag_height = self.aggregate_height(altered_grid, num_lines_completed)
        holes = self.num_holes(altered_grid)
        bumpiness = self.bumpiness(altered_grid)
        avg_width = self.avg_width(altered_grid) + num_lines_completed
        reward = a * ag_height + b * complete_lines + c * holes + d * bumpiness + e * avg_width
        reward = reward + 10 * self.TetrisGame.delta_score ** 2
        out = reward - self.prev_score
        self.prev_score = reward
        return out / 10
        
        """
        a = -0.510066
        b = 0.760666
        c = -0.35663
        d = -0.184483
        altered_grid = self.alter_grid()
        num_lines_completed = self.complete_lines(altered_grid)
        complete_lines = self.TetrisGame.delta_score + num_lines_completed
        ag_height = self.aggregate_height(altered_grid, num_lines_completed)
        holes = self.num_holes(altered_grid)
        bumpiness = self.bumpiness(altered_grid)
        reward = a * ag_height + b * complete_lines + c * holes + d * bumpiness
        return (reward + 50.0) / 100.0
        """
        #return self.TetrisGame.delta_score ** 2
        
    def alter_grid(self):
        grid = copy.deepcopy(self.state[0])
        p1 = copy.deepcopy(self.state[1])
        x = p1.topLeftXBlock-1
        y = p1.topLeftYBlock-1
        w = p1.width
        h = p1.height
        m = p1.matrix
        continue_dropping = True
        while (continue_dropping):
            for i in range(w):
                if ((y+h >= len(grid)) or (m[h-1][i] == 1 and grid[y+h][w+i] == 1)):
                    continue_dropping = False
            y += 1
        for i in range(y, y+h):
            for j in range(x, x+w):
                if ((i < len(grid)) and grid[i][j] == 0):
                    grid[i][j] = m[y-i][x-j]
        return grid
        
        
        
    def aggregate_height(self, altered_grid, num_lines):
        grid = altered_grid
        ag_height = 0
        for j in range(len(grid[0])):
            highest_index = self.gridHeight - 1
            for i in range(len(grid)):
                if (grid[i][j] == 1):
                    highest_index = i
            ag_height += (self.gridHeight - highest_index - 1)
        return ag_height - self.gridWidth * num_lines
        
    def complete_lines(self, altered_grid):
        grid = altered_grid
        num_lines = 0
        for i in range(len(grid)):
            num = 0
            for j in range(len(grid[0])):
                if (grid[i][j] == 0):
                    break
                else:
                    num += 1
            if (num == self.gridWidth): 
                num_lines += 1
        return num_lines
        
    def num_holes(self, altered_grid):
        grid = altered_grid
        num_holes = 0
        for j in range(len(grid[0])):
            hit_block = False
            for i in range(len(grid)):
                if (grid[i][j] == 1):
                    hit_block = True
                if (hit_block and grid[i][j] == 0):
                    num_holes += 1
        return num_holes
        
    def bumpiness(self, altered_grid):
        grid = altered_grid
        heights = []
        bumpiness = 0
        for j in range(len(grid[0])):
            highest_index = self.gridHeight - 1
            for i in range(len(grid)):
                if (grid[i][j] == 1):
                    heights.append(i)
                    
        for i in range(len(heights)-1):
            bumpiness += abs(heights[i] - heights[i+1])
        return bumpiness
        
    def avg_width(self, altered_grid):
        grid = altered_grid
        is_occupied = np.zeros(len(grid), dtype=np.float)
        running_total = 0.0
        for i in range(len(grid)):
            total = 0.0
            for j in range(len(grid[i])):
                if (grid[i][j] == 1):
                    is_occupied[i] = 1.0
                    total += 1.0
            running_total += total
        if (np.sum(is_occupied) == 0):
            return 0
        avg = running_total / np.sum(is_occupied)
        return avg
        
    def rand_action_from_dist(self, dist):
        number = random.random()
        dist = dist / sum(dist)
        total = 0.0
        for i in range(len(dist)):
            total += dist[i]
            if (number < total):
                return i
        return 0
    ##########################################################    
    # USER DEFINED FUNCTIONS
    ##########################################################   
    
    
     
    #############################
    # Define action_from_state
    #############################
    def action_from_state(self):
        branches = self.TetrisGame.next_states()
        next_states = []
        for i in range(len(branches)):
            next_states.append(self.generate_state(branches[i]))
            
        # Determine action from policy here. Currently defined as random policy
        return random.randint(0, len(self.action_space())-1)

        
     
def main():
    policy = None
    Agent = TetrisAgent(10, 20, policy)
    Agent.run()
    Agent.TetrisGame.visualize(Agent.history)
    #Agent.write_game_data()
        
if __name__ == "__main__":
    main()




