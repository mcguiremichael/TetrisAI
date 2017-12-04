
import Tetris
import Piece
import numpy as np
import random
import copy

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
        
    # Resets the environment
    def reset(self):
        self.TetrisGame.reset()
        self.history = []
        self.sars_data = []
        self.state = None
        self.current_sars = []

    # Runs a game to completion
    def run(self):
        action = 0
        while True:
            self.current_sars = []
            
            # Generate state
            state = self.generate_state(self.TetrisGame)
            self.state = state
            # Add state to history
            self.history.append(copy.deepcopy(state)) #
            
            # Add s to sars'
            self.current_sars.append(copy.deepcopy(state)) #
            # Choose a
            action = self.makeActions()
            # Add a to sars'
            self.current_sars.append(action)
            
            # Move the game forward one step
            value = self.TetrisGame.step(action)
            
            # Add r to sars'
            r = self.reward()
            self.current_sars.append(r)
            #Add s' to sars'
            self.current_sars.append(copy.deepcopy(state)) #
            self.sars_data.append(self.current_sars)
            if value == -1:
                break
        
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
    def actions(self):
        return range(6)
        
        
    ########################
    # Reward function derived from state
    ########################
    def reward(self):
        grid = self.state[0]
        highest_block_idx = -1
        reward = 0
        lambda_1 = 1
        lambda_2 = 10
        # Iterate through grid rows
        for i in range(self.gridHeight):
            num_blocks = 0
            # Iterate through grid columns
            for j in range(self.gridWidth):
                # Count number of blocks in rows, highest block index
                if (grid[i][j] == 1):
                    num_blocks += 1
            reward += lambda_1 * (num_blocks ** 2)    
            if (num_blocks == 0 and highest_block_idx != -1):
                highest_block_idx = self.gridHeight - i
    
        reward -= lambda_2 * (highest_block_idx ** 2)
        reward += self.TetrisGame.delta_score
        print(reward)
        return reward
        
    ##########################################################    
    # USER DEFINED FUNCTIONS
    ##########################################################   
    
    
     
    #############################
    # Define action_from_state
    #############################
    def action_from_state(self):
        state = self.state
        branches = self.TetrisGame.next_states()
        next_states = []
        for i in range(len(branches)):
            next_states.append(self.generate_state(branches[i]))
            
        # Determine action from policy here
        return random.randint(0, 5)
        
     
def main():
    policy = None
    Agent = TetrisAgent(10, 20, policy)
    Agent.run()
    Agent.TetrisGame.visualize(Agent.history)
    #Agent.write_game_data()
        
if __name__ == "__main__":
    main()




