
import TetrisAgent
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import random
import copy
import time
import math
import matplotlib
import matplotlib.pyplot as plt

from Piece import from_array

# This file serves as a template for Tetris AI.


""" Global Variables """

width = 10
height = 20
depth = 2

""" End Global Variables """




class Net(nn.Module):

    def __init__(self, num_convolutions, num_convolutions_bot, linear_sizes, ks, ks_bot, input_shape=(1, depth, height, width), output_shape = 5):
        super(Net, self).__init__()
        
        self.activation = F.relu
        self.input_shape = input_shape
        
        self.conv1 = nn.Conv3d(in_channels=num_convolutions[0],
                                    out_channels=num_convolutions[1],
                                    kernel_size=ks[0],
                                    padding=(0, 2, 2))
        #self.conv1.weight.data.normal_(0, 0.1)
        #self.conv1.bias.data.normal_(0, 0.1)
                                    
        self.conv2 = nn.Conv3d(in_channels=num_convolutions[1],
                                    out_channels=num_convolutions[2],
                                    kernel_size=ks[1],
                                    padding=(0, 1, 1))
        #self.conv2.weight.data.normal_(0, 0.1)
        #self.conv2.bias.data.normal_(0, 0.1)
        
        self.conv3 = nn.Conv3d(in_channels=num_convolutions[2],
                                    out_channels=num_convolutions[3],
                                    kernel_size=ks[2],
                                    padding=(0, 2, 2))
        #self.conv3.weight.data.normal_(0, 0.1)
        #self.conv3.bias.data.normal_(0, 0.1)
        
        
        # Bottom convolution
        
        self.convBot = nn.Conv3d(in_channels=num_convolutions_bot[0],
                                    out_channels=num_convolutions_bot[1],
                                    kernel_size=ks_bot[0],
                                    padding=(0, 1, 0))
        #self.convBot.weight.data.normal_(0, 0.1)
        #self.convBot.bias.data.normal_(0, 0.1)
        
                                    
        self.n_size = self._get_conv_output(input_shape)
        self.n_size_bot = self._get_bot_conv_output(input_shape)
        self.linear_sizes = linear_sizes
        self.linear_sizes.insert(0, self.n_size)  #+self.n_size_bot)
        
        
        self.fc1 = nn.Linear(linear_sizes[0], linear_sizes[1])
        #self.fc1.weight.data.normal_(0, 0.02)
        #self.fc1.bias.data.normal_(0, 0.02)
        
        self.fc2 = nn.Linear(linear_sizes[1], linear_sizes[2])
        #self.fc2.weight.data.normal_(0, 0.02)
        #self.fc2.bias.data.normal_(0, 0.02)
        
        self.fc3 = nn.Linear(linear_sizes[2], output_shape)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.normal_(0, 0.1)
        
        self.loss = nn.MSELoss()
        
    def _get_conv_output(self, shape):
        inp = Variable(torch.rand(1, *shape))
        output_feat = self._forward_features(inp)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size
        
    def _get_bot_conv_output(self, shape):
        inp_bot = Variable(torch.rand(1, *shape))
        output_feat_bot = self.convBot(inp_bot)
        return output_feat_bot.data.view(1, -1).size(1)
    
    def _forward_features(self, x):
        f = self.conv1
        x = self.activation(f(x))
        x = self.activation(self.conv2(x))
        #x = self.activation(self.conv3(x))
        return x
         
    def forward(self, x):
        self.conv1.double()
        self.conv2.double()
        #self.conv3.double()
        self.convBot.double()
        self.fc1.double()
        self.fc2.double()
        self.fc3.double()
        feat_top = self._forward_features(x)
        #feat_bot = self.activation(self.convBot(x))
        #x = torch.cat([feat_top.view(-1, self.n_size), feat_bot.view(-1, self.n_size_bot)], 1)
        x = feat_top.view(-1, self.n_size)
        x = self.activation(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.activation(self.fc2(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x
        
    def mse_loss(self, inp, target):
        return torch.sum((inp - target) ** 2) / inp.data.nelement()

class MyAgent(TetrisAgent.TetrisAgent):

    ########################
    # Constructor
    ########################state
    def __init__(self, gridWidth, gridHeight, policy=None, optimizer=None, epsilon_min=0.03, epsilon_max = 1.0, epsilon_decay = 50, training=True, batch_size=80, QDepth=1, Gamma=0.99, replay_mem_len = 30000):
        TetrisAgent.TetrisAgent.__init__(self, gridWidth, gridHeight, policy)
        self.Q = policy
        self.Q.train()
        self.cached_Q = copy.deepcopy(policy)
        self.optimizer = optimizer
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.step = 0
        self.training = training
        self.batch_size = batch_size
        
        self.all_histories = []
        self.replay_memory = []
        self.QDepth = QDepth
        self.Gamma = Gamma
        self.replay_mem_len = replay_mem_len
        
        self.original_history = []
        self.training_history = []
        self.testing_history = []
        
        self.errors = []
        
        self.num_frames = 0
    
    #############################
    # Define action_from_state
    #
    # The actions are as follows:
    # 0: Move the current piece left, if it is a valid move
    # 1: Move the current piece right, if it is a valid move
    # 2: Rotate the current piece clockwise
    # 3: Rotate the current piece counter-clockwise
    # 4: Drop the piece to the bottom
    # 5: no-op
    #############################   
    def action_from_state(self):
        current_state = self.state
        action = 0
        if (self.training):            # epsilon search
            number = random.random()
            eps = self.epsilon_at_step()
            if (number < eps):
                action = random.randint(0, len(self.action_space()) - 1)
            else:
                outputs = self.evaluate(current_state)
                action = np.argmax(outputs.data.numpy())        # default action
            
            if (self.num_frames % 20 == 0):
                err = self.gradient_step()
                print(str(self.num_frames), err)
                self.errors.append(err)
            
            if (len(self.errors) % 20 == 0):
                self.plot_errors()
                
            # Update the target every 300 iterations
            if (len(self.errors) % 15 == 0):
                self.cached_Q = copy.deepcopy(self.Q)
            
            self.num_frames += 1
            
        else:
            outputs = self.evaluate(current_state)
            action = np.argmax(outputs.data.numpy())
        return action
        
    def plot_errors(self):
    
        
   
        plt.figure(1)
        plt.clf()
        errors = np.array(self.errors)
        plt.title('Training . . .')
        plt.xlabel('Episode')
        plt.ylabel('Error')
        #plt.ylim(0, 100)
        plt.plot(errors)
        plt.pause(0.001)
        
    def generate_replay_memory(self):
        self.training = False
        for i in range(self.replay_mem_len):
            score = self.perform_iteration(rand=True, selective=True)
            print (str(i+1) + "th replay memory game generated with score ", score)
        self.original_history = copy.deepcopy(self.all_histories)
        print(len(self.original_history))
        
    # Returns values of all actions from a single state
    def evaluate(self, state):
        s = stack(state)
        shape = s.shape
        s = s.reshape((1, 1, shape[0], shape[1], shape[2]))
        data = Variable(torch.from_numpy(s))
        data = data.type(torch.DoubleTensor)
        answer = self.Q(data)
        return answer
        
    def evaluate_cached(self, state):
        s = stack(state)
        shape = s.shape
        s = s.reshape((1, 1, shape[0], shape[1], shape[2]))
        data = Variable(torch.from_numpy(s))
        data = data.type(torch.DoubleTensor)
        answer = self.cached_Q(data)
        return answer
        
    # Returns values of all actions from a group of states
    def evaluate_many(self, in_states, policy):
        s = stack(in_states[0])
        shape = s.shape
        states = np.zeros((len(in_states), 1, shape[0], shape[1], shape[2]))
        states[0] = states[0]
        for i in range(1, len(states)):
            states[i] = stack(in_states[i])
        data = Variable(torch.from_numpy(states))
        data = data.type(torch.DoubleTensor)
        answer = policy(data)
        return answer
        
    # Compute epsilon exploration factor
    def epsilon_at_step(self):
        exp = (-1. * self.step / self.epsilon_decay)
        exp = math.e ** exp
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * exp
    
    # Return torch.ByteTensor to condense output of evaluate_many to desired vector
    def action_mask(self, actions):
        actions = np.array(actions)
        mask = np.array([[1 if actions[j] == i else 0 for i in range(len(self.action_space()))] for j in range(len(actions))])
        mask = Variable(torch.from_numpy(mask)).type(torch.ByteTensor)
        return mask
        
    # Performs gradient descent, handles the training
    def gradient_step(self):
        data, actions, rewards, data_next = self.random_batch_samples()
        
        actions = self.action_mask(actions)
        values = self.evaluate_many(data, self.Q)
        
        targets = self.Gamma * self.evaluate_many(data_next, self.cached_Q)
        targets = torch.max(targets, 1)[0]
        
        #print(targets.data.numpy())
        
        rewards = (Variable(torch.from_numpy(np.array(rewards)))).type(torch.DoubleTensor)
        
        #print(values, targets, rewards)
        
        targets += (rewards / 10)
        #targets += np.array(rewards)
        
        #targets = Variable(torch.from_numpy(targets))
        #targets = targets.type(torch.DoubleTensor)
        output = torch.masked_select(values, actions)
        t0 = time.time()
        self.optimizer.zero_grad()
        error = self.Q.mse_loss(targets, output)
        error.backward()
        self.optimizer.step()
        #print(output.data.numpy()[0], targets.data.numpy()[0])
        
        return error.data.numpy()[0]
        
    def train(self, num_iterations):
        print ("Now training : \n \n ")
        self.training = True
        self.Q.train()
        for i in range(num_iterations):
            t0 = time.time()
            reward = self.perform_iteration()
            #print("full iteration time: ", time.time() - t0)
            if (self.step % 2 == 0):
                self.visualize([self.all_histories[-1]])
            print("Game " + str(self.step) + " had a score of " + str(reward))
            self.step += 1
            
        self.training_history = self.all_histories[self.replay_mem_len:( self.replay_mem_len + num_iterations)]
            
    def test(self, num_iterations):
        self.training = False
        self.Q.eval()
        for i in range(num_iterations):
            reward = self.perform_iteration()
            print(reward)
            
        # Seperate out test results
        lower = self.replay_mem_len + len(self.training_history)
        upper = len(self.all_histories)
        self.testing_history = self.all_histories[lower : upper]
            
    # Runs an entire game and records the data
    def perform_iteration(self, rand=False, selective=False):
        self.run(rand)
        final_score = self.TetrisGame.score
        if (selective):
            if (final_score > 0.0):
                self.all_histories.append(self.history)
                self.replay_memory.append(self.sars_data)
        else:
            self.all_histories.append(self.history)
            self.replay_memory.append(self.sars_data)
        self.reset()
        return final_score
    
    # returns self.batch_size samples from replay memory, stored as states, targets, and actions
    def random_batch_samples(self):
        output_data = []
        data_next = []
        output_targets = np.zeros((self.batch_size, 1), dtype=np.float)
        output_actions = []
        rewards = []
        for i in range(self.batch_size):
            episode = random.randint(0, len(self.replay_memory)-1)
            iteration = random.randint(0, len(self.replay_memory[episode])-1)
            output_data.append(self.replay_memory[episode][iteration][0])
            #target = self.compute_target(episode, iteration, self.QDepth)
            
            rewards.append(self.replay_memory[episode][iteration][2])
            data_next.append(self.replay_memory[episode][iteration][3])
            
            #output_targets[i] = target
            output_actions.append(self.replay_memory[episode][iteration][1])
        return output_data, output_actions, rewards, data_next
        
    # Evaluates the target value of states in the replay memory
    # QDepth determines how many steps forward the Q function is computed
    def compute_target(self, episode, iteration, QDepth):
        if (iteration >= len(self.replay_memory[episode])):
            return 0.0
        if (QDepth == 0):
            v = self.evaluate(self.replay_memory[episode][iteration][0])
            outputs = v.data.numpy()
            index = np.argmax(outputs)
            return outputs[0][index]
        return self.replay_memory[episode][iteration][2] + self.Gamma * self.compute_target(episode, iteration + 1, QDepth - 1)
        
        
    def visualize(self, history):
        for i in range(len(history)):
            self.TetrisGame.visualize(history[i])
        
    def convert_replay_mem(self):
        new_replay_mem = []
        for i in range(len(self.replay_memory)):
            new_game = []
            game = self.replay_memory[i]
            for j in range(len(game)):
                new_sars = []
                sars = game[j]
                new_sars.append(self.convert_state(sars[0]))
                new_sars.append(sars[1])
                new_sars.append(sars[2])
                new_sars.append(self.convert_state(sars[3]))
                
                new_game.append(np.array(new_sars))
            new_replay_mem.append(np.array(new_game))
        return np.array(new_replay_mem)                
               
    def convert_state(self, state):
        new_state = []
        new_state.append(state[0])
        new_state.append(state[1].to_array())
        new_state.append(state[2].to_array())
        return np.array(new_state)
        
    def recover_replay_mem(self, data):
        replay_mem = []
        for i in range(len(data)):
            new_game = []
            game = data[i]
            for j in range(len(game)):
                new_sars = []
                sars = game[j]
                
                new_sars.append(self.recover_state(sars[0]))
                new_sars.append(sars[1])
                new_sars.append(sars[2])
                new_sars.append(self.recover_state(sars[3]))
                
                new_game.append(np.array(new_sars))
            replay_mem.append(np.array(new_game))
        return np.array(replay_mem)
        
    def recover_state(self, state):
        new_state = []
        new_state.append(state[0])
        new_state.append(from_array(state[1]))
        new_state.append(from_array(state[2]))
        return np.array(new_state)
    
    
    def write_replay_mem(self):
        #print(np.array(self.original_history))
        np.save("replay_mem.npy", self.convert_replay_mem())
        #print(np.load("replay_mem.npy"), "loaded data")    
        
    def read_replay_mem(self):
        self.replay_memory = np.load("replay_mem.npy")
        self.replay_memory = self.recover_replay_mem(self.replay_memory).tolist()
        print(len(self.replay_memory))
        self.replay_memory = n_copies(self.replay_memory, 10)
        print(len(self.replay_memory))
        #print(self.original_history)
    
def stack(s):
    '''
    s is structured as a list containing: [numpy_array_grid, Piece, Piece]
    return:
        depth by h by w numpy array. Top layer is grid, middle layer is current piece, bottom layer is next piece
    '''
    grid = s[0]
    p1 = s[1]
    p2 = s[2]
    shape = (depth, grid.shape[0], grid.shape[1])
    output = np.zeros(shape, dtype=np.float)
    
    # Top layer is the grid
    output[0,:,:] = grid
    
    """
    Remove the following five lines if you want to restore the old data structuring.
    This section is for a 1 by h by w output, where the current piece is stored as
    negative 1's. The other option is to do depth x h x w where the piece is stored
    in a deeper layer.
    """
    
    """
    x = p1.topLeftXBlock-1
    y = p1.topLeftYBlock-1
    w = p1.width
    h = p1.height
    output[0,y:y+h,x:x+w] = -1 * p1.matrix
    """
    
    # Middle layer is ones where the current piece is, zero elsewhere
    x = p1.topLeftXBlock-1
    y = p1.topLeftYBlock-1
    w = p1.width
    h = p1.height
    output[1,y:y+h,x:x+w] = p1.matrix
    
    """
    # Bottom layer is ones where the next piece will start, zero elsewhere
    x = p2.topLeftXBlock-1
    y = p2.topLeftYBlock-1
    w = p2.width
    h = p2.height
    output[2,y:y+h,x:x+w] = p2.matrix
    
    """
    
    return output
    
def generateModel():
    num_convolutions_top = [1, 40, 40, 40]
    num_convolutions_bot = [1, 30]
    linear_sizes = [4000, 10000]
    ks_top = [(depth, 5, 5), (1, 3, 3), (1, 2, 2)]
    ks_bot = [(depth, 4, width)]
    output = Net(num_convolutions_top, num_convolutions_bot, linear_sizes, ks_top, ks_bot)
    return output
    
def n_copies(list_to_copy, n):
    output = []
    for i in range(len(list_to_copy)):
        for j in range(n):
            output.append(copy.deepcopy(list_to_copy[j]))
            
    return output    
    
def main():
    # Define your policy
    policy = generateModel()
    print(policy)
    # Define your optimizer
    optimizer = optim.SGD(policy.parameters(), lr=5e-5, momentum=0.0)
    # Declare Agent - it is constructed assuming it is to be trained
    Agent = MyAgent(10, 20, policy, optimizer)
    #print ("Generating replay memory . . .")
    #Agent.generate_replay_memory()
    #Agent.write_replay_mem()
    Agent.read_replay_mem()
    
    
    print ("Training . . .")
    Agent.train(1000)
    print ("Testing . . .")
    Agent.test(50)
    # Visualize random games
    Agent.visualize(Agent.testing_history)
    
    
    
    
if __name__ == "__main__":
    main()
