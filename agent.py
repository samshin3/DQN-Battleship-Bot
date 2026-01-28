import numpy as np
import math
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from battleship import *

from collections import deque

# Reward Constants
HIT = 1
SINK = 10
MISS = -0.1
WIN = 100

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 2, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels= 64, kernel_size = 3, padding = 1)

        self.fc = nn.Linear(64 * 10 * 10, 100)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
class DQN_MLP(nn.Module):

    def __init__(self):
        super(DQN_MLP, self).__init__()
        self.fc1 = nn.Linear(200,256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128,100)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x= torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_step(model, memory, optimizer, batch_size=64, gamma=0.99):
    batch = memory.sample(batch_size)
    states, actions, rewards, next_states, next_masks, dones = zip(*batch)

    states = torch.cat(states).float()
    next_states = torch.cat(next_states).float()
    actions = torch.tensor(actions).unsqueeze(1)
    rewards = torch.tensor(rewards).float()
    dones = torch.tensor(dones).float()
    next_masks = torch.tensor(np.array(next_masks)).view(batch_size, -1)

    current_q = model(states).gather(1, actions)

    with torch.no_grad():
        next_q = model(next_states)
        next_q[next_masks == 1] = -1e9
        max_q = next_q.max(1)[0]
        target_q = rewards + (gamma * max_q * (1 - dones))

    loss = nn.MSELoss()(current_q.squeeze(), target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 

class ReplayMemory():
    
    def __init__(self, capacity = 10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, next_mask, done):
        self.memory.append((state, action, reward, next_state, next_mask, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
def get_bot_move(model, board):
    state = board.get_tensor().float()

    with torch.no_grad():
        q_values = model(state)

        masked_q_values = q_values.clone()
        masked_q_values[0][board.shots_taken == 1] = -float('inf')

        action = masked_q_values.argmax().item()

    return action

if __name__ == "__main__":
    model = DQN_MLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    memory = ReplayMemory()
    epsilon = 1.0

    for episode in range(1000):
        board = BattleShip(random_ships=True)
        state = board.get_tensor().float()
        done = False
        total_reward = 0

        while not done:

            if random.random() < epsilon:
                action = random.randint(0,99)
            else:
                with torch.no_grad():
                    q_values = model(state)
                    masked_q_values = q_values.clone()
                    masked_q_values[0][board.shots_taken == 1] = -1e9

                    action = masked_q_values.argmax().item()
            
            x = action % 10
            y = action // 10
            is_hit = board.shoot((x,y))
            is_sunk = board.add_sunk_ship()

            reward = HIT if is_hit else MISS
            if is_sunk : reward += SINK

            next_state = board.get_tensor().float()
            next_mask = board.shots_taken
            done = board.all_ships_sunk()

            memory.push(state, action, reward, next_state, next_mask, done)
            state = next_state

            if (len(memory.memory) > 64):
                train_step(model, memory, optimizer)

            total_reward += reward
        
        epsilon = max(0.01, epsilon * 0.995)

        if episode % 100 == 0:
            print(f"Episode {episode} | Total Reward: {total_reward} | Epsilon: {epsilon:.2f}")

    torch.save(model.state_dict(), "battleship_dqn.pth")
    print("model saved!")
    results = []
    results_from_random = []
    model.eval()
    for i in range(100):
        test_board = BattleShip(random_ships=True)
        turns = 0
        done = False

        while not done:
            action = get_bot_move(model, test_board)
            x, y = action % 10, action // 10
            test_board.shoot((x,y))
            print(action)
            test_board.add_sunk_ship()

            turns += 1
            done = test_board.all_ships_sunk()
        # while not done_random:
        #     x = randint(0,9)
        #     y = randint(0,9)
        #     if not random_test_board.already_shot((x,y)):
        #         random_test_board.shoot((x, y))
        #         turns_random += 1
        #         done_random = random_test_board.add_sunk_ship()

            results.append(turns)
            print(turns)
        #   results_from_random.append(turns_random)

        print(f"Average turns: {np.mean(results)}")
        #print(f"Average for random: {np.mean(results_from_random)}")