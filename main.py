from agent import DQN_MLP
from battleship import *
import torch
from copy import deepcopy
from random import randint

trained_model = DQN_MLP()
trained_model.load_state_dict(torch.load("battleship_dqn.pth"))
trained_model.eval()

def get_bot_move(model, board):
    state = board.get_tensor().float()

    with torch.no_grad():
        q_values = model(state)

        masked_q_values = q_values.clone()
        masked_q_values[0][board.shots_taken == 1] = -float('inf')

        action = masked_q_values.argmax().item()

    return action

results = []
results_from_random = []

for i in range(100):
    test_board = BattleShip(random_ships=True)
    random_test_board = deepcopy(test_board)
    turns = 0
    turns_random = 0
    done = False
    done_random = False

    while not done:
        action = get_bot_move(trained_model, test_board)
        x, y = action % 10, action // 10
        test_board.shoot((x,y))
        print(action)
        test_board.add_sunk_ship()

        turns += 1
        done = test_board.all_ships_sunk()
    while not done_random:
        x = randint(0,9)
        y = randint(0,9)
        if not random_test_board.already_shot((x,y)):
            random_test_board.shoot((x, y))
            turns_random += 1
            done_random = random_test_board.add_sunk_ship()

    results.append(turns)
    print(turns)
    results_from_random.append(turns_random)

print(f"Average turns: {np.mean(results)}")
print(f"Average for random: {np.mean(results_from_random)}")