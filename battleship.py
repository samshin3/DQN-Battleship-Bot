import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from random import randint

class BattleShip():

    # Board reads from top left to bottom right (Example: (0,0) is the top left of the grid)
    def __init__(self, random_ships = False):
        self.hitboard = np.zeros(100)
        self.missboard = np.zeros(100)
        self.shipsboard = np.zeros(100)
        self.ships = []
        self.sunk_ships = []
        self.shots_taken = np.zeros(100)

        if (random_ships):
            self.place_random()

    # Function that places a ship for user
    def place_ship(self, ship, posn, is_vertical):
        start = self.coord_to_int(posn)

        if (self.is_out_of_bounds(ship.length, start, is_vertical) or self.is_occupied(ship.length, start, is_vertical)):
            return False

        ship.add_posn(start,is_vertical)
        count = 0

        if (is_vertical):
            i = 10
        else:
            i = 1

        while(count < ship.length):
            self.shipsboard[start] = 1
            start += i
            count += 1

        self.ships.append(ship)
        return True

    # Returns true if hit
    def shoot(self, posn):
        index = self.coord_to_int(posn)
        self.shots_taken[index] = 1

        if (self.shipsboard[index] == 1):
            self.hitboard[index] = 1
            return True
        
        self.missboard[index] = 1
        return False

    # Checks if any ship was sunk from previous shot
    def add_sunk_ship(self):
        for ship in self.ships:
            if (ship.is_ship_sunk(self.hitboard) and ship not in self.sunk_ships):
                self.sunk_ships.append(ship)
                return True
        return False

    @staticmethod
    def coord_to_int(posn):
        x = posn[0]
        y = posn[1]
        return x + (10 * y)
    
    # Prettify ship grid for printing
    def print_grid(self):
        hit = self.hitboard.reshape((10,10))
        miss = self.missboard.reshape((10,10))
        ships = self.shipsboard.reshape((10,10))
        print("Hits:")
        print(f"{hit}\n")
        print("Misses:")
        print(f"{miss}\n")
        print("Ships:")
        print(f"{ships}\n")
    
    def is_occupied(self, length, start, is_vertical):
        count = 0
        if (is_vertical):
            i = 10
        else:
            i = 1
        
        while(count < length):
            if (self.shipsboard[start] == 1):
                return True
            count += 1
            start += i

        return False
    
    def is_out_of_bounds(self, length, start, is_vertical):
        x = start % 10
        y = start // 10
        if is_vertical:
            return y + length > 10
        else:
            return x + length > 10
    
    def get_tensor(self):
        state = np.stack((self.hitboard, self.missboard), axis = 0)
        return torch.from_numpy(state).unsqueeze(0)
    
    def place_random(self):
        carrier = Ship("carrier", 5)
        battleship = Ship("battleship", 4)
        submarine = Ship("submarine", 3)
        patrol_boat = Ship("patrol boat", 2)
        ships = [carrier, battleship, submarine, patrol_boat]

        #Random Grid Placement
        while len(ships) > 0:
            x = randint(0,9)
            y = randint(0,9)
            vertical = randint(0,1)
            if (self.place_ship(ships[-1], (x, y), vertical)):
                ships.pop()

    def all_ships_sunk(self):
        return len(self.sunk_ships) == 4 
    
    def already_shot(self,posn):
        index = self.coord_to_int(posn)
        return self.shots_taken[index] == 1


class Ship():
    
    def __init__(self, name, length):
        self.name = name
        self.length = length
    
    # initialize ship's position on the board
    def add_posn(self, start, is_vertical):
        self.start = start
        self.is_vertical = is_vertical

    # check if ship is sunk
    def is_ship_sunk(self, board):
        count = 0
        point = self.start

        if (self.is_vertical):
            i = 10
        else:
            i = 1
        
        while(count < self.length):
            if board[point] == 0:
                return False
            
            count += 1
            point += i
        return True
        