from easyAI import TwoPlayersGame, Human_Player, AI_Player, Negamax, SSS
import numpy as np
import random
from random import randint
from os import system
def clear():
    system('clear')
size = 20

P1 = 5
P2 = 6

class TronGameController(TwoPlayersGame):
    def __init__(self, players):
        # Define the players
        self.players = players

        # Define who starts the game
        self.nplayer = 1 
        
        
        self.positionXP1 = randint(0, size - 1)
        self.positionYP1 = randint(0, size - 1)
        self.positionXP2 = randint(0, size - 1)
        self.positionYP2 = randint(0, size - 1)
        
        # Define the board
        self.board = np.array([[0 for i in range(size)] for j in range(size)])

    def show(self):
        clear()
        print('(X) Player 1 Position: ', self.positionXP1+1, ',', self.positionYP1+1)
        print('(O) Player 2 Position: ', self.positionXP2+1, ',', self.positionYP2+1)

        self.board[self.positionXP1, self.positionYP1] = P1
        self.board[self.positionXP2, self.positionYP2] = P2
        print('\n' + '\n'.join(
            [' '.join([['.', '<', '>', '^', 'v', 'X', 'O'][self.board[i][j]]
            for j in range(size)]) for i in range(size)])
        )

    def check_pos(self, i, j):
        return i >= 0 and j >= 0 and i < size and j < size

    def possible_moves(self):
        
        res = []
        if self.nplayer == 1:
            if self.check_pos(self.positionXP1, self.positionYP1 - 1) and self.board[self.positionXP1, self.positionYP1 - 1] == 0:
                res.append(1)
            if self.check_pos(self.positionXP1, self.positionYP1 + 1) and self.board[self.positionXP1, self.positionYP1 + 1] == 0:
                res.append(2)
            if self.check_pos(self.positionXP1 - 1, self.positionYP1) and self.board[self.positionXP1 - 1, self.positionYP1] == 0:
                res.append(3)
            if self.check_pos(self.positionXP1 + 1, self.positionYP1) and self.board[self.positionXP1 + 1, self.positionYP1] == 0:
                res.append(4)
            
        
        if self.nplayer == 2:
            if self.check_pos(self.positionXP2, self.positionYP2 - 1) and self.board[self.positionXP2, self.positionYP2 - 1] == 0:
                res.append(1)
            if self.check_pos(self.positionXP2, self.positionYP2 + 1) and self.board[self.positionXP2, self.positionYP2 + 1] == 0:
                res.append(2)
            if self.check_pos(self.positionXP2 - 1, self.positionYP2) and self.board[self.positionXP2 - 1, self.positionYP2] == 0:
                res.append(3)
            if self.check_pos(self.positionXP2 + 1, self.positionYP2) and self.board[self.positionXP2 + 1, self.positionYP2] == 0:
                res.append(4)

        return res
    
    def make_move(self, direction):
        if self.nplayer == 1:
            if direction is 1: #Left
                self.board[self.positionXP1, self.positionYP1] = 1
                self.positionYP1 -= 1
            if direction is 2: #Right
                self.board[self.positionXP1, self.positionYP1] = 2
                self.positionYP1 += 1
            if direction is 3: #Up
                self.board[self.positionXP1, self.positionYP1] = 3
                self.positionXP1 -= 1
            if direction is 4: #Down
                self.board[self.positionXP1, self.positionYP1] = 4
                self.positionXP1 += 1

        if self.nplayer == 2:
            if direction is 1: #Left
                self.board[self.positionXP2, self.positionYP2] = 1
                self.positionYP2 -= 1
            if direction is 2: #Right
                self.board[self.positionXP2, self.positionYP2] = 2
                self.positionYP2 += 1
            if direction is 3: #Up
                self.board[self.positionXP2, self.positionYP2] = 3
                self.positionXP2 -= 1
            if direction is 4: #Down
                self.board[self.positionXP2, self.positionYP2] = 4
                self.positionXP2 += 1
    
    def loss_condition(self):
        if not self.possible_moves(): return True
        if self.positionXP1 >= size or self.positionYP1 >= size:
            #print('A')
            return True
        if self.positionXP1 < 0 or self.positionYP1 < 0:
            #print('B')
            return True
        if self.board[self.positionXP1][self.positionYP1] not in [P1, 0]:
            #print('C')
            return True

        if self.positionXP2 < 0 or self.positionXP2 >= size or self.positionYP2 < 0 or self.positionYP2 >= size:
            return True
        if self.board[self.positionXP2, self.positionYP2] not in [P2, 0]:
            #print('F')
            return True

        return False
    
    def is_over(self):
        return self.loss_condition()
        
    def scoring(self):
        return -100 if self.loss_condition() else 0

def main():

    # Search algorithm of the AI player
    algorithm = Negamax(18)

    # Start the game
    #TronGameController([Human_Player(), Human_Player()]).play()
    #TronGameController([AI_Player(algorithm), AI_Player(SSS(18))]).play()
    TronGameController([Human_Player(), AI_Player(algorithm)]).play()

if __name__ == '__main__':
    main()