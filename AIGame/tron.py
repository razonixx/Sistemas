from easyAI import TwoPlayersGame, Human_Player, AI_Player, Negamax
import numpy as np
import random

size = 6

class TronGameController(TwoPlayersGame):
    def __init__(self, players):
        # Define the players
        self.players = players

        # Define who starts the game
        self.nplayer = 1 
        
        
        self.positionXP1 = random.randrange(size) 
        self.positionYP1 = random.randrange(size) 
        self.positionXP2 = random.randrange(size) 
        self.positionYP2 = random.randrange(size) 

        # Define the board
        self.board = np.array([[0 for i in range(size)] for j in range(size)])
    
    def show(self):
        print('(X) Player 1 Position: ', self.positionXP1+1, ',', self.positionYP1+1)
        print('(O) Player 2 Position: ', self.positionXP2+1, ',', self.positionYP2+1)

        self.board[self.positionYP1, self.positionXP1] = 5
        self.board[self.positionYP2, self.positionXP2] = 6

        print('\n' + '\n'.join(
            [' '.join([['.', '<', '>', '^', 'v', 'X', 'O'][self.board[(size-1) - j][i]]
            for i in range(size)]) for j in range(size)])
        )

    #TODO: Como le hacemos para que sepa si puede moverse a un lugar?
    def possible_moves(self):
        res = []
        if self.nplayer == 1:
            if self.positionXP1 < 20 and self.board[self.positionYP1, self.positionXP1-1] == 0:
                res.append(1)
            if self.positionXP1 > 0 and self.board[self.positionYP1, self.positionXP1+1] == 0:
                res.append(2)
            if self.positionYP1 < 20 and self.board[self.positionYP1+1, self.positionXP1] == 0:
                res.append(3)
            if self.positionXP1 > 0 and self.board[self.positionYP1-1, self.positionXP1] == 0:
                res.append(4)
        
        if self.nplayer == 2:
            if self.positionXP2 < 20 and self.board[self.positionYP2, self.positionXP2-1] == 0:
                res.append(1)
            if self.positionXP2 > 0 and self.board[self.positionYP2, self.positionXP2+1] == 0:
                res.append(2)
            if self.positionYP2 < 20 and self.board[self.positionYP2+1, self.positionXP2] == 0:
                res.append(3)
            if self.positionXP2 > 0 and self.board[self.positionYP2-1, self.positionXP2] == 0:
                res.append(4)

        return res
    
    def make_move(self, direction):
        if self.nplayer == 1:
            if direction is 1: #Left
                self.board[self.positionYP1, self.positionXP1] = 1
                self.positionXP1-= 1
            if direction is 2: #Right
                self.board[self.positionYP1, self.positionXP1] = 2
                self.positionXP1+= 1
            if direction is 3: #Up
                self.board[self.positionYP1, self.positionXP1] = 3
                self.positionYP1+= 1
            if direction is 4: #Down
                self.board[self.positionYP1, self.positionXP1] = 4
                self.positionYP1-= 1

        if self.nplayer == 2:
            if direction is 1: #Left
                self.board[self.positionYP2, self.positionXP2] = 1
                self.positionXP2-= 1
            if direction is 2: #Right
                self.board[self.positionYP2, self.positionXP2] = 2
                self.positionXP2+= 1
            if direction is 3: #Up
                self.board[self.positionYP2, self.positionXP2] = 3
                self.positionYP2+= 1
            if direction is 4: #Down
                self.board[self.positionYP2, self.positionXP2] = 4
                self.positionYP2-= 1
    
    #TODO: Porque se muere cuando no se debe de morir?
    #Se muere cuando x=y, y otras veces pero no se porque
    def loss_condition(self):
        if self.positionXP1 >= size or self.positionYP1 >= size:
            print('A')
            return True
        if self.positionXP1 < 0 or self.positionYP1 < 0:
            print('B')
            return True
        if self.board[self.positionXP1, self.positionYP1] != 0:
            print('C')
            return True

        if self.positionXP2 >= size or self.positionYP2 >= size:
            print('D')
            return True
        if self.positionXP2 < 0 or self.positionYP2 < 0:
            print('E')
            return True
        if self.board[self.positionXP2, self.positionYP2] != 0:
            print('F')
            return True

        return False
    
    def is_over(self):
        return self.loss_condition()
        
    def scoring(self):
        return -100 if self.loss_condition() else 0

def main():

    # Search algorithm of the AI player
    algorithm = Negamax(12)

    # Start the game
    TronGameController([Human_Player(), Human_Player()]).play()
    #TronGameController([AI_Player(algorithm), AI_Player(algorithm)]).play()

if __name__ == '__main__':
    main()