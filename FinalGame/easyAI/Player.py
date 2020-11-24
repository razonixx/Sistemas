"""
This module implements the Player (Human or AI), which is basically an
object with an ``ask_move(game)`` method
"""

import socket
from time import time, sleep
from matplotlib.mlab import psd
import numpy as np

try:
    input = raw_input
except NameError:
    pass


UDP_HOST = '127.0.0.1'
UDP_PORT = 8080

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((UDP_HOST, UDP_PORT))

class Human_Player:
    """
    Class for a human player, which gets asked by text what moves
    she wants to play. She can type ``show moves`` to display a list of
    moves, or ``quit`` to quit the game.
    """

    moves = {
        101:1,
        102:2,
        103:3,
        104:4
    }

    def __init__(self, name = 'Human', mlp = None , encoders = None):
        self.name = name
        self.old_move = None
        if(mlp is None or encoders is None):
            print('Error initializing game. Exiting....')
            exit(-1)
        else:
            self.mlp = mlp
            self.encoder = encoders

    def ask_move(self, game):
        chan1 = []
        chan2 = []
        accumPSDPower = []
        possible_moves = game.possible_moves()
        # The str version of every move for comparison with the user input:
        possible_moves_str = list(map(str, game.possible_moves()))
        move = "NO_MOVE_DECIDED_YET"

        while True:
            try:
                while len(chan1) < 256:
                    data = sock.recv(1024*1024)
                    values = np.frombuffer(data)

                    for i in range(0, len(values), 2):
                        chan1.append(values[i])
                        chan2.append(values[i + 1])
                # PSD
                power1, freq1 = psd(chan1, NFFT=256, Fs=256)
                power2, _ = psd(chan2, NFFT=256, Fs=256)
                
                start_index = np.where(freq1 >= 4.0)[0][0]
                end_index = np.where(freq1 >= 60.0)[0][0]
                chan1 = []
                chan2 = []

                power1 = power1[start_index:end_index]
                power2 = power2[start_index:end_index]
                row = np.append(power1, power2)
                accumPSDPower.append(row)

                prediction = self.mlp.predict([accumPSDPower]) # 
                accumPSDPower = []
                classes = list(map(np.argmax, prediction))
                labels = self.encoder.inverse_transform(classes)
                new_move = self.moves[labels[0]]
                if new_move is not None:
                    if str(new_move) in possible_moves_str:
                        accumPSDPower = []
                        print('Label:',labels)
                        # Transform the move into its real type (integer, etc. and return).
                        move = possible_moves[possible_moves_str.index(str(new_move))]
                        self.old_move = move
                        return move
                    elif str(self.old_move) in possible_moves_str:
                        print('Label:',labels, 'invalid, keeping old move')
                        return self.old_move
                    else:
                        print('Label:',labels)
                        print('You lose!')
                        raise FloatingPointError
                        

            except socket.timeout:
                accumPSDPower = []
                pass
            except ConnectionRefusedError:
                print('Check game server')
                sleep(2)
                pass



            

class AI_Player:
    """
    Class for an AI player. This class must be initialized with an
    AI algortihm, like ``AI_Player( Negamax(9) )``
    """

    def __init__(self, AI_algo, name = 'AI'):
        self.AI_algo = AI_algo
        self.name = name
        self.move = {}

    def ask_move(self, game):
        return self.AI_algo(game)
