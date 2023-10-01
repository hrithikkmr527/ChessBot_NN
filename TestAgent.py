
# import chess
# import chess.pgn
# from ChessBot import ChessBot
# import random

# #Training using Anand's games
# def training_from_dataset(agent, stockfish, board):
#     count = 0
#     with open("2400-2599.pgn") as pgn_file:

        

#         while True:
#             game = chess.pgn.read_game(pgn_file)
#             if game is None:
#                 print("Completed Game: " + str(count))
#                 count = count+1
#                 break

#             #temp = game.board().legal_moves
#             try:
#                 moves = game.next()

#                 while moves != None:
                        

            
#                     # set a standard priority
#                     priority = 1
                    
#                     # convert board in 16 bitboards
#                     state = agent.convert_State(board)
                    
#                     # get valid moves tensor
#                     valid_moves, _ = agent.mask_valid_moves(board)
                    
#                     # choose random move and compute its index (out of 4096)
#                     #random_move = random.choice(list(board.legal_moves))
#                     next_move = moves.move
#                     action = agent.get_move_index(next_move)
#                     print("Got move")
#                     # make random move for white and black and compute reward
#                     board_score_before = stockfish.analyse(board=board, limit=chess.engine.Limit(depth=5))\
#                         ['score'].relative.score(mate_score=10000)
                    
#                     try:
#                         print("Here")
#                         board.push(next_move)
#                         print(board)
                        
#                         #board.push(random.choice(list(board.legal_moves)))

                        
#                         board_score_after = stockfish.analyse(board=board, limit=chess.engine.Limit(depth=4))\
#                             ['score'].relative.score(mate_score=10000)
                        
#                         # divide by 100 to convert from centipawns to pawns score
#                         reward = board_score_after / 100 - board_score_before / 100 - 0.01
                        
#                         # convert board in 16 bitboard
#                         next_state = agent.convert_State(board)
                        
#                         # if board.result() == * the game is not finished
#                         done = board.result() != '*'
                        
#                         # get valid moves tensor
#                         next_valid_moves, _ = agent.mask_valid_moves(board)
                        
#                         # undo white and black moves
#                         #board.pop()
#                         #board.pop()
                        
#                         # store in agent memory
#                         agent.remember(priority, state, action, reward, next_state, done, valid_moves, next_valid_moves)
#                         moves = moves.next()
                        
#                     except:
#                         print("Exception Caught!")
#                         board.reset()
#                         break
#             except:
#                 board.reset()
#                 break
                



# # generate a random training sample
# def generate_random_sample(agent, stockfish, board, game):
    
#     # set a standard priority
#     priority = 1
    
#     # convert board in 16 bitboards
#     state = agent.convert_State(board)
    
#     # get valid moves tensor
#     valid_moves, _ = agent.mask_valid_moves(board)
    
#     # choose random move and compute its index (out of 4096)
#     #random_move = random.choice(list(board.legal_moves))
#     next_move = game.next()
#     action = agent.get_move_index(next_move)
    
#     # make random move for white and black and compute reward
#     board_score_before = stockfish.analyse(board=board, limit=chess.engine.Limit(depth=5))\
#         ['score'].relative.score(mate_score=10000)
    
#     board.push(next_move)
#     board.push(random.choice(list(board.legal_moves)))
    
#     board_score_after = stockfish.analyse(board=board, limit=chess.engine.Limit(depth=5))\
#         ['score'].relative.score(mate_score=10000)
    
#     # divide by 100 to convert from centipawns to pawns score
#     reward = board_score_after / 100 - board_score_before / 100 - 0.01
    
#     # convert board in 16 bitboard
#     next_state = agent.convert_State(board)
    
#     # if board.result() == * the game is not finished
#     done = board.result() != '*'
    
#     # get valid moves tensor
#     next_valid_moves, _ = agent.mask_valid_moves(board)
    
#     # undo white and black moves
#     board.pop()
#     board.pop()
    
#     # store in agent memory
#     agent.remember(priority, state, action, reward, next_state, done, valid_moves, next_valid_moves)


# board = chess.Board()

# stockfish = chess.engine.SimpleEngine.popen_uci(".\stockfish-windows-x86-64-modern\stockfish\stockfish-windows-x86-64-modern.exe")
# agent = ChessBot()

# # with open("Anand.pgn") as pgn_file:

# #     while True:
# #         game = chess.pgn.read_game(pgn_file)
# #         if game is None:
# #             break
# #         generate_random_sample(agent, stockfish, chess.Board(game.board()), game)
        
        

# for i in range(16):
#     print("Training Run: " + str(i))
#     training_from_dataset(agent,stockfish, board)

# print(len(agent.memory))
# agent.learn_experience_replay(debug = True)
# agent.save_model("Trained_bot")



# import matplotlib.pyplot as plt
# import time
# import pandas as pd

# loss =[]
# for i in range(30):
#     loss.append(agent.learn_experience_replay(debug = False))

# plt.plot(loss)
# plt.show()


# generate a random training sample

import random
import chess
from ChessBot import ChessBot
import chess.engine

def generate_random_sample(agent, stockfish, board):
    
    # set a standard priority
    priority = 1
    
    # convert board in 16 bitboards
    state = agent.convert_State(board)
    
    # get valid moves tensor
    valid_moves, _ = agent.mask_valid_moves(board)
    
    # choose random move and compute its index (out of 4096)
    random_move = random.choice(list(board.legal_moves))
    action = agent.get_move_index(random_move)
    
    # make random move for white and black and compute reward
    board_score_before = stockfish.analyse(board=board, limit=chess.engine.Limit(depth=5))\
        ['score'].relative.score(mate_score=10000)
    
    board.push(random_move)
    board.push(random.choice(list(board.legal_moves)))
    
    board_score_after = stockfish.analyse(board=board, limit=chess.engine.Limit(depth=5))\
        ['score'].relative.score(mate_score=10000)
    
    # divide by 100 to convert from centipawns to pawns score
    reward = board_score_after / 100 - board_score_before / 100 - 0.01
    
    # convert board in 16 bitboard
    next_state = agent.convert_State(board)
    
    # if board.result() == * the game is not finished
    done = board.result() != '*'
    
    # get valid moves tensor
    next_valid_moves, _ = agent.mask_valid_moves(board)
    
    # undo white and black moves
    board.pop()
    board.pop()
    
    # store in agent memory
    agent.remember(priority, state, action, reward, next_state, done, valid_moves, next_valid_moves)



board = chess.Board()

stockfish = chess.engine.SimpleEngine.popen_uci(".\stockfish-windows-x86-64-modern\stockfish\stockfish-windows-x86-64-modern.exe")
agent = ChessBot(input_model_path="Trained_final_bot")

for i in range(2000):
    generate_random_sample(agent, stockfish, board)

agent.learn_experience_replay(debug = True)

import matplotlib.pyplot as plt

loss = []
for i in range(1000):
    loss.append(agent.learn_experience_replay(debug=False))

plt.plot(loss)
plt.show()

agent.save_model("Trained_final_bot_v2")

from ChessAI import test

#test(agent, games=5000)