import random
#from TestAgent import agent
import tensorflow as tf
import chess
import io

pieceScore = {'K': 0, 'Q': 10, 'R': 5, 'B': 3, 'N':3, 'p':1}
CHECKMATE = 1000
STALEMATE = 0
DEPTH = 3
'''
function to convert Board from ChessEngine file to FEN string.
Output from this is fed to Chess library for easier processing.
'''
def board_to_fen(board):
    # Use StringIO to build string more efficiently than concatenating
    with io.StringIO() as s:
        for row in board:
            empty = 0
            for cell in row:
                c = cell[0]
                if c in ('w', 'b'):
                    if empty > 0:
                        s.write(str(empty))
                        empty = 0
                    s.write(cell[1].upper() if c == 'w' else cell[1].lower())
                else:
                    empty += 1
            if empty > 0:
                s.write(str(empty))
            s.write('/')
        # Move one position back to overwrite last '/'
        s.seek(s.tell() - 1)
        # If you do not have the additional information choose what to put
        s.write(' ')
        return s.getvalue()



def findRandomMove(validMoves):
    return validMoves[random.randint(0, len(validMoves) - 1)]


# '''
# find best move based on material
# '''
# def findBestMove(gs, validMoves):
#     turnMultiplier = 1 if gs.whiteToMove else -1


#     opponentMinMaxScore = CHECKMATE
#     bestPlayerMove = None
#     random.shuffle(validMoves)
#     for playerMove in validMoves:
#         gs.makeMove(playerMove)
#         opponentMoves = gs.getValidMoves()
#         if gs.stalemate:
#             opponentMaxScore = STALEMATE
#         elif gs.checkmate:
#             opponentMaxScore = -CHECKMATE
#         else:
#             opponentMaxScore = -CHECKMATE
#             for opponentMove in opponentMoves:
#                 gs.makeMove(opponentMove)
#                 gs.getValidMoves()
#                 if gs.checkmate:
#                     score = CHECKMATE
#                 elif gs.stalemate:
#                     score = STALEMATE
#                 else:
#                     score = -turnMultiplier * scoreMaterial(gs.board)
#                 if (score > opponentMaxScore):
#                     opponentMaxScore = score
#                 gs.undoMove()
#         if opponentMaxScore < opponentMinMaxScore:
#             opponentMinMaxScore = opponentMaxScore
#             bestPlayerMove = playerMove
#         gs.undoMove()
    
#     return bestPlayerMove
'''
agent chooses move
'''
def choose_move(agent, gs):
    board = chess.Board(board_to_fen(gs.board))
    if agent == 'random':
        print("here")
        chosen_move = random.choice(list(board.legal_moves))
        print(str(chosen_move) + ":Chosen Move")
    else:

        bit_state = agent.convert_State(board)
        valid_moves_tensor, valid_move_dict = agent.mask_valid_moves(board)

        with tf.device('/CPU:0'):
            tensor = tf.convert_to_tensor(bit_state, dtype=tf.float32)
            tensor = tf.expand_dims(tensor, axis=0)
            policy_values = agent.policy_net(tensor, valid_moves_tensor)
            chosen_move_index = tf.argmax(policy_values, axis = 1, output_type=tf.int32)
            chosen_move_index = tf.reshape(chosen_move_index,(1,1))
            chosen_move_index = int(chosen_move_index.numpy())
            if chosen_move_index not in valid_move_dict:
                print("Selecting Random AI Move")
                chosen_move = random.choice(list(board.legal_moves))
            else:
                print("Selecting Intelligent Move")
                chosen_move = valid_move_dict[chosen_move_index]
        
        print(str(chosen_move)+" Move_AI")
    return chosen_move


def test(agent, games=1, board_config=None):

    outcomes = []
    for game in range(games):
        
        done = False
        
        # Create a new standard board
        if board_config is None:
            board = chess.Board()
        
        # Create a board with the desired configuration (pieces and starting positions)
        else:
            board = chess.Board(board_config)
        
        game_moves = 0
        while not done:
            game_moves += 1
            
            # white moves
            if game_moves % 2 != 0:
                board.push(choose_move(agent, board))
            
            # black moves
            else:
                board.push(choose_move('random', board))
                
            done = board.result(claim_draw=True) != '*'

        outcomes.append(board.result(claim_draw=True))

    outcome_dict = {"1-0":"White won", "1/2-1/2":"Draw", "0-1":"Black won"}
    for o in set(outcomes):
        print(f"{o} {outcome_dict[o]}: {round(outcomes.count(o)/len(outcomes)*100)}%")
    agent.save_model("Trained_final_bot")



'''
Helper method to make first recursive call
'''
def findBestMove(gs, validMoves):
    global nextMove
    nextMove = None
    findMoveNegaMaxAlphaBeta(gs, validMoves,DEPTH, -CHECKMATE, CHECKMATE,1 if gs.whiteToMove else -1 )
    #findMoveNegaMax(gs, validMoves, DEPTH, 1 if gs.whiteToMove else -1)
    return nextMove

'''
find the best move using recurison
'''
def findMoveMinMax(gs, validMoves, depth, whiteToMove):
    #global nextMove
    if depth == 0:
        return scoreBoard(gs.board)
    
    if whiteToMove:
        maxScore = -CHECKMATE
        for move in validMoves:
            gs.makeMove(move)
            nextMoves = gs.getValidMoves()
            score = findMoveMinMax(gs,nextMoves, depth -1 , False)
            if score > maxScore:
                maxScore = score
                if depth == DEPTH:
                    nextMove = move
            gs.undoMove()
        return maxScore
    else:
        minScore = CHECKMATE
        for move in validMoves:
            gs.makeMove(move)
            nextMoves = gs.getValidMoves()
            score = findMoveMinMax(gs, nextMoves, depth -1, True)
            if score < minScore:
                minScore = score
                if depth == DEPTH:
                    nextMove = move
            gs.undoMove()
        return minScore    


def findMoveNegaMax(gs, validMoves, depth, turnMultiplier):
    #global nextMove
    if depth == 0:
        return turnMultiplier * scoreBoard(gs)
    
    maxScore = -CHECKMATE
    for move in validMoves:
        gs.makeMove(move)
        nextMoves = gs.getValidMoves()
        score = -findMoveNegaMax(gs, nextMoves, depth-1, -turnMultiplier)
        if score > maxScore:
            maxScore = score
            if depth == DEPTH:
                nextMove = move
        gs.undoMove()
    return maxScore

'''
A postive score from this is good for white, a negative score from this
means its good for black
'''
def scoreBoard(gs, validMoves):
    if gs.checkmate:
        if gs.whiteToMove:
            return -CHECKMATE #black wins
        else:
            return CHECKMATE #white wins
    elif gs.stalemate:
        return STALEMATE

    score = 0
    for row in gs.board:
        for square in row:
            if square[0] == 'w':
                score += pieceScore[square[1]]
            elif square[0] == 'b':
                score -= pieceScore[square[1]]
    

    if gs.whiteToMove:
        score += len(validMoves)
    else:
        score -= len(validMoves)

    return score

'''
Score the board based on material
'''
def scoreMaterial(board):

    score = 0
    for row in board:
        for square in row:
            if square[0] == 'w':
                score += pieceScore[square[1]]
            elif square[0] == 'b':
                score -= pieceScore[square[1]]
    

    return score

'''
NegaMax with Alpha Beta pruning
'''
def findMoveNegaMaxAlphaBeta(gs, validMoves, depth, alpha, beta, turnMultiplier):
    #global nextMove
    if depth == 0:
        return turnMultiplier * scoreBoard(gs,validMoves)
    
    maxScore = -CHECKMATE
    for move in validMoves:
        gs.makeMove(move)
        nextMoves = gs.getValidMoves()
        
        score = -findMoveNegaMaxAlphaBeta(gs, nextMoves, depth-1, -beta, -alpha, -turnMultiplier)
        gs.undoMove()
        if score > maxScore:
            maxScore = score
            if depth == DEPTH:
                nextMove = move
        if maxScore > alpha: # pruning
            alpha = maxScore
        if alpha >= beta:
            break
    return maxScore


##"C:\stockfish-windows-x86-64-modern\stockfish-windows-x86-64-modern