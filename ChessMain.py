'''
This is our main driver file. it will be responsible for handling user input and 
displaying current state of the game
'''

import pygame as p
import ChessEngine
import ChessAI
from ChessBot import ChessBot
from ChessEngine import Move
import random

BOARD_WIDTH = BOARD_HEIGHT = 512
MOVE_LOG_PANEL_WIDTH = 250
MOVE_LOG_PANEL_HEIGHT = BOARD_HEIGHT
DIMENSION = 8
SQ_SIZE = BOARD_HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES ={}

agent = ChessBot(input_model_path="Trained_final_bot_v2")


'''
Initialiaze global dictionary of images
'''

def loadImages():

    pieces = ['wp','wR','wN','wB','wK','wQ','bp','bR','bN','bB','bK','bQ']

    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load("Images/" + piece + ".png"),(SQ_SIZE,SQ_SIZE))




def main():
    ranksToRows = {"1":7,"2":6,"3":5,"4":4,
                   "5":3,"6":2,"7":1,"8":0}
    
    rowsToRanks = {v: k for k,v in ranksToRows.items()}

    filesToCols = {"a":1,"b":2,"c":3,"d":4,
                   "e":5,"f":6,"g":7,"h":8}
    
    colsToFiles = {v: k for k,v in filesToCols.items()}
    p.init()
    screen = p.display.set_mode((BOARD_WIDTH + MOVE_LOG_PANEL_WIDTH,BOARD_HEIGHT))
    clock = p.time.Clock()
    screen.fill(p.Color("white"))
    moveLogFont = p.font.SysFont("Arial",12,False, False)
    gs = ChessEngine.GameState()

    validMoves = gs.getValidMoves()
    moveMade = False # flag variable for when a move is made

    loadImages()

    
    running = True

    sqSelected = () #no squares selected (tuple :(row,col))
    playerClicks = [] #keeps track of player clicks (two tuples: [(6,4),(4,4)]
    gameOver = False
    playerOne = False # if human is playing then true, if computer is playing then False
    playerTwo = True # same as above but for black

    while running:
        humanTurn = (gs.whiteToMove and playerOne) or (not gs.whiteToMove and playerTwo)
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False
            elif e.type == p.MOUSEBUTTONDOWN:
                if not gameOver and humanTurn:
                    location = p.mouse.get_pos() #(x,y) location of the mouse
                    col = location[0] // SQ_SIZE
                    row = location[1] // SQ_SIZE
                    if sqSelected == (row,col) or col >= 8: #the user clicked the same square twice
                        sqSelected =() #deselect the square/piece
                        playerClicks = [] #clear player clicks
                    else:
                        sqSelected = (row,col)
                        playerClicks.append(sqSelected) #append for both 1st and 2nd clicks

                        # possibleMoves = gs.highlightMoves(row,col, gs.board)
                        # print(possibleMoves)
                    if len(playerClicks) == 2: #after 2nd click
                        move  = ChessEngine.Move(playerClicks[0], playerClicks[1], gs.board)
                        print(move.getChessNotation())
                        for i in range(len(validMoves)):
                            if move == validMoves[i]:
                                gs.makeMove(validMoves[i])
                                moveMade = True
                                #gs.moveLog.append(move)
                                sqSelected = ()
                                playerClicks = []
                                break
                        if not moveMade:
                            playerClicks = [sqSelected]
            # Key handlers
            elif e.type == p.KEYDOWN:
                if e.key == p.K_z: # undo when 'z' is pressed
                    gs.undoMove()
                    moveMade = True
                    gameOver = False
                if e.key == p.K_r: # reset the board when 'r' is pressed
                    gs = ChessEngine.GameState()
                    validMoves = gs.getValidMoves()
                    sqSelected = ()
                    playerClicks = []
                    moveMade = False
                    gameOver = False
        #AI move finder
        if not gameOver and not humanTurn:
            AIMove = str(ChessAI.choose_move(agent, gs))
            #print(str(AIMove) + str(type(AIMove)))
            if AIMove is None:
                AIMove = ChessAI.findRandomMove(validMoves)
            found_move = False
            # print(ranksToRows[(AIMove[1])])
            # print(filesToCols[AIMove[0]]-1)
            # print(ranksToRows[(AIMove[3])])
            # print(filesToCols[AIMove[2]]-1)
            print(AIMove)
            new_move = Move((ranksToRows[(AIMove[1])],filesToCols[AIMove[0]]-1),(ranksToRows[(AIMove[3])],filesToCols[AIMove[2]]-1),gs.board)
            print("new_move:" + new_move.getChessNotation() + " || AIMove:"+str(AIMove))
            for i in range(0, len(validMoves)):
                #print(validMoves[i].getChessNotation() + "\\"+ new_move.getChessNotation())
                if validMoves[i].getChessNotation() == new_move.getChessNotation():
                    found_move = True
                    print("found Move")
                    gs.makeMove(validMoves[i])
                    moveMade =  True
                    sqSelected = ()
                    playerClicks = []
            if found_move == False:
                gs.makeMove(ChessAI.findRandomMove(validMoves))
                moveMade =  True
                sqSelected = ()
                playerClicks = []
        
        if moveMade:
            validMoves = gs.getValidMoves()
            moveMade = False
        drawGameState(screen, gs, validMoves,sqSelected, moveLogFont)


        if gs.inCheck and len(validMoves) == 0:# king in check and no valid moves, i.e Checkmate
            gameOver = True
            if gs.whiteToMove:
                drawEndGameText(screen, 'Black wins by checkmate')
            else:
                drawEndGameText(screen, 'White wins by checkmate')
        elif not gs.inCheck and len(validMoves) == 0:# king not in check but no valid moves, so stalemate
            gameOver = True
            drawEndGameText(screen, 'Its a Stalemate')
        clock.tick(MAX_FPS)
        p.display.flip()
    
'''
Highlight the square selected and moves for piece selected
'''
def highlightSquares(screen,gs, validMoves, sqSelected):
    if sqSelected != ():
        r, c = sqSelected
        if gs.board[r][c][0] == ('w' if gs.whiteToMove else 'b'): #sqSelected is a piece that can be moved
            # highlight selected square
            s = p.Surface((SQ_SIZE,SQ_SIZE))
            s.set_alpha(100) # transperancy value -> 0 transparent, 255 opaque
            s.fill(p.Color('blue'))
            screen.blit(s, (c*SQ_SIZE,r*SQ_SIZE))

            #highlight moves from that square
            s.fill(p.Color('yellow'))

            for move in validMoves:
                if move.startRow == r and move.startCol == c:
                    screen.blit(s, (move.endCol*SQ_SIZE, move.endRow*SQ_SIZE))

'''
Responsible for the graphics within the current game state
'''
def drawGameState(screen,gs,validMoves,sqSelected, moveLogFont):
    drawBoard(screen) #draw the tiles of chessboard
    highlightSquares(screen, gs, validMoves, sqSelected)
    drawPieces(screen, gs.board) #draw the pieces
    drawMoveLog(screen, gs, moveLogFont)




'''
Draw the tiles of chessboard
'''
def drawBoard(screen):
    colors = [p.Color("white"), p.Color("gray")]

    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[((r+c)%2)]
            p.draw.rect(screen,color,p.Rect(c*SQ_SIZE,r*SQ_SIZE,SQ_SIZE,SQ_SIZE))


'''
Draw the pieces on the board
'''
def drawPieces(screen, board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]

            if piece != "--":
                screen.blit(IMAGES[piece],p.Rect(c*SQ_SIZE,r*SQ_SIZE,SQ_SIZE,SQ_SIZE))


'''
Draw the move log
'''
def drawMoveLog(screen,gs,font):
    moveLogRect = p.Rect(BOARD_WIDTH, 0, MOVE_LOG_PANEL_WIDTH, MOVE_LOG_PANEL_HEIGHT)
    p.draw.rect(screen, p.Color("black"),moveLogRect)
    moveLog = gs.moveLog
    moveTexts = []
    for i in range(0, len(moveLog),2):
        moveString = str(i//2 + 1) + ". " + moveLog[i].getChessNotation() + " "
        if i+1 < len(moveLog): # just to make sure black made a move
            moveString += moveLog[i+1].getChessNotation()
        moveTexts.append(moveString)
    padding = 5
    textY = padding
    for i in range(len(moveTexts)):
        text = moveTexts[i]
        textObject = font.render(text,0,p.Color("White"))
        textLocation = moveLogRect.move(padding,textY)
        screen.blit(textObject,textLocation)
        textY += textObject.get_height()
    # textObject = font.render(text, 0, p.Color('Gray'))
    # textLocation = p.Rect(0, 0, WIDTH, HEIGHT).move(WIDTH/2 - textObject.get_width()/2, HEIGHT/2 - textObject.get_height()/2)
    # screen.blit(textObject, textLocation)
    # textObject = font.render(text,0,p.Color('Black'))
    # screen.blit(textObject,textLocation.move(2,2))

def drawEndGameText(screen, text):
    font = p.font.SysFont("Helvitca", 32, True, False)
    textObject = font.render(text, 0, p.Color('Gray'))
    textLocation = p.Rect(0, 0, BOARD_WIDTH, BOARD_HEIGHT).move(BOARD_WIDTH/2 - textObject.get_width()/2, BOARD_HEIGHT/2 - textObject.get_height()/2)
    screen.blit(textObject, textLocation)
    textObject = font.render(text,0,p.Color('Black'))
    screen.blit(textObject,textLocation.move(2,2))

if __name__ == "__main__":
    main()
