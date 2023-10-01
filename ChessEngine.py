import copy

'''
This class is responsible for storing all the information about the current state of
the chess game. Also responsible for determining the valid moves at the current state.
'''

class GameState():

    def __init__(self):

        self.board = [
            ["bR","bN","bB","bQ","bK","bB","bN","bR"],
            ["bp","bp","bp","bp","bp","bp","bp","bp"],
            ["--","--","--","--","--","--","--","--",],
            ["--","--","--","--","--","--","--","--",],
            ["--","--","--","--","--","--","--","--",],
            ["--","--","--","--","--","--","--","--",],
            ["wp","wp","wp","wp","wp","wp","wp","wp"],
            ["wR","wN","wB","wQ","wK","wB","wN","wR"]
        ]

        self.moveFunctions = {'p': self.getPawnMoves, 'R': self.getRookMoves, 'N': self.getKnightMoves,
                              'B': self.getBishopMoves,'Q': self.getQueenMoves,'K': self.getKingMoves}

        self.whiteToMove = True
        self.moveLog = []
        self.whiteKingLocation = (7,4)
        self.blackKingLocation = (0,4)
        self.inCheck = False
        self.pins = []
        self.checks = []
        self.checkmate = False
        self.stalemate = False
        self.enpassantPossible = () # coordinates for the square where en passant is possible
        self.enpassantPossibleLog = [self.enpassantPossible]
        self.currentCastlingRight = CastleRights(True,True,True,True)
        self.castleRightsLog = [CastleRights(self.currentCastlingRight.wks, self.currentCastlingRight.bks,
                                             self.currentCastlingRight.wqs, self.currentCastlingRight.bqs)]

    def makeMove(self, move):
        self.board[move.startRow][move.startCol] = "--" # make the initial tile empty
        self.board[move.endRow][move.endCol] = move.pieceMoved # make the final square have the new piece
        self.moveLog.append(move) # log the move
        self.whiteToMove = not self.whiteToMove # change player turn

        

        # updating king's location if moved
        if move.pieceMoved == 'wK':
            self.whiteKingLocation = (move.endRow,move.endCol)
        elif move.pieceMoved == 'bK':
            self.blackKingLocation = (move.endRow,move.endCol)
        

        #pawn promotion
        if move.isPawnPromotion:
            self.board[move.endRow][move.endCol] = move.pieceMoved[0] + 'Q'
        

        #en passant move
        if move.isEnPassantMove:
            self.board[move.startRow][move.endCol] = "--" #capturing the pawn
        
        #update enPassantPossible variable
        if move.pieceMoved[1] == 'p' and abs(move.startRow - move.endRow) == 2: # only on 2 square pawn advances
            self.enpassantPossible = ((move.startRow + move.endRow)//2, move.endCol)
        else:
            self.enpassantPossible = ()

        #castle move

        if move.isCastleMove:
            
            if move.endCol - move.startCol == 2: # kingside castle
                
                self.board[move.endRow][move.endCol-1] = self.board[move.endRow][move.endCol+1] # moves the rook
                self.board[move.endRow][move.endCol+1] = "--" # erase the old rook
            
            else: # queenside castle move
                self.board[move.endRow][move.endCol+1] = self.board[move.endRow][move.endCol-2] # moves the rook
                self.board[move.endRow][move.endCol-2] = "--"
                
        
        self.enpassantPossibleLog.append(self.enpassantPossible)
        #update castling rights - whenver it is a rook or a king move
        self.updateCastleRights(move)
        self.castleRightsLog.append(CastleRights(self.currentCastlingRight.wks, self.currentCastlingRight.bks,
                                             self.currentCastlingRight.wqs, self.currentCastlingRight.bqs))
        

    
    '''
    Undo the last move
    '''
    def undoMove(self):
        if len(self.moveLog) != 0: # make sure there is a move to undo
            move = self.moveLog.pop()
            self.board[move.startRow][move.startCol] = move.pieceMoved
            self.board[move.endRow][move.endCol] = move.pieceCaptured
            self.whiteToMove = not self.whiteToMove
        # updating king's location if moved
        if move.pieceMoved == 'wK':
            self.whiteKingLocation = (move.startRow,move.startCol)
        elif move.pieceMoved == 'bK':
            self.blackKingLocation = (move.startRow,move.startCol)
        
        #undo en passant
        if move.isEnPassantMove:
            self.board[move.endRow][move.endCol] = "--" # leave landing square blank
            self.board[move.startRow][move.endCol] = move.pieceCaptured
            self.enpassantPossible = (move.endRow, move.endCol)
        
        self.enpassantPossibleLog.pop()
        self.enpassantPossible = self.enpassantPossibleLog[-1]

        
        #undo castling rights
        # self.castleRightsLog.pop() # get rid of the new castle rights from the move we are undoing
        # self.currentCastlingRight = self.castleRightsLog[-1] # set the current castle rights to the last one in the list after undo
        self.castleRightsLog.pop()
        castleRights = copy.deepcopy(self.castleRightsLog[-1])
        self.currentCastlingRight = castleRights

        #undo castle move
        if move.isCastleMove:
            if move.endCol - move.startCol == 2:# king side
                self.board[move.endRow][move.endCol+1] = self.board[move.endRow][move.endCol-1]
                self.board[move.endRow][move.endCol-1] = "--"
            else: # queen side
                self.board[move.endRow][move.endCol-2] = self.board[move.endRow][move.endCol+1]
                self.board[move.endRow][move.endCol+1] = "--"
        
        self.checkmate = False
        self.stalemate = False
        


    '''
    Update Castle Rights
    '''
    def updateCastleRights(self, move):
        if move.pieceMoved == 'wK':
            self.currentCastlingRight.wks = False
            self.currentCastlingRight.wqs = False
        elif move.pieceMoved == 'bK':
            self.currentCastlingRight.bks = False
            self.currentCastlingRight.bqs = False
        elif move.pieceMoved == 'wR':
            if move.startRow == 7:
                if move.startCol == 0: #left Rook
                    self.currentCastlingRight.wks = False
                elif move.startCol == 7: # right Rook
                    self.currentCastlingRight.wqs = False
        elif move.pieceMoved == 'bR':
            if move.startRow == 0:
                if move.startCol == 0: #left Rook
                    self.currentCastlingRight.wks = False
                elif move.startCol == 7: # right Rook
                    self.currentCastlingRight.wqs = False
        elif move.pieceCaptured == 'wR': # if rook is captured
            if move.endRow == 7:
                if move.endCol == 0:
                    self.currentCastlingRight.wqs = False
                elif move.endCol == 0:
                    self.currentCastlingRight.wks = False
        elif move.pieceCaptured == 'bR':
            if move.endRow == 0:
                if move.endCol == 0:
                    self.currentCastlingRight.bqs = False
                elif move.endCol == 7:
                    self.currentCastlingRight.bks = False
        


    '''
    All moves considering checks
    '''
    def getValidMoves(self):
        moves = []
        self.inCheck, self.pins, self.checks =  self.checkForPinsAndChecks()
        if self.whiteToMove:
            kingRow = self.whiteKingLocation[0]
            kingCol = self.whiteKingLocation[1]
            
        else:
            kingRow = self.blackKingLocation[0]
            kingCol = self.blackKingLocation[1]
        
        
        
        if len(self.checks) != 0:
            
            if len(self.checks) == 1: # only 1 check, block check or move king
                moves = self.getAllPossibleMoves()
                # to block a check you must move a piece into one of the squares between enemy piece and king
                check = self.checks[0] # check information
                checkRow = check[0]
                checkCol = check[1]
                pieceChecking = self.board[checkRow][checkCol] # enemy piece causing the check
                
                validSquares = [] # squares that piece can move to
                # if knight, must capture knight or move king, other pieces and be blocked
                if pieceChecking[1] == 'N':
                    validSquares.append((checkRow,checkCol))
                    
                    
                else:
                    for i in range(1,8):
                        validSquare =(kingRow +  check[2]*i, kingCol + check[3]*i) # check[2] and check[3] are check directions
                        validSquares.append(validSquare)
                        if validSquare[0] == checkRow and validSquare[1] == checkCol: # once you get to piece end checks
                            break
                # get rid of any moves that don't block check or move king
                for i in range(len(moves)-1,-1,-1): 
                    if moves[i].pieceMoved[1] != 'K': # move doesn't move king so it must block or capture
                        if not (moves[i].endRow, moves[i].endCol) in validSquares: # move doesn't block check or capture piece
                            moves.remove(moves[i])
            else: # double check, king has to move
                self.getKingMoves(kingRow, kingCol,moves)
        else: # not in check so all moves are fine
            moves = self.getAllPossibleMoves()
            if self.whiteToMove:
                self.getCastleMoves(self.whiteKingLocation[0], self.whiteKingLocation[1], moves)
            else:
                self.getCastleMoves(self.blackKingLocation[0], self.blackKingLocation[1], moves)
        
        return moves

    # '''
    # Determines if current player is in check
    # '''
    # def inCheck(self):
    #     if self.whiteToMove:
    #         return self.squareUnderAttack(self.whiteKingLocation[0],self.whiteKingLocation[1])
    #     else:
    #         return self.squareUnderAttack(self.blackKingLocation[0],self.blackKingLocation[1])


    '''
    Determine if the enemy can attack the square r,c
    '''
    def squareUnderAttack(self, r, c):
        self.whiteToMove = not self.whiteToMove # switch to opponent's turn
        oppMoves = self.getAllPossibleMoves()
        

        for move in oppMoves:
            if move.endRow == r and move.endCol == c: #square is under attack
                return True
        self.whiteToMove = not self.whiteToMove # switch turn back
        return False
        


    '''
    Returns if the player is in check, a list of pins, and a list of checks
    '''
    def checkForPinsAndChecks(self):
        pins = [] # squares where the allied pinned piece is and direction pinned from
        checks = [] # squares where enemy is applying a check
        inCheck = False

        if self.whiteToMove:
            enemyColor = 'b'
            allyColor = 'w'
            startRow = self.whiteKingLocation[0]
            startCol = self.whiteKingLocation[1]
        else:
            enemyColor = 'w'
            allyColor = 'b'
            startRow = self.blackKingLocation[0]
            startCol = self.blackKingLocation[1]
        
        # check outward from king for pins and checks, keep track of pins
        directions = ((-1,0),(0,-1),(1,0),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))
        for j in range(len(directions)):
            d = directions[j]
            possiblePin = () # reset possible pins
            for i in range(1,8):
                endRow = startRow + d[0] * i
                endCol = startCol + d[1] * i
                if 0 <= endRow < 8 and 0 <= endCol < 8:
                    endPiece = self.board[endRow][endCol]
                    if endPiece[0] == allyColor and endPiece[1] != 'K':
                        if possiblePin == (): # 1st allied piece could be pinned
                            possiblePin = (endRow, endCol, d[0],d[1])
                        else: # 2nd allied piece, so no pin or check possible in this direction
                            break
                    elif endPiece[0] == enemyColor:
                        type = endPiece[1]
                        # 5 possibilities here based on the type of piece
                        #1. orthogonally away from king and piece is a rook
                        #2. diagonally away from king and piece is a bishop
                        #3. 1 square away diagonally away from king and piece is a pawn
                        #4. any direction and piece is a queen
                        #5. any direction 1 square away and piece is a king

                        if (0 <= j <= 3 and type == 'R') or (4 <= j <= 7 and type == 'B') or \
                                (i == 1 and type =='p' and ((enemyColor == 'w' and 6 <= j <= 7) or (enemyColor == 'b' and 4 <= j <= 5))) or \
                                (type == 'Q' and i<=endRow) or (i == 1 and type == 'K'):
                            if possiblePin == (): # no piece blocking, so check
                                inCheck = True
                                checks.append((endRow,endCol,d[0],d[1]))
                                break
                            else: # piece blocking so pin
                                pins.append(possiblePin)
                                break
                        else: # enemy piece is not applying check
                            break
                else: # off the board
                    break
        # check for knight checks
        knightMoves = ((-2,-1),(-2,1),(-1,2),(-1,-2),(1,-2),(1,2),(2,-1),(2,1))
        for m in knightMoves:
            endRow = startRow + m[0]
            endCol = startCol + m[1]
            if 0 <= endRow < 8 and 0 <= endCol < 8:
                endPiece = self.board[endRow][endCol]
                if endPiece[0] == enemyColor and endPiece[1] == 'N': # enemy knight attacking king
                    inCheck =  True
                    checks.append((endRow,endCol, m[0],m[1]))
        
        return inCheck, pins, checks
                 

                             



    '''
    all moves without considering checks
    '''
    def getAllPossibleMoves(self):
        moves = []

        for r in range(len(self.board)): # number of rows
            for c in range(len(self.board[r])): #number of cols
                turn = self.board[r][c][0]
                if (turn == 'w' and self.whiteToMove) or (turn == 'b' and not self.whiteToMove):
                    piece = self.board[r][c][1]
                    self.moveFunctions[piece](r,c, moves) # calls appropriate piece function to get moves
        
        return moves

    
    '''
    get all pawn moves for the pawn located at row, col and add these moves to the list
    '''
    def getPawnMoves(self, r, c, moves):
        piecePinned = False
        pinDirection = ()
        for i in range(len(self.pins)-1,-1,-1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piecePinned = True
                pinDirection = (self.pins[i][2], self.pins[i][3])
                self.pins.remove(self.pins[i])
                break


        if self.whiteToMove: # white to move
            if self.board[r-1][c] == "--": # if square in front of white pawn is empty
                if not piecePinned or pinDirection == (-1,0):
                    moves.append(Move((r,c),(r-1,c), self.board))
                    if r==6 and self.board[r-2][c] == "--": #if second square of starting pawn position is empty
                        moves.append(Move((r,c),(r-2,c), self.board))
            if c-1 >= 0: # capture to left
                if self.board[r-1][c-1][0] == "b": # enemy piece to capture
                    if not piecePinned or pinDirection == (-1,-1):
                        moves.append(Move((r,c),(r-1,c-1),self.board))
                elif (r-1,c-1) == self.enpassantPossible:
                    moves.append(Move((r,c),(r-1,c-1),self.board,isEnPassantMove= True))
            if c+1 <= 7: # capture to right
                if self.board[r-1][c+1][0] == "b":
                    if not piecePinned or pinDirection == (-1,1):
                        moves.append(Move((r,c),(r-1,c+1),self.board))
                elif (r-1,c+1) == self.enpassantPossible:
                    moves.append(Move((r,c),(r-1,c+1),self.board,isEnPassantMove= True))
        
        else: # black to move
            if self.board[r+1][c] == "--": # if square in front of black pawn is empty
                if not piecePinned or pinDirection == (1,0):
                    moves.append(Move((r,c),(r+1,c), self.board))
                    if r == 1 and self.board[r+2][c] == "--": # second square after starting pawn is empty
                        moves.append(Move((r,c),(r+2,c), self.board))
            if c-1 >= 0: # capture to the left
                if self.board[r+1][c-1][0] == "w":  # enemy piece to capture
                    if not piecePinned or pinDirection == (1,-1):
                        moves.append(Move((r,c),(r+1,c-1), self.board))
                elif (r+1,c-1) == self.enpassantPossible:
                    moves.append(Move((r,c),(r+1,c-1),self.board,isEnPassantMove= True))
            if c+1 <= 7: # capture to the right
                if self.board[r+1][c+1][0] == "w":
                    if not piecePinned or pinDirection == (1,1):
                        moves.append(Move((r,c),(r+1,c+1), self.board))
                elif (r+1,c+1) == self.enpassantPossible:
                    moves.append(Move((r,c),(r+1,c+1),self.board,isEnPassantMove= True))
        


    '''
    get all Rook moves for the Rook located at row, col and add these moves to the list
    '''
    def getRookMoves(self, r, c, moves):
        piecePinned = False
        pinDirection = ()
        for i in range(len(self.pins)-1,-1,-1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piecePinned =  True
                pinDirection = (self.pins[i][2],self.pins[i][3])
                if self.board[r][c][1] != 'Q': # can't remove queen from pin on rook moves, only remove it on bishop moves
                    self.pins.remove(self.pins[i])
                break
        directions = ((-1,0),(0,-1),(1,0),(0,1))
        enemyColor = "b" if self.whiteToMove else "w"

        for d in directions:
            for i in range(1,8):

                endRow = r + d[0] * i
                endCol = c + d[1] * i
                if 0 <= endRow < 8 and 0 <= endCol < 8: # on board
                    if not piecePinned or pinDirection == d or pinDirection == (-d[0], -d[1]):
                        endPiece = self.board[endRow][endCol]
                        if endPiece == "--": # empty square
                            moves.append(Move((r,c),(endRow,endCol), self.board))
                        elif endPiece[0] == enemyColor: # enemy piece to capture
                            moves.append(Move((r,c),(endRow,endCol), self.board))
                            break
                        else: # friendly piece
                            break
                else: # out of bounds
                    break


    '''
    get all Knight moves for the Knight located at row, col and add these moves to the list
    '''
    def getKnightMoves(self, r, c, moves):
        piecePinned = False
        
        for i in range(len(self.pins)-1,-1,-1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piecePinned = True
                self.pins.remove(self.pins[i])
                break

        directions = ((-2,1),(2,1),(2,-1),(-2,-1),(1,-2),(-1,2),(1,2),(-1,-2))
        enemyColor = "b" if self.whiteToMove else "w"

        for d in directions:
            for i in range(1,8):

                endRow = r + d[0] 
                endCol = c + d[1]
                if 0 <= endRow < 8 and 0 <= endCol < 8: # on board
                    if not piecePinned:
                        endPiece = self.board[endRow][endCol]

                        if endPiece == "--": # empty square
                            moves.append(Move((r,c),(endRow,endCol), self.board))
                        elif endPiece[0] == enemyColor: # enemy piece to capture
                            moves.append(Move((r,c),(endRow,endCol), self.board))
                        else: # friendly piece
                            break
                else: # out of bounds
                    break

    '''
    get all King moves for the King located at row, col and add these moves to the list
    '''
    def getKingMoves(self, r, c, moves):
        rowMoves = (-1,-1,-1,0,0,1,1,1)
        colMoves = (-1,0,1,-1,1,-1,0,1)
        allyColor = "w" if self.whiteToMove else "b"

        for i in range(8):
            endRow = r + rowMoves[i]
            endCol = c + colMoves[i]

            if 0 <= endRow < 8 and 0 <= endCol < 8:
                endPiece =self.board[endRow][endCol]
                if endPiece[0] != allyColor: # not an ally piece(empty or enemy piece)
                    # place king on end square and check for checks
                    if allyColor == "w":
                        self.whiteKingLocation = (endRow, endCol)
                    else:
                        self.blackKingLocation = (endRow, endCol)
                    inCheck, pins, checks = self.checkForPinsAndChecks()
                    if not inCheck:
                        moves.append(Move((r,c),(endRow,endCol), self.board))
                    #place king back on original location
                    if allyColor == "w":
                        self.whiteKingLocation = (r,c)
                    else:
                        self.blackKingLocation = (r,c)
        
    

    '''
    Generate all valid castle moves for king at (r,c) and add them to the list of moves
    '''
    def getCastleMoves(self,r,c,moves):
        if self.squareUnderAttack(r, c):
            
            return #can't castle while we are in check
        if (self.whiteToMove and self.currentCastlingRight.wks):
            
            self.getKingsideCastleMoves(r,c,moves)
        elif (not self.whiteToMove and self.currentCastlingRight.bks):
            self.getKingsideCastleMoves(r,c,moves)
            
        
        if (self.whiteToMove and self.currentCastlingRight.wqs):
            
            self.getQueensideCastleMoves(r, c, moves)
        elif (not self.whiteToMove and self.currentCastlingRight.wqs):
            self.getQueensideCastleMoves(r,c,moves)
    

    def getKingsideCastleMoves(self, r, c, moves):
        
        if self.board[r][c+1] == "--" and self.board[r][c+2] == "--":
            
            if not self.squareUnderAttack(r, c+1) and not self.squareUnderAttack(r, c+2):
                
                moves.append(Move((r,c),(r,c+2),self.board, isCastleMove = True))
                


    def getQueensideCastleMoves(self, r, c, moves):
        if self.board[r][c-1] == "--" and self.board[r][c-2] == "--" and self.board[r][c-3]=="--":
            if not self.squareUnderAttack(r,c-1) and not self.squareUnderAttack(r,c-2):
                
                moves.append(Move((r,c),(r,c-2), self.board, isCastleMove=True))
                
    '''
    get all Bishop moves for the Bishop located at row, col and add these moves to the list
    '''
    def getBishopMoves(self, r, c, moves):

        piecePinned = False
        pinDirection = ()
        for i in range(len(self.pins)-1,-1,-1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piecePinned = True
                pinDirection = (self.pins[i][2], self.pins[i][3])
                self.pins.remove(self.pins[i])
                break
        directions = ((-1,-1),(1,-1),(1,1),(-1,1))
        enemyColor = "b" if self.whiteToMove else "w"

        for d in directions:
            for i in range(1,8):

                endRow = r + d[0] * i
                endCol = c + d[1] * i
                if 0 <= endRow < 8 and 0 <= endCol < 8: # on board
                    if not piecePinned or pinDirection == d or pinDirection == (-d[0],-d[1]):

                        endPiece = self.board[endRow][endCol]

                        if endPiece == "--": # empty square
                            moves.append(Move((r,c),(endRow,endCol), self.board))
                        elif endPiece[0] == enemyColor: # enemy piece to capture
                            moves.append(Move((r,c),(endRow,endCol), self.board))
                            break
                        else: # friendly piece
                            break
                else: # out of bounds
                    break
    
    '''
    get all Queen moves for the Queen located at row, col and add these moves to the list
    '''
    def getQueenMoves(self, r, c, moves):
        # self.getRookMoves(r,c, moves)
        # self.getBishopMoves(r,c, moves)
        piecePinned = False
        pinDirection = ()
        for i in range(len(self.pins)-1,-1,-1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piecePinned =  True
                pinDirection = (self.pins[i][2],self.pins[i][3])
                if self.board[r][c][1] != 'Q': # can't remove queen from pin on rook moves, only remove it on bishop moves
                    self.pins.remove(self.pins[i])
                break
        directions = ((-1,0),(0,-1),(1,0),(0,1),(-1,-1),(1,1),(1,-1),(-1,1))
        enemyColor = "b" if self.whiteToMove else "w"

        for d in directions:
            for i in range(1,8):

                endRow = r + d[0] * i
                endCol = c + d[1] * i
                if 0 <= endRow < 8 and 0 <= endCol < 8: # on board
                    if not piecePinned or pinDirection == d or pinDirection == (-d[0], -d[1]):
                        endPiece = self.board[endRow][endCol]
                        if endPiece == "--": # empty square
                            moves.append(Move((r,c),(endRow,endCol), self.board))
                        elif endPiece[0] == enemyColor: # enemy piece to capture
                            moves.append(Move((r,c),(endRow,endCol), self.board))
                            break
                        else: # friendly piece
                            break
                else: # out of bounds
                    break


class CastleRights():
    def __init__(self, wks,bks,wqs,bqs):
        self.wks = wks
        self.bks = bks
        self.wqs = wqs
        self.bqs = bqs
        

class Move():

    # maps keys to values
    # key : value
    ranksToRows = {"1":7,"2":6,"3":5,"4":4,
                   "5":3,"6":2,"7":1,"8":0}
    
    rowsToRanks = {v: k for k,v in ranksToRows.items()}

    filesToCols = {"a":0,"b":1,"c":2,"d":3,
                   "e":4,"f":5,"g":6,"h":7}
    
    colsToFiles = {v: k for k,v in filesToCols.items()}

    def __init__(self, startSq, endSq, board, isEnPassantMove = False, isCastleMove = False):
        self.startRow = startSq[0]
        self.startCol = startSq[1]

        self.endRow = endSq[0]
        self.endCol = endSq[1]

        self.pieceMoved = board[self.startRow][self.startCol]
        self.pieceCaptured = board[self.endRow][self.endCol]

        self.isPawnPromotion = ((self.pieceMoved == 'wp' and self.endRow == 0) or (self.pieceMoved == 'bp' and self.endRow == 7))
        self.isEnPassantMove = isEnPassantMove

        self.isCastleMove = isCastleMove

        if self.isEnPassantMove:
            self.pieceCaptured = 'wp' if self.pieceMoved == 'bp' else 'bp'

        self.moveID = self.startRow * 1000 + self.startCol * 100 + self.endRow * 10 + self.endCol
        
    

    '''
    Overriding the equals method
    '''
    def __eq__(self, other):
        if isinstance(other, Move):
            return self.moveID == other.moveID
        return False

        

    def getChessNotation(self):
        return self.getRankFile(self.startRow,self.startCol) + self.getRankFile(self.endRow,self.endCol)

    
    def getRankFile(self, r, c):
        return self.colsToFiles[c] + self.rowsToRanks[r]

        