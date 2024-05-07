# Chess Bot using CNN and Reinforcement Learning


## State Representation
The idea here is to convert a chess board into a set of tensors to process them through a Convolution Neural Network. A chess
board can be stored as a list of bitboards. A bitboard is a specialized bit array data structure commonly used in computer
systems that play board games, where each bit corresponds to a game board space or piece.

To represent the whole board position using bitboards we need one bitboard for each piece type and one bitboard for the 
empty squares.I 
input size of neural network = 8 (rows) x 8 (cols) x 16 (bitboards)
6 bitboards for white pieces
6 bitboards for black pieces
1 for empty squares
1 for castling rights
1 for en passant
1 for player

The output layer of neural network will consist of neurons equivalent to number of moves on the board. A mask layer is added
to output only valid moves.

## Reward Signal
The main signals should be the ones regarding the result of a match: +1 if win, -1 if lose, 0 if draw.
However, it is better to introduce intermediate signals, so that we can "teach" the agent also during a 
single game and not only at the end.
For example a small reward if by making a move the position improves and a lower score if the position worsen.
We will stockfish analyse function to do this.

## Exploration Strategy
It is crucial to design the agent with an exploration strategy that will allow it to explore new moves instead of picking always the ones that temporarily thinks as best.

Otherwise, the agent will become stuck in some local minimum instead of getting to the best possible solution.

A simple exploration strategy is called epsilon-greedy.

We have a parameter called epsilon that ranges between 0 and 1 and represent the exploration probability.

At the beginning of the training the epsilon is set to a high value, like 1 and it is progressively reduced.

For each step the agent takes a random action with probability epsilon and the "best move" with probability 1 - epsilon.

## Training the agent (Experience replay)

Experience Replay is a replay memory technique used in reinforcement learning where we store
the agent’s experiences at each time-step, e<sub>t</sub> = (s<sub>t</sub>, a<sub>t</sub>, r<sub>t</sub>, s<sub>t+1</sub>)  in a data-set D = e<sub>t</sub>, ..., e<sub>N</sub>  , pooled over many episodes into a replay memory.
We then usually sample the memory randomly for a minibatch of experience, and use this to learn off-policy, as
with Deep Q-Networks
