from ChessNN_final import NN_Bot
import chess
import numpy as np
import tensorflow as tf
import os
import random

'''
Building the bot
'''
class ChessBot:
    #Constructor
    def __init__(self, input_model_path=None):

        #Exploration parameters
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01

        # training parameters
        self.gamma = 0.5 # prefer long term or short term rewards, 0:greedy,1:long term

        self.learning_rate = 1e-03 # how fast the weights are updated

        self.MEMORY_SIZE = 512 #how many moves to store

        self.MAX_PRIORITY = 1e+06 # higher priority means it will be included in training

        self.memory = []
        self.batch_size = 16 # samples per training step

        self.policy_net = NN_Bot()

        #load trained model if exists
        if input_model_path is not None and os.path.exists(input_model_path):
            self.policy_net.load_weights(input_model_path)
        
        # define the loss function and optimizer
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    # Convert board into a 3d np.array of 16 bitboards
    def convert_State(self, board):
        piece_bitboards ={}

        for color in chess.COLORS:

            for piece_type in chess.PIECE_TYPES:

                v = board.pieces_mask(piece_type, color)
                symbol = chess.piece_symbol(piece_type)
                i = symbol.upper() if color else symbol
                piece_bitboards[i] = v

        
        piece_bitboards['-'] = board.occupied ^ 2 **64 -1

        #player bitboard(full 1s if player is white, else full 0s)
        player = 2 ** 64 - 1 if board.turn else 0

        #castling rights bitboard
        castling_rights = board.castling_rights

        #en passant bitboard
        en_passant = 0
        ep = board.ep_square
        if ep is not None:
            en_passant |= (1 << ep)

        # bitboards 16 total = 12 for pieces, 1 for empty squares, 1 for
        # player, 1 for castling rights, 1 for en passant
        bitboards = [b for b in piece_bitboards.values()] + [player] + [castling_rights] + [en_passant]

        # for each bitboard transform integer into a matrix of 1s and 0s
        bitarray = np.array([
            np.array([(bitboard >> i & 1) for i in range(64)])
            for bitboard in bitboards
        ]).reshape((16,8,8))

        return bitarray
    
    # get the move index out of 4096 moves
    def get_move_index(self, move):
        index = 64 * (move.from_square) + (move.to_square)
        return index
    
    # get the mask of valid moves + dictionary with valid moves 
    # and indices
    def mask_valid_moves(self, board):

        mask = np.zeros((64,64))
        valid_moves_dict = {}

        for move in board.legal_moves:
            mask[move.from_square, move.to_square] = 1
            valid_moves_dict[self.get_move_index(move)] = move
        
        #flatten the mask
        tensor = tf.constant(mask, dtype=tf.float32)
        tensor = tf.reshape(tensor, shape=[-1])
        return tensor, valid_moves_dict
    
    #Insert a step/move/sample into memory for training as 
    #experience replay
    def remember(self, priority, state, action, reward, next_state, done, valid_moves, next_valid_moves):

        if len(self.memory) >= self.MEMORY_SIZE:

            min_value = self.MAX_PRIORITY
            min_index = 0

            for i, n in enumerate(self.memory):
                if n[0] < min_value:
                    min_value = n[0]
                    min_index = i

            del self.memory[min_index]

        self.memory.append((priority,state, action, reward, next_state, done, valid_moves, next_valid_moves))

    #select an action based on current state
    def select_action(self, board, best_move):
        bit_state = self.convert_State(board)
        valid_moves_tensor, valid_move_dict = self.mask_valid_moves(board)

        if random.uniform(0,1) <= self.epsilon:
            r = random.uniform(0,1)

            if r <= 0.1:
                chosen_move = best_move
            else:
                chosen_move = random.choice(list(valid_move_dict.values()))
        else:
            tensor = tf.convert_to_tensor(bit_state, dtype=tf.float32)
            tensor = tf.expand_dims(tensor,0)
            policy_values = self.policy_net(tensor, valid_moves_tensor)

            chosen_move_index = tf.argmax(policy_values, axis=1)
            chosen_move_index = tf.reshape(chosen_move_index, (1,1))

            if chosen_move_index in valid_move_dict:
                chosen_move = valid_move_dict[chosen_move_index]
            else:
                chosen_move = random.choice(list(board.legal_moves))

        return self.get_move_index(chosen_move), chosen_move, bit_state, valid_moves_tensor

    #decay epsilon(exploration rate)
    def adaptiveEGreedy(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


    # save the trained model
    def save_model(self, path):
        self.policy_net.save_weights(path)
        print("Model Saved!")    


    # train the model with experience replay
    def learn_experience_replay(self, debug = False):
        batch_size = self.batch_size

        if len(self.memory) < batch_size:
            return
        
        priorities = [x[0] for x in self.memory]
        priorities_tot = np.sum(priorities)
        weights = priorities / priorities_tot

        minibatch_indexes = np.random.choice(range(len(self.memory)), size=batch_size, replace=False, p = weights)
        minibatch = [self.memory[x] for x in minibatch_indexes]

        state_list = []
        state_valid_moves = []
        action_list = []
        reward_list = []
        next_state_list = []
        next_state_valid_moves = []
        done_list = []
        for priority, bit_state, action, reward, next_bit_state, done, state_valid_move, next_state_valid_move in minibatch:
            state_list.append(bit_state)
            state_valid_moves.append(state_valid_move)
            action_list.append([action])
            reward_list.append(reward)
            done_list.append(done)

            if not done:
                next_state_list.append(next_bit_state)
                next_state_valid_moves.append(next_state_valid_move)
        
        state_valid_move_tensor = tf.convert_to_tensor(state_valid_moves, dtype=tf.float32)
        next_state_valid_move_tensor = tf.convert_to_tensor(next_state_valid_moves, dtype=tf.float32)
        state_tensor = tf.convert_to_tensor(state_list, dtype=tf.float32)
        action_list_tensor = tf.convert_to_tensor(action_list, dtype=tf.int64)
        reward_list_tensor = tf.convert_to_tensor(reward_list, dtype=tf.float32)
        next_state_tensor = tf.convert_to_tensor(next_state_list, dtype=tf.float32)

        bool_array = [not x for x in done_list]
        not_done_mask = tf.convert_to_tensor(bool_array, dtype=tf.bool)

        policy_action_values = self.policy_net(state_tensor, state_valid_move_tensor)
        policy_action_values = tf.gather(policy_action_values, action_list_tensor, axis=1)

        max_reward_in_next_state = tf.zeros(batch_size, dtype=tf.float32)

        with tf.GradientTape() as tape:
            max_reward_in_next_state = tf.where(not_done_mask, tf.reduce_max(self.policy_net(next_state_tensor, next_state_valid_move_tensor), axis=1), max_reward_in_next_state)
            target_action_values = (max_reward_in_next_state * self.gamma) + reward_list_tensor
            target_action_values = tf.expand_dims(target_action_values, axis=1)
            loss = self.loss_function(target_action_values, policy_action_values)

        for i in range(batch_size):
            predicted_value = policy_action_values[i]
            target_value = target_action_values[i]
            priority = tf.reduce_mean(tf.square(predicted_value - target_value)).numpy()
            sample = list(self.memory[minibatch_indexes[i]])
            sample[0] = priority
            self.memory[minibatch_indexes[i]] = tuple(sample)
        
        gradients = tape.gradient(loss, self.policy_net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_net.trainable_variables))

        if debug:
            print("state_tensor shape", state_tensor.shape)
            print("\naction_list_tensor shape", action_list_tensor.shape)
            print("\naction_list_tensor (chosen move out of 4096)", action_list_tensor)
            print("\npolicy_action_values (expected reward of chosen move)", policy_action_values)
            print("\nnot_done_mask", not_done_mask)
            print("\ntarget_action_values", target_action_values)
            print("\nreward_list_tensor", reward_list_tensor)
            print("\nloss:", loss)

        return loss
