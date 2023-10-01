import tensorflow as tf

'''
Final Mask Layer
'''
class MaskLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MaskLayer, self).__init__()

    def call(self, x, mask):
        return tf.multiply(x, mask)



'''
Creating the layers of neural network 
'''
class NN_Bot(tf.keras.Model):

    def __init__(self):
        super(NN_Bot, self).__init__()

        #input size = 8(rows) x 8(cols) x 16(bitboards)
        # 6 bitboards for white pieces
        # 6 bitboards for black pieces
        # 1 for empty squares
        # 1 for castling rights
        # 1 for en passant
        # 1 for player

        # building the layers

        #first layer 8x8x16 => 8x8x32
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        #second layer 8x8x32 => 8x8x64
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()

        #third layer 8x8x64 => 8x8x128
        self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()

        #first fully connected layer 8192 => 8192
        self.fc1 = tf.keras.layers.Dense(128*64, activation='relu')

        #second fully connected layer 8192 => 4096
        self.fc2 = tf.keras.layers.Dense(64 * 64)


        # Mask layer
        self.mask = MaskLayer()


    def call(self, x, mask= None, debug= False):

        #conv1 + bn1 with activation function ReLU
        x = tf.nn.relu(self.bn1(self.conv1(x)))

        #conv2 + bn2 with activation function ReLU
        x = tf.nn.relu(self.bn2(self.conv2(x)))

        #conv3 + bn3 with activation function ReLU
        x = tf.nn.relu(self.bn3(self.conv3(x)))

        # flatten to transform data from 3d 8x8x128 to 1d 8192
        x = tf.keras.layers.Flatten()(x)

        # fully connected with activation function ReLU
        x = tf.nn.relu(self.fc1(x))

        # fully connected without ReLU
        x = self.fc2(x)

        # apply mask to set to 0 for all invalid moves
        if mask is not None:
            x = self.mask(x, mask)

        return x