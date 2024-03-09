import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

##############################                         #####################################################################
##############################     MODELO ORIGINAL     #####################################################################
##############################                         #####################################################################
 
class DQCnnNet(tf.keras.Model):
    """
    Original DQCnnNet
    """
    def __init__(self, inp_shape = (264,18)):
        super(DQCnnNet, self).__init__()
        self.inp_shape = inp_shape

        self.kernel_size_0 = 20
        self.kernel_size_1 = 6
        self.drop_rate = 0.4

        self.conv1 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_0,
                                            activation='relu',
                                            padding= "same",
                                            input_shape=self.inp_shape)
        self.batch_n_1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_0,
                                            activation='relu',
                                            padding= "valid")
        self.batch_n_2 = tf.keras.layers.BatchNormalization()
        self.spatial_drop_1 = tf.keras.layers.SpatialDropout1D(self.drop_rate)
        self.conv3 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding= "valid")
        self.avg_pool1 = tf.keras.layers.AvgPool1D(pool_size=2)
        self.conv4 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding= "valid")
        self.conv5 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding= "valid")
        self.spatial_drop_2 = tf.keras.layers.SpatialDropout1D(self.drop_rate)
        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(296, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(self.drop_rate)
        self.dense2 = tf.keras.layers.Dense(148, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(self.drop_rate)
        self.dense3 = tf.keras.layers.Dense(74, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(self.drop_rate)
        self.out = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, input_tensor):
        conv1 = self.conv1(input_tensor)
        batch_n_1 = self.batch_n_1(conv1)
        conv2 = self.conv2(batch_n_1)
        batch_n_2 = self.batch_n_2(conv2)
        spatial_drop_1 = self.spatial_drop_1(batch_n_2)
        conv3 = self.conv3(spatial_drop_1)
        avg_pool1 = self.avg_pool1(conv3)
        conv4 = self.conv4(avg_pool1)
        conv5 = self.conv5(conv4)
        spatial_drop_2 = self.spatial_drop_2(conv5)
        flat = self.flat(spatial_drop_2)
        dense1 = self.dense1(flat)
        dropout1 = self.dropout1(dense1)
        dense2 = self.dense2(dropout1)
        dropout2 = self.dropout2(dense2)
        #
        dense3   = self.dense3(dropout2)
        dropout3 = self.dropout3(dense3)
        return self.out(dropout3)


##############################                         #####################################################################
##############################         MODELO 2        #####################################################################
##############################                         #####################################################################

class DQCnnNet2(tf.keras.Model):
    """
    - 6 capas convolucionales. Funcion de activación ReLU. 
    - 0 capas de normalización.
    - 3 capas de dropout (0.4 droprate). 
    - 1 capa de pooling (mean).
    - 3 capas fully conected.
    """
    def __init__(self, inp_shape=(264,18)):
        super(DQCnnNet2, self).__init__()
        self.inp_shape = inp_shape

        self.kernel_size_0 = 20
        self.kernel_size_1 = 6
        self.drop_rate = 0  # 0.7 en DQCnnNet2.1

        self.conv1 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_0,
                                            activation='relu',
                                            padding="same",
                                            input_shape=self.inp_shape)
        self.conv2 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_0,
                                            activation='relu',
                                            padding="valid")
        self.conv3 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_0,
                                            activation='relu',
                                            padding="valid")  # Nueva capa convolucional
        self.batch_n_1 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding="valid")
        self.avg_pool1 = tf.keras.layers.AvgPool1D(pool_size=2)
        self.conv5 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding="valid")
        self.spatial_drop_1 = tf.keras.layers.SpatialDropout1D(self.drop_rate)  
        self.conv6 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding="valid")  # Nueva capa convolucional
        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(296, activation='relu')
        self.dense2 = tf.keras.layers.Dense(148, activation='relu')
        self.dense3 = tf.keras.layers.Dense(74, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(self.drop_rate)  # Reducción de la tasa de dropout
        self.dropout2 = tf.keras.layers.Dropout(self.drop_rate)  # Reducción de la tasa de dropout
        self.dropout3 = tf.keras.layers.Dropout(self.drop_rate)  # Reducción de la tasa de dropout
        self.out = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, input_tensor):
        conv1 = self.conv1(input_tensor)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        avg_pool1 = self.avg_pool1(conv4)
        conv5 = self.conv5(avg_pool1)
        spatial_drop_1 = self.spatial_drop_1(conv5)
        conv6 = self.conv6(spatial_drop_1)
        flat = self.flat(conv6)
        dense1 = self.dense1(flat)
        dropout1 = self.dropout1(dense1)
        dense2 = self.dense2(dropout1)
        dropout2 = self.dropout2(dense2)
        dense3 = self.dense3(dropout2)
        dropout3 = self.dropout3(dense3)
        return self.out(dropout3)


##############################                         #####################################################################
##############################         MODELO 3        #####################################################################
##############################                         #####################################################################

class DQCnnNet3(tf.keras.Model):
    """
    - 7 capas convolucionales. Funcion de activación ReLU. 
    - 2 capas de normalización.
    - 3 capas de dropout (0.4 droprate). 
    - 1 capa de pooling (mean).
    - 3 capas fully conected.
    """
    def __init__(self, inp_shape=(264, 18)):
        super(DQCnnNet3, self).__init__()
        self.inp_shape = inp_shape

        self.kernel_size_0 = 20
        self.kernel_size_1 = 6
        self.drop_rate = 0.4  

        self.conv1 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_0,
                                            activation='relu',
                                            padding="same",
                                            input_shape=self.inp_shape)
        self.batch_n1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_0,
                                            activation='relu',
                                            padding="valid")
        self.conv3 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_0,
                                            activation='relu',
                                            padding="valid")
        self.conv4 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding="valid")
        self.conv5 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding="valid")
        self.conv6 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding="valid")
        self.conv7 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding="valid")
        self.batch_n2 = tf.keras.layers.BatchNormalization()
        self.spatial_drop = tf.keras.layers.SpatialDropout1D(self.drop_rate)
        self.avg_pool = tf.keras.layers.AvgPool1D(pool_size=2)
        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(296, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(self.drop_rate)
        self.dense2 = tf.keras.layers.Dense(148, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(self.drop_rate)
        self.dense3 = tf.keras.layers.Dense(74, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(self.drop_rate)
        self.out = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, input_tensor):
        conv1 = self.conv1(input_tensor)
        batch_n1 = self.batch_n1(conv1)
        conv2 = self.conv2(batch_n1)
        batch_n2 = self.batch_n2(conv2)
        conv3 = self.conv3(batch_n2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        spatial_drop = self.spatial_drop(conv7)
        avg_pool = self.avg_pool(spatial_drop)
        flat = self.flat(avg_pool)
        dense1 = self.dense1(flat)
        dropout1 = self.dropout1(dense1)
        dense2 = self.dense2(dropout1)
        dropout2 = self.dropout2(dense2)
        dense3 = self.dense3(dropout2)
        dropout3 = self.dropout3(dense3)
        return self.out(dropout3)
