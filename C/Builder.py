import tensorflow as tf
from keras.regularizers import *

class ConvNetBuilder:
    """
    A utility class for building convolutional neural network architectures.

    Args:
        kernel_widths (list): List of kernel widths for convolutional layers.
        filters (list): List of filter numbers for convolutional layers.
        dropouts (list): List of dropout rates for dropout layers.
        layer_types (list): List of layer types ('C' for convolutional, 'N' for normalization, 
                            'P' for pooling, 'flatten' for flattening, 'D' for dropout, and 'F' for dense).
        name (str): Name of the convolutional neural network model.

    Raises:
        ValueError: If there is inconsistency in the provided architecture specifications.

    Methods:
        find_integer_indices(): Identifies the indices of integer elements in the layer_types list.
        build_model(): Constructs a TensorFlow Sequential model based on the provided architecture specifications.
    """
    def __init__(self, kernel_widths, filters, dropouts, layer_types, tags, metadata, test_suite_thresholds):
        # self.name = name 
        self.kernel_widths = kernel_widths
        self.filters = filters
        self.dropouts = dropouts
        self.layer_types = layer_types
        self.integer_indices = []  # List to store the indices of integer elements
        self.tags = tags
        self.metadata = metadata
        self.test_suite_thresholds = test_suite_thresholds

        # Check consistency in list sizes
        num_f_layers = layer_types.count('F')
        num_int_values = sum(1 for val in layer_types if isinstance(val, int))
        if num_f_layers != num_int_values:
            raise ValueError("The number of integer values at the end of the layer_types list must be equal to the number of dense layers.")

    def find_integer_indices(self):
        """
        Identifies the indices of integer elements in the layer_types list.
        """
        # Iterate over the layer_types list and obtain the indices of the integer elements
        for i, element in enumerate(self.layer_types):
            if isinstance(element, int):
                self.integer_indices.append(i)

    def build_model(self):
        """
        Constructs a TensorFlow Sequential model based on the provided architecture specifications.

        Returns:
            tuple: A tuple containing the constructed TensorFlow model and its name.
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input((264, 18)))
        # model.add(tf.keras.layers.GaussianNoise(0.1))

        self.find_integer_indices()  # Calculate the indices of integer elements

        denses = 0  # Counter for dense layers
        drops = 0  # Counter for dropout layers
        kernel_index = 0  # Index for kernel widths list
        filter_index = 0  # Index for filter numbers list

        for idx, layer_type in enumerate(self.layer_types):
            if layer_type == 'C':  # Convolutional layer
                if kernel_index < len(self.kernel_widths) and filter_index < len(self.filters):  # Check if kernel widths and filters are available
                    model.add(tf.keras.layers.Conv1D(filters=self.filters[filter_index], kernel_size=self.kernel_widths[kernel_index], padding='same', activation='relu'))
                    kernel_index += 1
                    filter_index += 1
                else:
                    print("Not enough kernel widths or filter numbers available to add convolutional layers.")
            elif layer_type == 'N':  # Normalization layer
                model.add(tf.keras.layers.BatchNormalization())
            elif layer_type == 'P':  # Pooling layer
                model.add(tf.keras.layers.AveragePooling1D(pool_size=2))
            elif layer_type == 'flatten':  # Flatten layer
                model.add(tf.keras.layers.Flatten())
            elif layer_type == 'D':  # Dropout layer
                if drops < len(self.dropouts):  # Check if dropout rates are available
                    model.add(tf.keras.layers.Dropout(self.dropouts[drops]))
                    drops += 1
                else:
                    print("Not enough dropout rates available to add dropout layers.")
            elif layer_type == 'F':  # Dense layer
                if denses < len(self.integer_indices):  # Check if integer indices are available
                    neurons = self.layer_types[self.integer_indices[denses]]  # Get the number of neurons
                    if denses == len(self.integer_indices) - 1:  # If it's the last dense layer
                        model.add(tf.keras.layers.Dense(neurons, activation='softmax'))  # Use softmax
                    else:
                        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
                    denses += 1
                else:
                    print("Not enough integer indices available to add dense layers.")

        return model, self.tags, self.metadata, self.test_suite_thresholds

