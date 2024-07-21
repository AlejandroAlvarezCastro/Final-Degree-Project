import numpy as np

class Data:
    def __init__(self, base_path):
        """
        Initializes the Data object with a base path for loading data.

        Parameters:
            base_path (str): The base path where data files are located.
        """
        self.base_path = base_path

    def load_data(self, filename):
        """
        Loads data from a .npz file and returns features and labels.

        Parameters:
            filename (str): The filename of the .npz file to load.

        Returns:
            tuple: A tuple containing:
                - features (numpy.ndarray): The feature data.
                - labels (numpy.ndarray): The label data.
        """
        data = np.load(self.base_path + filename)
        features = data['features']
        labels = data['labels']
        return features, labels

    def get_data(self, train_filename, valid_filename, test_filename, assess_filename):
        """
        Loads and concatenates training, validation, and test data.

        Parameters:
            train_filename (str): The filename of the training data file.
            valid_filename (str): The filename of the validation data file.
            test_filename (str): The filename of the test data file.
            assess_filename (str): The filename of the assessment data file.

        Returns:
            tuple: A tuple containing:
                - X (numpy.ndarray): Concatenated features of training, validation, and test data.
                - Y (numpy.ndarray): Concatenated labels of training, validation, and test data.
                - ass_f (numpy.ndarray): Features of assessment data.
                - ass_l (numpy.ndarray): Labels of assessment data.
        """
        train_f, train_l = self.load_data(train_filename)
        valid_f, valid_l = self.load_data(valid_filename)
        test_f, test_l = self.load_data(test_filename)
        ass_f, ass_l = self.load_data(assess_filename)
        
        x_train_valid = np.concatenate((train_f, valid_f), axis=0)
        y_train_valid = np.concatenate((train_l, valid_l), axis=0)

        X = np.concatenate((x_train_valid, test_f), axis=0)
        Y = np.concatenate((y_train_valid, test_l), axis=0)
        
        return X, Y, ass_f, ass_l
