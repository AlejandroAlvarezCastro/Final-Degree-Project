import numpy as np

class Data():
    def __init__(self, base_path):
        self.base_path = base_path

    def load_data(self, filename):
        data = np.load(self.base_path + filename)
        features = data['features']
        labels = data['labels']
        return features, labels

    def get_data(self, train_filename, valid_filename, test_filename, assess_filename):
        train_f, train_l = self.load_data(train_filename)
        valid_f, valid_l = self.load_data(valid_filename)
        test_f, test_l = self.load_data(test_filename)
        ass_f, ass_l = self.load_data(assess_filename)
        
        x_train_valid = np.concatenate((train_f, valid_f), axis=0)
        y_train_valid = np.concatenate((train_l, valid_l), axis=0)

        X = np.concatenate((x_train_valid, test_f), axis=0)
        Y = np.concatenate((y_train_valid, test_l), axis=0)
        
        return X, Y, ass_f, ass_l



