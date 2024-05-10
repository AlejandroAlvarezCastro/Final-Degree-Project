import os
from keras.callbacks import ModelCheckpoint, EarlyStopping

class TrainConfig():
    def __init__(self, X, metadata):

        self.BATCH_SIZE = metadata['batch_size']
        self.EPOCHS = metadata['epochs']
        self.STEPS_PER_EPOCH = ((X.size)*0.8) / self.BATCH_SIZE
        self.SAVE_PERIOD = 1
        self.modelPath = os.path.join(os.getcwd(), 'ZN_1D_imgs/bestModel.keras')
    
    def CreateTrainParams(self):
        checkpoint = ModelCheckpoint( # set model saving checkpoints
                        self.modelPath, # set path to save model weights
                        monitor='loss', # set monitor metrics
                        verbose=1, # set training verbosity
                        save_best_only=True, # set if want to save only best weights
                        save_weights_only=False, # set if you want to save only model weights
                        mode='auto', # set if save min or max in metrics
                        save_freq= int(self.SAVE_PERIOD * self.STEPS_PER_EPOCH) # interval between checkpoints
                        )

        earlystopping = EarlyStopping(
                monitor='loss', # set monitor metrics
                min_delta=0.0001, # set minimum metrics delta
                patience=25, # number of epochs to stop training
                restore_best_weights=True, # set if use best weights or last weights
                )
        
        callbacksList = [checkpoint, earlystopping]

        return callbacksList, self.BATCH_SIZE, self.EPOCHS
