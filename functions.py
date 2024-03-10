import sys, os, datetime, pickle, time
import string, pdb, tqdm
import random, cv2, keras, os.path
import pandas as pd
import numpy as np

import plotly.graph_objects as go

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import ProbClassificationPerformanceTab

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import keras
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def simulations(num_simulations, model, train_f, train_l, valid_f, valid_l, ass_f, ass_l):

    learning_rate   = 2e-4
    BATCH_SIZE      = 50
    STEPS_PER_EPOCH = train_l.size / BATCH_SIZE
    SAVE_PERIOD     = 1
    epochs = 100

    #
    loss = tf.keras.losses.categorical_crossentropy
    # loss = tf.keras.losses.binary_crossentropy

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []

    save_path = os.path.join(os.getcwd(), 'ZN_1D_imgs/')
    modelPath = os.path.join(os.getcwd(), 'ZN_1D_imgs/bestModel.h5')

    # Perform 10 simulations of network training
    for _ in range(10):
        # Create the model
        model = model

        # Compile the model
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        # Configure checkpoints
        checkpoint = keras.callbacks.ModelCheckpoint(
            modelPath,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            save_freq=int(SAVE_PERIOD * STEPS_PER_EPOCH)
        )

        earlystopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=25,
            restore_best_weights=True
        )

        callbacksList = [checkpoint, earlystopping]

        # Train the model
        hist = model.fit(train_f, train_l, epochs=epochs, batch_size=BATCH_SIZE,
                        validation_data=(valid_f, valid_l), callbacks=callbacksList) 

        # Save training history
        with open(os.path.join(save_path, f"hist_{_}.pkl"), "wb") as file:
            pickle.dump(hist.history, file)

        # Get predictions on the independent set
        yPredClass = np.argmax(model.predict(ass_f), axis=-1)
        yTestClass = np.argmax(ass_l, axis=1)

        # Calculate metrics per class
        precision = precision_score(yTestClass, yPredClass, average=None)
        recall = recall_score(yTestClass, yPredClass, average=None)
        f1 = f1_score(yTestClass, yPredClass, average=None)
        accuracy = accuracy_score(yTestClass, yPredClass)

        # Store metrics for this simulation
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)

    # Convert lists to arrays to facilitate calculation of mean and standard deviation
    precision_scores = np.array(precision_scores)
    recall_scores = np.array(recall_scores)
    f1_scores = np.array(f1_scores)
    accuracy_scores = np.array(accuracy_scores)

    # Calculate mean and standard deviation of metrics per class
    mean_precision = np.mean(precision_scores, axis=0)
    mean_recall = np.mean(recall_scores, axis=0)
    mean_f1 = np.mean(f1_scores, axis=0)
    mean_accuracy = np.mean(accuracy_scores)

    std_dev_precision = np.std(precision_scores, axis=0)
    std_dev_recall = np.std(recall_scores, axis=0)
    std_dev_f1 = np.std(f1_scores, axis=0)
    std_dev_accuracy = np.std(accuracy_scores)

    # Print results
    print("\nMean Precision:", mean_precision)
    print("Mean Recall:", mean_recall)
    print("Mean F1-score:", mean_f1)
    print("Mean Accuracy:", mean_accuracy)

    print("\nStandard Deviation of Precision:", std_dev_precision)
    print("Standard Deviation of Recall:", std_dev_recall)
    print("Standard Deviation of F1-score:", std_dev_f1)
    print("Standard Deviation of Accuracy:", std_dev_accuracy)

    # Obtain a plot to visualize results
    fig = plot_metrics(mean_precision, std_dev_precision, mean_recall, std_dev_recall, mean_f1, std_dev_f1)

    return fig, mean_precision, std_dev_precision, mean_recall, std_dev_recall, mean_f1, std_dev_f1, mean_accuracy, std_dev_accuracy


def plot_metrics(mean_precision, std_dev_precision, mean_recall, std_dev_recall, mean_f1, std_dev_f1):
    metrics = {
        'Precision': {
            'mean': mean_precision,
            'stddev': std_dev_precision,
            'color': 'rgb(153, 43, 132)'},
        'Recall': {
            'mean': mean_recall,
            'stddev': std_dev_recall,
            'color': 'rgb(255, 153, 51)'},
        'F1-Score': {
            'mean': mean_f1,
            'stddev': std_dev_f1,
            'color': 'rgb(51, 153, 255)'}}

    classes = ['OK', 'NOK']

    # Create the figure
    fig = go.Figure()

    # Add bars for each metric
    for metric_name, metric_data in metrics.items():
        fig.add_trace(go.Bar(
            x=classes,
            y=metric_data['mean'],
            error_y=dict(type='data', array=metric_data['stddev'], visible=True),
            name=metric_name,
            marker_color=metric_data['color']))

    # Update the layout of the plot
    fig.update_layout(
        title='Metrics per class',
        xaxis=dict(title='Classes'),
        yaxis=dict(title='Value'),
        yaxis_tickformat=".2%",
        barmode='group',
        bargap=0.15)

    return fig
