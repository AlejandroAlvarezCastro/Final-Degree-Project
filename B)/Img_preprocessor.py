import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Preprocessor:
    """
    Class for preprocessing image data and visualizing images.

    Attributes:
        npzfile_train (numpy.lib.npyio.NpzFile): NumPy NPZ file containing training data.
        npzfile_validation (numpy.lib.npyio.NpzFile): NumPy NPZ file containing validation data.
        npzfile_test (numpy.lib.npyio.NpzFile): NumPy NPZ file containing test data.
        npzfile_assess (numpy.lib.npyio.NpzFile): NumPy NPZ file containing assessment data.

    Methods:
        preprocess(goal_preprocessed: str)
            Preprocesses the image data and saves the processed images and labels.

        visualize_first_image_assess(features_dir_assess: str, labels_dir_assess: str)
            Visualizes the first image in the assessment dataset along with its label.

        visualize_images_assess(features_dir_assess: str, labels_dir_assess: str)
            Visualizes all the images in the assessment dataset in a grid format.
    """
    def __init__(self, npzfile_train, npzfile_validation, npzfile_test, npzfile_assess):
        self.npzfile_train = npzfile_train
        self.npzfile_validation = npzfile_validation
        self.npzfile_test = npzfile_test
        self.npzfile_assess = npzfile_assess
    
    def preprocess(self, goal_preprocessed):
        """
        Preprocesses the image data and saves the processed images and labels.

        Args:
            goal_preprocessed (str): Directory path where the preprocessed data will be saved.
        """
        # Define classes
        class_OK = "0"
        class_NOK = "1"

        # List of tuples (file name, data, labels)
        datasets = [('train', self.npzfile_train), ('validation', self.npzfile_validation), ('test', self.npzfile_test), ('assess', self.npzfile_assess)]

        # Iterate over each dataset
        for dataset_name, npzfile in datasets:
            # Create directory for current dataset
            dataset_dir = os.path.join(goal_preprocessed, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            # Create directory for features and labels
            features_dir = os.path.join(dataset_dir, 'features')
            labels_dir = os.path.join(dataset_dir, 'labels')
            os.makedirs(features_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            # Get features and labels from the dataset
            features = npzfile['features']
            labels = npzfile['labels']

            # Define the new desired size for the images (square)
            new_size_y = 264  # Choose desired size for y-axis (pixels)
            new_size_x = new_size_y  # Make images square

            # Create an array to store the processed images
            processed_features = np.zeros((features.shape[0], new_size_y, new_size_x), dtype=np.uint8)

            # Iterate over each image in the dataset
            for i in range(features.shape[0]):
                image = features[i]

                # Process the image similar to the training set
                repetition_factor = int(np.ceil(new_size_y / image.shape[1])) 
                repeated_image = np.repeat(image, repetition_factor, axis=1)
                resized_image = repeated_image[:, :new_size_y]
                scaled_image = (resized_image + 1) * 127.5
                scaled_image = np.clip(scaled_image, 0, 255)
                grayscale_image = scaled_image.astype(np.uint8)

                # Save image in features directory as PNG
                image_file_name = f'image_{i}.png'
                image_png = Image.fromarray(grayscale_image)
                image_png.save(os.path.join(features_dir, image_file_name))

                # Create and save label in labels directory
                label_file_name = f'image_{i}.txt'

                # Determine class based on label
                class_label = class_OK if np.array_equal(labels[i], [1., 0.]) else class_NOK
                
                # Save label in text file
                with open(os.path.join(labels_dir, label_file_name), 'w') as file:
                    # Write class followed by normalized bounding box coordinates
                    file.write(f'{class_label} 0.5 0.5 1.0 1.0')  # Bounding box covers entire image

    def visualize_first_image_assess(self, features_dir_assess, labels_dir_assess):
        """
        Visualizes the first image in the assessment dataset along with its label.

        Args:
            features_dir_assess (str): Directory path where the assessment dataset images are stored.
            labels_dir_assess (str): Directory path where the assessment dataset labels are stored.
        """
        # Get the first text file (txt) from labels folder
        first_label_file = os.listdir(labels_dir_assess)[0]
        label_path = os.path.join(labels_dir_assess, first_label_file)

        # Read class from contents of the text file
        with open(label_path, 'r') as file:
            label_content = file.readline().strip().split()
            class_label = label_content[0]

        # Set title according to class
        title = "OK" if class_label == "0" else "NOK"

        # Load the first image from the PNG file
        first_file = os.listdir(features_dir_assess)[0]
        image_path = os.path.join(features_dir_assess, first_file)
        assess_image = Image.open(image_path)

        # Visualize the image with the appropriate title
        plt.imshow(assess_image, cmap='gray')
        plt.axis('off')
        plt.title(f'Label: {title}')
        plt.show()

    def visualize_images_assess(self, features_dir_assess, labels_dir_assess):
        """
        Visualizes all the images in the assessment dataset in a grid format.

        Args:
            features_dir_assess (str): Directory path where the assessment dataset images are stored.
            labels_dir_assess (str): Directory path where the assessment dataset labels are stored.
        """
        # Get the list of files from features and labels folder
        features_files_assess = sorted(os.listdir(features_dir_assess))
        labels_files_assess = sorted(os.listdir(labels_dir_assess))

        # Set up the grid to display the images
        num_rows = 6 
        num_columns = 10
        num_images = num_rows * num_columns

        # Create a figure and axes for the image grid
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, 12))

        # Iterate over the rows and columns of the grid
        for i in range(num_rows):
            for j in range(num_columns):
                # Calculate the index of the current image in the files list
                image_index = i * num_columns + j

                if image_index < len(features_files_assess):
                    # Load the image from the PNG file
                    image_file_name = features_files_assess[image_index]
                    image_path = os.path.join(features_dir_assess, image_file_name)
                    image = plt.imread(image_path)

                    # Load the label from the text file
                    label_file_name = labels_files_assess[image_index]
                    label_path = os.path.join(labels_dir_assess, label_file_name)
                    with open(label_path, 'r') as file:
                        label_content = file.readline().strip().split()  
                        class_label = label_content[0]  

                    # Convert class to a readable label
                    title = 'OK' if class_label == '0' else 'NOK'

                    # Show the image in the corresponding subplot
                    ax = axes[i, j]
                    ax.imshow(image, cmap='gray')
                    ax.axis('off')
                    ax.set_title(f'Label: {title}')

        # Adjust spacing and display the image grid
        plt.tight_layout()
        plt.show()