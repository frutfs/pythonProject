import cv2
import os
from tqdm import tqdm
import numpy as np

class Dataset:
    def __init__(self, class_names, image_size, path_to_train_data, path_to_test_data):
        self.class_names = class_names
        self.IMAGE_SIZE = image_size
        self.path_to_train_data = path_to_train_data
        self.path_to_test_data = path_to_test_data
        self.class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
    def load_data(self):
        data_paths = [self.path_to_train_data, self.path_to_test_data]
        dataset = []
        for data in data_paths:
            images = []
            labels = []
            print('Loading data {}'.format(data))
            for folder in os.listdir(data):
                label = self.class_names_label[folder]
                for file in tqdm(os.listdir(os.path.join(data, folder))):
                    img_path = os.path.join(os.path.join(data, folder), file)
                    image = cv2.imread(img_path)
                    RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(RGB, self.IMAGE_SIZE)
                    images.append(resized)
                    labels.append(label)
            images = np.array(images, dtype='float32')
            labels = np.array(labels, dtype='int32')
            dataset.append((images, labels))
        return dataset