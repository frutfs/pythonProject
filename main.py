import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from dataset import Dataset
from plots import Utils
from sklearn.utils import shuffle
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten

IMAGE_SIZE = (200, 200)
class_names = ['0 mM', '10 mM', '50 mM', '200 mM', '500 mM']

def create_VGG16_model():
    model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=model.inputs, outputs=model.layers[-5].output)
    return model

def load_trained_model(path_to_model):
    return tf.keras.models.load_model(path_to_model)

def load_images_to_predict(path_to_folder):
    list_of_images = []
    image_names = []
    for root, dirs, files in os.walk(path_to_folder):
        for file in files:
            list_of_images.append(os.path.join(root, file))
            image_names.append(file)
    prepared_images = []
    for image in list_of_images:
        img = cv2.imread(image)
        RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(RGB, (IMAGE_SIZE))
        prepared_images.append(resized)
    prepared_images = np.array(prepared_images, dtype='float32')
    prepared_images = prepared_images / 255.0
    prepared_images = create_VGG16_model().predict(prepared_images)
    return prepared_images

def get_file_names(path_to_data):
    file_names = []
    file_paths = []
    for root, dirs, files in os.walk(path_to_data):
        for file in files:
            file_names.append(file)
            file_paths.append(os.path.join(root, file))
    return file_names, file_paths

def predict_images(path_to_model, path_to_folder):
    prepared_images = load_images_to_predict(path_to_folder)
    prediction = load_trained_model(path_to_model).predict(prepared_images)
    pred_labels = np.argmax(prediction, axis=1)
    return pred_labels

def make_prediction_for_file(file_path):
    empty_list = []
    img = cv2.imread(file_path)
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    RGB_for_plot = np.array(RGB)
    resized = cv2.resize(RGB, (IMAGE_SIZE))
    empty_list.append(resized)
    image = np.array(empty_list, dtype='float32')
    prepared_image = image / 255.0
    features = create_VGG16_model().predict(prepared_image)
    prediction = load_trained_model(os.path.join('saved_model', 'VGG16.h5')).predict(features)
    label = np.argmax(prediction, axis=1)
    for conc in class_names:
        if label == class_names.index(conc):
            label = conc
    fig, ax = plt.subplots()
    ax.imshow(RGB_for_plot)
    ax.set_title(f'Predicted concentration of Mg ions {label}')
    return fig


def make_prediction_for_all_files(path_to_model, path_to_folder):
    pred_labels = predict_images(path_to_model, path_to_folder)
    pred_labels_class_names = []
    for index in pred_labels:
        for name in class_names:
            if index == class_names.index(name):
                pred_labels_class_names.append(name)

    return pred_labels_class_names
    # if os.path.exists('predicted_images'):
    #     pass
    # else:
    #     os.mkdir('predicted_images')
    #     print('predicted_images folder was created!')
def plot_and_save_predicted_images(path_to_folder, pred_labels_class_names, save_folder):
    if os.path.exists(os.path.join(save_folder, 'predicted_images')):
        pass
    else:
        os.mkdir(os.path.join(save_folder, 'predicted_images'))
    file_names, file_paths = get_file_names(path_to_folder)
    for i in range(len(file_paths)):
        img = cv2.imread(file_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = np.array(img)
        plt.imshow(image)
        plt.title(f'Predicted concentration of Mg ions {pred_labels_class_names[i]}')
        plt.axis('off')
        plt.savefig(os.path.join(save_folder, 'predicted_images', file_names[i]))
        plt.close()
#
# if __name__ == '__main__':
#     main(os.path.join('saved_model', 'VGG16.h5'))

IMAGE_SIZE = (200, 200)
class_names = ['0 mM', '10 mM', '50 mM', '200 mM', '500 mM']

if __name__ == '__main__':
    IMAGE_SIZE = (200, 200)
    path_to_train_data = 'database/DNAINKtrain'
    path_to_test_data = 'database/DNAINKtest'
    class_names = ['0 mM', '10 mM', '50 mM', '200 mM', '500 mM']
    classes_number = len(class_names)
    batch_size = 128
    epochs = 15
    validation_split = 0.2
    (train_images, train_labels), (test_images, test_labels) = Dataset(class_names, IMAGE_SIZE, path_to_train_data,
                                                                       path_to_test_data).load_data()
    train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
    train_number = train_labels.shape[0]
    test_number = test_labels.shape[0]
    print('Number of training examples: {}'.format(train_number))
    print('Number of testing examples: {}'.format(test_number))
    print("Each image is of size: {}".format(IMAGE_SIZE))

    Utils.bar_plot(train_labels, test_labels, class_names)
    Utils.pie_plot(train_labels, class_names)

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    Utils.display_examples(class_names, train_images, train_labels)


    model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=model.inputs, outputs=model.layers[-5].output)
    train_features = model.predict(train_images)
    test_features = model.predict(test_images)
    model2 = VGG16(weights='imagenet', include_top=False)
    input_shape = model2.layers[-4].get_input_shape_at(0)
    layer_input = Input(shape=(12, 12, 512))
    x = layer_input
    for layer in model2.layers[-4::1]:
        x = layer(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(classes_number, activation='softmax')(x)
    new_model = Model(layer_input, x)
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    new_model.summary()
    history = new_model.fit(train_features, train_labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    Utils.plot_accuracy_and_loss(history)
    predictions = new_model.predict(test_features)
    pred_labels = np.argmax(predictions, axis=1)
    print("Accuracy : {}".format(accuracy_score(test_labels, pred_labels)))
    Utils.plot_confusion_matrix(test_labels, pred_labels, class_names, normalize=True)
    plt.show()
    new_model.save('saved_model/VGG16_BN.h5')