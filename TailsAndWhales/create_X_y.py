import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas as pd
import pickle
from tensorflow.keras.utils import to_categorical
import os

def create_X_y(csv_path = '../data_csv/train_kaggle.csv'):
    # load and make changes to csv
    data = pd.read_csv(csv_path)
    data['speciesv2'] = data['species']
    data['speciesv2'].loc[data['speciesv2'].str.contains('dolphin')] = 'dolphin'
    data['speciesv2'].loc[data['speciesv2'].str.contains('dolpin')] = 'dolphin'
    data['speciesv2'].loc[data['speciesv2'].str.contains('whale')] = 'whale'
    data = data[data['speciesv2'] != 'globis']

    # create and save X in raw-data/preprocessed_data/size_64/
    path = "../raw_data/train_images/"
    images_list = list(data['image'])

    # This block code includes a resizing step
    loaded_images = []
    a = 1
    for index, image in enumerate(images_list):
        print(f" processing photo number {a}")

        # resizing image
        img = mpimg.imread(path + image)
        img = cv2.resize(img, dsize=(64, 64), interpolation= cv2.INTER_LINEAR)
        a +=1

        # reshaphe photo in 3D dimension
        if len(img.shape) != 3:
            img = np.stack((img,)*3, axis=-1)

        loaded_images.append(list(np.array(img)))

    # convert to 3D array
    X = np.array(loaded_images)
    print(X.shape)

    preprocessed_data_path = "../raw_data/preprocessed_data/size_64/"
    # save preprocessed images
    with open(os.path.join(preprocessed_data_path, 'X_total_64.npy'), 'wb') as f:
        np.save(f, X)

    # create and save y in raw_data/preprocessed_data/size_64/
    classes = {'whale':0, 'dolphin':1, 'beluga':2}
    #Add a new column 'class' on data_sample df
    data['class'] = data['speciesv2'].map(classes)
    num_classes = 3
    y = to_categorical(data['class'], num_classes=num_classes)
    with open(os.path.join(preprocessed_data_path, 'y_total_64.npy'), 'wb') as f:
        np.save(f, y)

    print("X and y are saved ! Well Done !")
