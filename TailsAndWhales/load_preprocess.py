# imports

import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas as pd
import pickle
from tensorflow.keras.utils import to_categorical
from google.colab import auth


"""def create_X_y(number_of_photos):
    path = 'tails-and-whales/train_kaggle.csv'
    data = pd.read_csv(path)
    data['speciesv2'] = data['species']
    data['speciesv2'].loc[data['speciesv2'].str.contains('dolphin')] = 'dolphin'
    data['speciesv2'].loc[data['speciesv2'].str.contains('dolpin')] = 'dolphin'
    data['speciesv2'].loc[data['speciesv2'].str.contains('whale')] = 'whale'

    # create and save X in raw-data/X.pkl
    path = "/content/tails-and-whales/train_images/"
    images_list = list(data['image'])

    # This block code includes a resizing step
    loaded_images = []
    a = 1
    for image in images_list[:number_of_photos]:   #Select this value in the parameters
        print(a)
        img = mpimg.imread(path + image)
        img = cv2.resize(img, dsize=(256, 256), interpolation= cv2.INTER_LINEAR)
        loaded_images.append(np.array(img))
        a +=1

    # Code a function which will detect the index of B&W images
    list_index_bw = []
    list_len = []
    for index,img in enumerate(loaded_images):
        if len(img.shape) != 3:
            list_index_bw.append(index)
            list_len.append(len(img.shape))
    for i in list_index_bw:
        loaded_images[i] = np.stack((loaded_images[i],)*3, axis=-1)


    X = np.array(loaded_images)
    with open("data/X.pkl", "wb") as fp:   #Pickling
        pickle.dump(X, fp)

    # create and save y in raw_data/y.pkl
    classes = {'whale':0, 'dolphin':1, 'beluga':2}
    #Add a new column 'class' on data_sample df
    data['class'] = data['speciesv2'].map(classes)
    num_classes = 3
    y = to_categorical(data['class'][:number_of_photos], num_classes=num_classes)
    with open("data/y.pkl", "wb") as fp:   #Pickling
        pickle.dump(y, fp)

    print("X and y are saved ! Well Done !")"""


def create_X_y(csv_path = '../raw_data/train_kaggle.csv'): #number of photos could be a parameter
    # load and make changes to csv
    data = pd.read_csv(csv_path)
    data['speciesv2'] = data['species']
    data['speciesv2'].loc[data['speciesv2'].str.contains('dolphin')] = 'dolphin'
    data['speciesv2'].loc[data['speciesv2'].str.contains('dolpin')] = 'dolphin'
    data['speciesv2'].loc[data['speciesv2'].str.contains('whale')] = 'whale'
    data = data[data['speciesv2'] != 'globis']
    # create and save X in raw-data/X.pkl
    path = "../raw_data/train_images/"
    images_list = list(data['image'])

    # This block code includes a resizing step
    loaded_images = []
    a = 1
    for index, image in enumerate(images_list):   #number of photos [:number_of_photos]
        print(f" processing photo number {a}")

        # resizing image
        img = mpimg.imread(path + image)
        img = cv2.resize(img, dsize=(256, 256), interpolation= cv2.INTER_LINEAR)
        a +=1

        # reshaphe photo in 3D dimension
        if len(img.shape) != 3:
            #list_index_bw.append(index)
            #list_len.append(len(img.shape))
            img = np.stack((img,)*3, axis=-1)

        loaded_images.append(np.array(img))

    # convert to 3D array
    X = np.array(loaded_images)
    print(X.shape)
    # get gcp bucket path

    preprocessed_data_path = "../raw_data/preprocessed_data/"
    # save preprocessed images to GCP bucket
    with open(os.path.join(preprocessed_data_path,"X.pkl"), "wb") as fp:   #Pickling
        print(fp)
        pickle.dump(X, fp)

    # create and save y in raw_data/y.pkl
    classes = {'whale':0, 'dolphin':1, 'beluga':2}
    #Add a new column 'class' on data_sample df
    data['class'] = data['speciesv2'].map(classes)
    num_classes = 3
    y = to_categorical(data['class'], num_classes=num_classes) #number of photos [:number_of_photos]
    with open(os.path.join(preprocessed_data_path,"y.pkl"), "wb") as fp:   #Pickling
        pickle.dump(y, fp)

    print("X and y are saved ! Well Done !")
