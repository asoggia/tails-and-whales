from tensorflow.keras.applications.vgg16 import VGG16 #vgg16
from tensorflow.keras.applications.resnet import ResNet101 #resnet101
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 #mobilenetv2
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
import numpy as np
import pickle

def load_data():
# Use the following method to load X and y

    with open('data/X_10.pkl', 'rb') as handle: # A voir si le chemin fonctionne.
    X = pickle.load(handle)
    with open('data/y_10.pkl', 'rb') as handle: # A voir si le chemin fonctionne.
    y = pickle.load(handle)

    return X, y

#### code for jupyter noteb: X, y = load_data()

def train_val_test_split(X, y)
# Use the following method to create X_train, y_train, X_val, y_val, X_test, y_test

    first_split = int(X.shape[0]) /6.)
    second_split = first_split + int(X.shape[0] * 0.2)
    X_test, X_val, X_train = X[:first_split], X[first_split:second_split], X[second_split:]
    y_test, y_val, y_train = y[:first_split], y[first_split:second_split], y[second_split:]

    return X_test, X_val, X_train, y_test, y_val, y_train

#### code for jupyter noteb: X_test, X_val, X_train, y_test, y_val, y_train = train_val_test_split(X, y)


def load_model(pretrained_model = 'vgg16'):
# Use the following method to import the pre-trained model, as default vgg16
# Important note about BatchNormalization layers (for MobileNetV2) !!

    if pretrained_model == 'vgg16':
      model = VGG16(weights="imagenet", include_top=False, input_shape=X_train[0].shape)

    if pretrained_model == 'resnet101':
      model = ResNet101(weights="imagenet", include_top=False, input_shape=X_train[0].shape)

    if pretrained_model == 'mobilenetv2':
      model = MobileNetV2(weights="imagenet", include_top=False, input_shape=X_train[0].shape)

    return model

#### code for jupyter noteb: model = load_model()


def set_trainable_layers(model, train = False):
# Use the following method to freeze the convolutional base and decide to retrain or not the last layers

    # Set the first layers to be untrainable
    if train == False:
        model.trainable = False
    elif train == True:
    # Set the last layers to be trainable
        model.trainable = True
    # Fine-tune from this layer onwards
        retrain_layers = int(input('How many layer(s) would you like to re-train? '))
        fine_tune_at = len(model.layers) - retrain_layers
    # Freeze all the layers before the `fine_tune_at` layer
        for layer in model.layers[:fine_tune_at]:
            layer.trainable = False

    return model

#### code for jupyter noteb: model = set_trainable_layers(model, train = False)


def add_last_layers(model):
# Use the following method to take a pre-trained model, set its parameters as non-trainables
# and retrain the last layers(opt) and finally add additional trainable layers on top

    base_model = set_trainable_layers(model)
    flatten_layer = layers.Flatten()
    dense_layer = layers.Dense(500, activation='relu')
    prediction_layer = layers.Dense(3, activation='softmax') ### modify this if necessary


    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer,
        prediction_layer
    ])

    return model
#### code for jupyter noteb: model = add_last_layers(model)

def print_mod_summary(model):
# Use the following method to see how many layers are in the base model and print a model summary

    print ("Number of layers in the base model: ", len(model.layers))
    print (model.summary())

#### code for jupyter noteb: print_mod_summary(model)


def compile_model(model):
# Use the following method to compile the model

    opt = optimizers.Adam(learning_rate=1e-4) #>>> We advise the adam optimizer with learning_rate=1e-4
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics= ['accuracy'])

    return model

#### code for jupyter noteb: model = compile_model(model)


def normalize_data(X_test, X_val, X_train):
# use the following method to normalize X using preprocess_input of
# the VGG16 pre-trained model (need to modify this fct if we use a different model)

    X_train = preprocess_input(X_train)
    X_val = preprocess_input(X_val)
    X_test = preprocess_input(X_test)

    return X_train, X_val, X_test

#### code for jupyter noteb: X_train, X_val, X_test = normalize_data(X_test, X_val, X_train)


def train_model(model, X_train, y_train, X_val, y_val):
# use the following method to train the model

    es = EarlyStopping(monitor = 'val_loss', #>>> we can change this!
                   mode = 'auto', #>>> depends on what is being monitored!
                   patience = 5,
                   verbose = 1,
                   restore_best_weights = True)

    history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=16,
                    callbacks=[es])

    return history, model

#### code for jupyter noteb: history = train_model(X_train, y_train, X_val, y_val)


def evaluate_model(model, X_test, y_test):
# use the following method to evalutate the model

    res_vgg = model.evaluate(X_test, y_test)
    test_accuracy_vgg = res_vgg[-1]
    print(res_vgg)
    print(f"test_accuracy_vgg = {round(test_accuracy_vgg,2)*100} %")

#### code for jupyter noteb: evaluate_model(model, X_test, y_test)


def predict_model(model, X_test, y_test):
# use the following method to print the confusion matrix

    y_pred = model.predict(X_test, verbose=1)
    print(confusion_matrix(y_true = np.argmax(y_test, axis=1), y_pred = np.argmax(y_pred, axis=1)))
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), target_names = ['whale', 'dolphin', 'beluga'], digits=3))

#### code for jupyter noteb: predict_model(model, X_test, y_test)
