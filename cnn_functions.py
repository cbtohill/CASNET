#importing modules
import matplotlib.pyplot as plt
import numpy as np 
import os 
import tensorflow as tf 
import pandas as pd
import matplotlib as mlp
import scipy
from scipy import stats
from astropy.io import fits 
from matplotlib.colors import LogNorm
from sklearn.utils import shuffle
import time
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, model_from_json, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam, Adadelta, Adamax, SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import expand_dims




# If using multiple GPUs, only use one of them 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# avoid hogging all the GPU memory
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config) 


def get_data(img_filename, val_filename):
    """
    Function to load training images and corrresponding parameters to train the network. 

    Inputs : 
        img_filename: filename containing the images 
        val_filename: filename containing the corresponding values 

    Returns: 
        x_train, x_test and x_val: Image arrays for training and testing 
        y_train, y_test and y_val: Corresponding parameter arrays 
    """
    test_imgs = np.load(img_filename)
    test_vals = np.load(val_filename)

    x_train = test_imgs
    y_train = test_vals

    # Split data into training, test and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 4)
    x_test,x_val, y_test, y_val = train_test_split(x_val, y_val, test_size = 0.5, random_state = 4)

    #Shuffling data
    x_train, y_train = shuffle(x_train, y_train)

    #Reshapping data to match input needed for network (number of images, (image size), 1)
    x_train = np.expand_dims(x_train, axis = 3)
    x_test = np.expand_dims(x_test, axis = 3)
    x_val = np.expand_dims(x_val, axis = 3)

    print("Number of training/test/validation images: "
          f"{len(x_train)}/{len(x_test)}/{len(x_val)}")
    return x_train, x_test, x_val, y_train, y_test, y_val

# plot of loss and mean_squared_error 
def histplot(history):
    """
    Plotting the training curves for loss and mean absolute error. 
    """
    hist = pd.DataFrame(history.history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    hist.plot(y=['loss', 'val_loss'], ax=ax1)
    min_loss = hist['val_loss'].min()
    ax1.hlines(min_loss, 0, len(hist), linestyle='dotted',
               label='min(val_loss) = {:.3f}'.format(min_loss))
    ax1.legend(loc='upper right')
    hist.plot(y=['mean_absolute_error', 'val_mean_absolute_error'], ax=ax2)
    min_acc = hist['val_mean_absolute_error'].min()
    ax2.hlines(min_acc, 0, len(hist), linestyle='dotted',
               label='min(val_mean_absolute_error) = {:.3f}'.format(min_acc))
    ax2.legend(loc='upper right')
    plt.savefig('Training_curve.png')


#RMSE     
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def pearson_r(y_true, y_pred):
    """
    Pearson coefficient between network prediction (y_pred) and true values (y_true). 
    """
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

def predict_plot(pred, true):
    """
    Plot of trained networks predictions vs the 'true' inputted values. 
    """
    fig = plt.figure()
    plt.hist2d(pred.squeeze(), true, bins =50,  norm=mlp.colors.LogNorm(),cmap = 'plasma',cmin=1)
    cbar = plt.colorbar()
    cbar.ax.tick_params( color='k', size = 5, labelsize = 15)
    #Change limits to match domain of training values
    plt.plot((-0.35, 1.2), (-0.35, 1.2))
    plt.xlim(-0.35, 1.2)
    plt.ylim(-0.35, 1.2)
    plt.xlabel('Network prediction', size =15, color = 'k')
    plt.ylabel('Measured Values', size = 15, color = 'k')
    plt.tight_layout()
    #plt.show()
    plt.savefig('Network_prediction.png')