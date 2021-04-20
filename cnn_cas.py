"""
Training a Convolutional Neural Network to Predict non-parametric measurements of galaxy structure. 

Inputs: 
    img_filename: Path to file containing you data in an array (shape(no. of images, (image_size)))
    val_filename: Path to file containing the corresponding values you want to train the network to predict


Returns: 
    trained_A_weights: The best model from training 
    Training_curve.png: Plot of training curves 
    Network_prediction.png: Plot of networks predictions vs the inputted values 

"""

from cnn_functions import *

#Load image and value arrays for training 
x_train, x_test, x_val, y_train, y_test, y_val = get_data(img_filename = 'testing_imgs.npy', val_filename = 'parameters_to_predict')

num_classes = 1 #the number of parameters you are training the network to predict

#Create the network (Asymmetry Network architecture)
### CNN ### 
cnnmodel = Sequential()
cnnmodel.add(Conv2D(32, (3, 3), activation='relu', input_shape=(60, 60, 1)))
cnnmodel.add(Conv2D(32, (3, 3), activation='relu'))
cnnmodel.add(AveragePooling2D((2, 2)))
cnnmodel.add(BatchNormalization())
cnnmodel.add(Conv2D(64, (3, 3), activation='relu'))
cnnmodel.add(Conv2D(64, (3, 3), activation='relu'))
cnnmodel.add(AveragePooling2D((2, 2)))
cnnmodel.add(BatchNormalization())
cnnmodel.add(Conv2D(128, (3, 3), activation='relu'))
cnnmodel.add(Conv2D(128, (3, 3), activation='relu'))
cnnmodel.add(AveragePooling2D((2, 2)))
cnnmodel.add(BatchNormalization())
cnnmodel.add(Flatten())
cnnmodel.add(Dense(1024, activation = 'relu'))
cnnmodel.add(Dense(1024, activation = 'relu'))
cnnmodel.add(Dropout(0.5))
cnnmodel.add(Dense(num_classes, activation='linear'))

#define batch size and number of epochs - compile the model
batch_size = 512
epochs = 300
cnnmodel.compile(loss= root_mean_squared_error ,optimizer= Adamax(0.001), metrics=['mean_absolute_error', pearson_r])
cnnmodel.summary()

#Train the model 
es = EarlyStopping(monitor='val_mean_absolute_error', mode='min', verbose=1, patience=30) #Implement early stopping
filepath = 'trained_weights.hdf5' #Name to save best model under 
checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', mode='min', save_best_only = True, verbose = 1)
history = cnnmodel.fit(x_train, y_train,batch_size=batch_size, epochs=epochs,verbose=1,validation_data=(x_test, y_test),callbacks=[es, checkpoint])
histplot(history) #Save plot of training curves (Training_curve.png)

#Evaluate the best model 
best_model = load_model('trained_weights.hdf5', 
                        custom_objects={'root_mean_squared_error': root_mean_squared_error,'pearson_r':pearson_r})
Tstart = time.time() #Records time taken for trained network to predict values 
score = best_model.evaluate(x_val, y_val, verbose=2)
Tprocess0 = time.time()
print('Test RMSE:', score[0])
print('Test Mean absolute error:', score[1])
print('\n', '## NETWORK RUNTIME:', np.round(Tprocess0-Tstart,decimals=3), "seconds") 

#prediction of parameter 
pred = best_model.predict(x_val)
true = y_val
#Plot trained networks prediction (Network_prediction.png)
predict_plot(pred, true)
