
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from keras import backend as K
#K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam, adadelta
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import sys
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

#Load training data
# Define data path
data_path = 'D:/CVSI2015/TrainDataset_CVSI2015/TrainDataset_CVSI2015'
data_dir_list = os.listdir(data_path)


# Define the number of classes
num_classes = 10

labels_name={'Arabic':0,'Bengali':1,'English':2,'Gujrathi':3, 'Hindi':4,'Kannada':5,'Oriya':6,'Punjabi':7,
             'Tamil':8,'Telegu':9}

img_data_list=[]
labels_list = []

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loading the images of dataset-'+'{}\n'.format(dataset))
	label = labels_name[dataset]
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		#input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		input_img_resize=cv2.resize(input_img,(224,224))
		img_data_list.append(input_img_resize)
		labels_list.append(label)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)

#img_data= np.expand_dims(img_data, axis=3) 
#print (img_data.shape)

labels = np.array(labels_list)
# print the count of number of samples for different classes
print(np.unique(labels,return_counts=True))
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
#x,y = shuffle(img_data,Y, random_state=2)
x_train,y_train = shuffle(img_data,Y, random_state=2)

##############################
img_data=[]
Y=[]
img_data_list=[]
labels_list = []
###############################
#Load validation data
data_path= 'D:/CVSI2015/ValidationDataset_CVSI2015/ValidationDataset_CVSI2015'
data_dir_list = os.listdir(data_path)

img_data_list=[]
labels_list = []

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loading the images of dataset-'+'{}\n'.format(dataset))
	label = labels_name[dataset]
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		#input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		input_img_resize=cv2.resize(input_img,(224,224))
		img_data_list.append(input_img_resize)
		labels_list.append(label)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)

#img_data= np.expand_dims(img_data, axis=3) 
#print (img_data.shape)

labels = np.array(labels_list)
# print the count of number of samples for different classes
print(np.unique(labels,return_counts=True))
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
#x,y = shuffle(img_data,Y, random_state=2)
x_val,y_val = shuffle(img_data,Y, random_state=2)


############################################
img_data_list=[]
labels_list = []
img_data=[]
############################################
##Load test data
data_path= 'D:/Test_cvsi'
data_dir_list = os.listdir(data_path)

img_data_list=[]
labels_list = []

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loading the images of dataset-'+'{}\n'.format(dataset))
	label = labels_name[dataset]
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		#input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		input_img_resize=cv2.resize(input_img,(224,224))
		img_data_list.append(input_img_resize)
		labels_list.append(label)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)

#img_data= np.expand_dims(img_data, axis=3) 
#print (img_data.shape)

labels = np.array(labels_list)
# print the count of number of samples for different classes
print(np.unique(labels,return_counts=True))
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
#x,y = shuffle(img_data,Y, random_state=2)
x_test,y_test = shuffle(img_data,Y, random_state=2)

# Split the dataset
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
img_data_list=[]
labels_list = []
img_data=[]
####################################################

# vgg16 model used for transfer learning 
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# define cnn model
#Custom_vgg_model
# load model
model = VGG16(include_top=False, input_shape=(224, 224, 3))
# mark loaded layers as not trainable
for layer in model.layers:
	layer.trainable = False
# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
class2 = Dense(128, activation='relu',  kernel_initializer='he_uniform')(class1)
output = Dense(num_classes, activation='sigmoid')(class2)
# define new model
model = Model(inputs=model.inputs, outputs=output)
# compile model
opt = SGD(lr=0.001, momentum=0.9) 
#opt = adam(lr=0.001)
#opt = RMSprop(lr=0.001)

#########################################################

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


model.summary()
model.save('best_model_29jan21.hdf5')

# create data generator
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
# prepare iterator
it_train = datagen.flow(x_train, y_train, batch_size=64)

steps = int(x_train.shape[0] / 64)
# Training with callbacks
from keras import callbacks
filename='29jan21_train_new_25epochs.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
#early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0,patience=10, verbose=1, mode='min')
filepath="Best-weights-29jan21_-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [ csv_log,checkpoint]

t=time.time()
history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=25, validation_data=(x_val, y_val), verbose=1,callbacks=callbacks_list)
print('Training time: %s' % (t - time.time()))
######################################################3			
#load model 
from keras.models import load_model
model=load_model('best_model_29jan21.hdf5')
model.summary()

model.load_weights('Best-weights-29jan21_-014-0.1113-0.9630.hdf5')

# evaluate model
_, acc = model.evaluate(x_val, y_val, verbose=0)
print('> %.3f' % (acc * 100.0))
# evaluate model on test dataset
_, acc = model.evaluate(x_test, y_test, verbose=0)
print('> %.3f' % (acc * 100.0))	
#################################################################################################
##############################################################3
#load model
from keras.models import load_model
model=load_model('best_model_29jan21.hdf5')
model.summary()

model.load_weights('Best-weights-29jan21_-014-0.1113-0.9630.hdf5')
###########################################################
# visualizing losses and accuracy
#import matplotlib.pyplot as plt
train_loss=history.history['loss']
val_loss=history.history['val_loss']
train_acc=history.history['acc']
val_acc=history.history['val_acc']
#xc=range(num_epoch)
xc=range(25)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
####################################################
# create a line plot of loss and save to file

	# plot loss
xc=range(25)
pyplot.subplot(2, 1, 1)
pyplot.plot(xc,train_loss)
pyplot.plot(xc,val_loss)
pyplot.xlabel('num of Epochs')
pyplot.ylabel('loss')
pyplot.grid(True)
pyplot.legend(['train','val'])
            
pyplot.subplot(2, 1, 2)
pyplot.plot(xc,train_acc)
pyplot.plot(xc,val_acc)
pyplot.xlabel('num of Epochs')
pyplot.ylabel('accuracy')
pyplot.grid(True)
pyplot.legend(['train','val'],loc=4)
pyplot.style.use(['classic'])

# save plot to file
pyplot.savefig('CVSI_adam.png')
pyplot.close()
############################################################

#confusion matrix
from sklearn.metrics import confusion_matrix
preds = model.predict(x_test)
predicts = np.argmax(preds, axis = 1)
Y_test_labels = np.argmax(y_test, axis =1)
cm = confusion_matrix(Y_test_labels, predicts)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
########################################3
#or other method
# Printing the confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = model.predict(x_test)
#print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
#print(y_pred)
#y_pred = model.predict_classes(X_test)
#print(y_pred)
target_names = ['Arabic', 'Bengali', 'English', 'Gujrathi', 'Hindi', 'Kannada', 'Oriya', 'Punjabi', 'Tamil', 'Telegu']
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

############################################################
# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
# save plot to file
pyplot.savefig('CM_cvsi_adam.png')
pyplot.close()
#plt.figure()
# Plot normalized confusion matrix
#plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
#                      title='Normalized confusion matrix')
#plt.figure()
plt.show()



###########################################
