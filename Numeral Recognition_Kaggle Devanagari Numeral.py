
import numpy as np 
import glob
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.utils import shuffle
import time

from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.models import Model, Sequential
from keras.optimizers import SGD,RMSprop,adam, Adadelta
from keras.preprocessing.image import ImageDataGenerator


# we have a model trained on DHCD for character classification (46 classes). This is a large dataset.
# Kaggle Devanagari Numeral dataset is a small dataset set. Lets choose the pretrained model on DHCD and Fine tune it with 
#10 classes of Kaggle Devanagari Numeral dataset.

##Read the Images and preprocessing
train_images = []
train_labels = [] 

for directory_path in glob.glob("D:/ashokPant/nhcd/nhcd/numerals/*"):
    print(directory_path)
    label = directory_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        #img_path="D:/ashokPant/nhcd/nhcd/numerals/5/001_01.jpg"
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)   
        img = 255 - img
        # Using cv2.copyMakeBorder() method
        img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT)
        #img = cv2.resize(img, (32,32))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)
train_images = np.array(train_images)
train_labels = np.array(train_labels)

label_to_id = {v:i for i,v in enumerate(np.unique(train_labels))}
id_to_label = {v: k for k, v in label_to_id.items()}
train_label_ids = np.array([label_to_id[x] for x in train_labels])

print(train_images.shape),print( train_label_ids.shape), print(train_labels.shape)

# Define the number of classes
num_classes = 10
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(train_label_ids  , num_classes)

#Shuffle the dataset
x,y = shuffle(train_images,Y, random_state=2)
#x,y = shuffle(train_images,Y, random_state=2)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
#x_train, x_test, y_train, y_test = train_test_split(train_images,Y, test_size=0.3, random_state=2)
print(y_test.shape),print(y_test.shape)#, print(train_labels.shape)


####################################################################################

#pretrained moel is DHCD
image_input = Input(shape=(32,32, 3))

#load pretrained model
model=load_model('my_model.hdf5')
model.summary()
#load weights of pretrained model
model.load_weights('weights-improvement-43-0.99.hdf5')

#model.summary()

last_layer = model.get_layer('dense_8').output


#x= Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output')(last_layer)

custom_DHCD_model = Model(model.input, out)
custom_DHCD_model.summary()

for layer in custom_DHCD_model.layers[:-1]:
	layer.trainable = False

custom_DHCD_model.layers[3].trainable


#########################################################################################

# make folder for results
from os import makedirs    
makedirs('results_pretrained', exist_ok=True)            
  
custom_DHCD_model.save('results_pretrained/custom_DHCD_model_kaggle_numeral.hdf5')

# compile model
#opt = SGD(lr=0.001, momentum=0.9)
#opt = RMSprop(lr=0.001)
opt = adam(lr=0.001)



custom_DHCD_model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


# Training with callbacks
from keras import callbacks

filename='results_pretrained/train_customModel_Adam_50epochs.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
#early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0,patience=10, verbose=1, mode='min')

filepath="results_pretrained/Best-weights-customModel_Adam_-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [ csv_log,checkpoint]


t=time.time()
#	t = now()
history = custom_DHCD_model.fit(x_train, y_train, batch_size=32, epochs=50, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks_list)
print('Training time: %s' % (t - time.time()))

(loss, accuracy) = custom_DHCD_model.evaluate(x_test, y_test, batch_size=32, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

#####################################

#################################################################################
#load model 
from keras.models import load_model
model=load_model('custom_DHCD_model_kaggle_numeral.hdf5')
model.summary()

model.load_weights('Best-weights-customModel_Adam_-049-0.1046-0.9648.hdf5')

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
# evaluate model
#_, acc = model.evaluate(x_val, y_val, verbose=0)
#print('> %.3f' % (acc * 100.0))
# evaluate model on test dataset
_, acc = model.evaluate(x_test, y_test, verbose=0)
print('> %.3f' % (acc * 100.0))	


###########################################################
# visualizing losses and accuracy
import matplotlib.pyplot as plt
train_loss=history.history['loss']
val_loss=history.history['val_loss']
train_acc=history.history['acc']
val_acc=history.history['val_acc']
#xc=range(num_epoch)
xc=range(50)

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
from matplotlib import pyplot
	# plot loss
xc=range(50)
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
pyplot.savefig('kaggleNumeral_Adam.png')
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

target_names = ['digit_0', 'digit_1', 'digit_2', 'digit_3', 'digit_4', 'digit_5', 'digit_6', 'digit_7', 'digit_8', 'digit_9']

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
pyplot.savefig('kaggle_numeral_Adam.png')
pyplot.close()
#plt.figure()
# Plot normalized confusion matrix
#plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
#                      title='Normalized confusion matrix')
#plt.figure()
plt.show()


#######################################################
