# import necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from aspectawarepreprocessor import AspectAwarePreprocessor
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simpledatasetloader import SimpleDatasetLoader
from keras.preprocessing.image import ImageDataGenerator
from minivggnet import MiniVGGNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to dataset')
args = vars(ap.parse_args())


# Grab the image and its label
imagePaths = list(paths.list_images(args['dataset']))
classLabels = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classLabels = [str(x) for x in np.unique(classLabels)]

# initialize the preprocessor
iap  = ImageToArrayPreprocessor()
aap = AspectAwarePreprocessor(64, 64)

sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype('float') / 255.0

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# construct augmentation generator
aug = ImageDataGenerator(rotation_range= 30, width_shift_range=0.5, height_shift_range=0.5,
	shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')




print('[INFO] Compiling the model.........')
opt = SGD(lr=0.05)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes = len(classLabels))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


print('[INFO] training the model.........')
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data = (testX, testY),
	steps_per_epoch = len(trainX) // 32, epochs=100, verbose=1)

# evalute the network
print('[INFO] evaluting the network..........')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), 
	target_names=classLabels))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


