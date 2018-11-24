import cv2
import numpy as np
import os

class SimpleDatasetLoader():
    def __init__(self, preprocessors=None):
        #Store the image processor
        self.preprocessors = preprocessors
        # if image preprocessor is none
        #then initialize then as empty list
        if self.preprocessors is None:
            self.preprocessors = []
    
    def load(self, imagePaths, verbose=-1):
        # initialize the features and labels
        data = []
        labels = []
        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label
            # assumed the our path is following format
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            # check the preprocessor empty or not
            if self.preprocessors is not None:
                # loop over the preprocessors and apply
                # each on the image
                for p in self.preprocessors:
                    image = p.preprocess(image)
            # treat our processed image as 'feature vector'
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)

            # show update of 'verbose' image
            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print('[INFO] processed: {}/{}'.format(i+1, len(imagePaths)))
        # return the tuple of data and labels
        return (np.array(data), np.array(labels))





