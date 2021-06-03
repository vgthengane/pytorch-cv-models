import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.backend as K
# from keras.models import load_model
from tensorflow.keras.models import load_model
from os import listdir
from os.path import isdir
from PIL import Image
import numpy as np
from numpy import load
from numpy import expand_dims
from numpy import asarray
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from tqdm import tqdm

def extract_face(filename, required_size=(160, 160)):
    """
    Extract a single face from a given photograph
    
    Inputs:
    - filename: path of a file to be converted

    Returns:
    - face_array: array of face image pixel with RGB channel
    """
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    face_array = pixels
    return face_array

def load_faces(directory):
    """
    Load images and extract faces for all images in a directory
    
    Inputs:
    - directory: path of a directory which has same person's face

    Returns:
    - face: list of face array
    """
    faces = list()
    for filename in listdir(directory):
        path = directory + filename
        face = extract_face(path)
        faces.append(face)
    return faces

def load_dataset(directory):
    """
    Load a dataset that contains one subdir for each class that in turn contains images
    
    Inputs:
    - directory: path of a directory which has all the train data or test data

    Returns:
    - asarray(X): face image array
    - asarray(y): class label array
    """
    X, y = list(), list()
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        if not isdir(path):
            continue
        if subdir.startswith('.'):
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)


def get_embedding(model, face_pixels):
    """
    Get the face embedding for one face
    
    Inputs:
    - model: facenet model which output 128-dim embedding
    - face_pixels: image array of a face

    Returns:
    - yhat[0]: embedding of a face
    """
    
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

def contrastive_loss(y, emb1, emb2, margin=1.0):
    """
    Compute the contrastive loss for two embeddings
    
    Inputs:
    - y: value of 1 if emb1 and emb2 are same person's face, 0 if not
    - emb1: embedding of a face
    - emb2: embedding of a face

    Returns:
    - loss
    """

    #### Question (b): your implementation starts here (don't delete this line)
    print(emb1.shape, emb2.shape)
    y_pred = tf.linalg.norm(emb1 - emb2)
    y = tf.cast(y, y_pred.dtype)
    loss = y * tf.math.square(y_pred) + (1.0 - y) * tf.math.square(
        tf.math.maximum(margin - y_pred, 0.0)
    )

    #### Question (b): your implementation ends here (don't delete this line)

    return loss

def triplet_loss(anchor, emb1, emb2, margin=1.0):
    """
    Compute the contrastive loss for two embeddings
    
    Inputs:
    - anchor: embedding of a face which to be the standard
    - emb1: embedding of a positive face
    - emb2: embedding of a negative face

    Returns:
    - loss
    """

    #### Question (c): your implementation starts here (don't delete this line)
    d_pos = tf.reduce_sum(tf.square(anchor - emb1))
    d_neg = tf.reduce_sum(tf.square(anchor - emb2))

    loss = tf.maximum(0., margin + d_pos - d_neg)
    loss = tf.reduce_mean(loss)

    #### Question (c): your implementation ends here (don't delete this line)

    return loss

def main():

    # load train dataset
    trainX, trainy = load_dataset('./LFW/train/')
    print(trainX.shape, trainy.shape)
    # load test dataset
    testX, testy = load_dataset('./LFW/val/')
    print(testX.shape, testy.shape)


    # load the pre-trained facenet model
    model = load_model('facenet_keras.h5', compile=False)

    # convert each face in the train set to an embedding
    print('[INFO] calculating train data embedding ...')
    newTrainX = list()
    for face_pixels in tqdm(trainX):
        embedding = get_embedding(model, face_pixels)
        newTrainX.append(embedding)
    trainX = asarray(newTrainX)
    # convert each face in the test set to an embedding
    print('[INFO] calculating test data embedding ...')
    newTestX = list()
    for face_pixels in tqdm(testX):
        embedding = get_embedding(model, face_pixels)
        newTestX.append(embedding)
    testX = asarray(newTestX)

    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)

    '''
    Generate linear classifier model which name is 'model'
    '''
    #### Question (a): your implementation starts here (don't delete this line)
    
    model = SVC(gamma='auto', verbose=True)

    #### Question (a): your implementation ends here (don't delete this line)

    # train
    print('[INFO] model is training ...')
    model.fit(trainX, trainy)
    print('[INFO] training is done.')
    # predict
    yhat_train = model.predict(trainX)
    yhat_test = model.predict(testX)
    # score
    score_train = accuracy_score(trainy, yhat_train)
    score_test = accuracy_score(testy, yhat_test)
    # summarize
    print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

    #loss function test with sample data
    print('Contrastive loss for same face: f' % (contrastive_loss(1,trainX[0], trainX[1])))
    print('Contrastive loss for different face: f' % (contrastive_loss(0,trainX[0], trainX[100])))
    print('Triplet loss: f' % (triplet_loss(trainX[0], trainX[0], trainX[100])))

if __name__ == '__main__':
  main()

