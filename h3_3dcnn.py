import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorflow.keras.layers import (
    Convolution3D, Dense, Dropout, Flatten, 
    Activation, MaxPooling3D, ZeroPadding3D, Conv3D
)
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import categorical_crossentropy


class ThreeDimCNN(object):
    """ 3D Convolutional Neural Network """
    def __init__(self, input_dim, input_frames, channel, batch_size, num_epochs, num_classes, data_dir):
        self.input_dim = input_dim
        self.input_frames = input_frames
        self.channel = channel
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.labels_map = self.label2numeric()
        # print(self.labels_map)

    
    def label_list(self):
        """
        Listing all 101 labels in provided dataset
        Extract from any source you can 
        
        Return:
        - label_list: list of labels which used for training 
        """
        label_list = []
        
        #### Question (a): your implementation starts here (don't delete this line)       
       	ucf_videos = sorted(os.listdir(self.data_dir))
        for video in ucf_videos:
            category = video[2:-12]
            if category not in label_list:
                label_list.append(category)

        #### Question (a): your implementation ends here (don't delete this line)
        return label_list


    def label2numeric(self):
    # def label2numeric(self, labels):
        label_list = self.label_list()
        labels_map = {x: int(i) for i, x in enumerate(label_list)}
        return labels_map
        # return list(map(lambda x: labels_map[x], labels))


    def numeric2labels(self, numeric):
        label_list = self.label_list()
        labels_map = {i: x for i, x in enumerate(label_list)}
        return list(map(lambda x: labels_map[x], labels))

    
    def get_loaders(self, X, Y):
        # print(X.shape, Y.shape)
        train_examples, test_examples, train_labels, test_labels = train_test_split(
            X, Y, test_size=0.2, random_state=99
        )
        # train_labels = tf.one_hot(train_labels, self.num_classes)
        # test_labels = tf.one_hot(test_labels, self.num_classes)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

        SHUFFLE_BUFFER_SIZE = 100
        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(self.batch_size)
        test_dataset = test_dataset.batch(self.batch_size)

        return train_dataset, test_dataset


    def load_data(self, color=False, skip=True):
        """
        Load video datas from dataset
        Input:
        - color: Default: False - Video without color
        - skip: Default: True
        Return:
        - X: clips which are splited by frames from original video 
            Shape = (N,W,H,F,C)
            W - Weight of input
            H - Height of input
            F - Frames
            C - Channel
        - Y: Shape = (N,C)
            N - Number of inputs
            C - Number of classes
        """
        X = []
        Y = []
        #### Question (b): your implementation starts here (don't delete this line)
        RESIZE_SHAPE = (171, 128) # (width, height) == (171, 128)
        CROP_SIZE = (112, 112) # (width, height) == (112, 112)

        def resize_and_crop(frame):
            frame = cv2.resize(frame, RESIZE_SHAPE)
            x, y = RESIZE_SHAPE
            x_c, y_c = CROP_SIZE
            width_idx = np.random.randint(x - x_c)
            height_idx = np.random.randint(y - y_c)
            frame = frame[height_idx:(height_idx + y_c), width_idx:(width_idx + x_c), :]

            return frame / 255.

        def load_video(path, max_frames=16):
            cap = cv2.VideoCapture(path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_freq = 5
            # make sure splitting video has at least 16 frames.
            while True:
                if frame_count // frame_freq <= max_frames:
                    frame_freq -= 1
                else:
                    break
                # print(f"Frame freq: {frame_count} <==> {frame_freq}")

            frames = []
            count = 0
            append_counter = 0

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if count % frame_freq == 0:
                        frame = resize_and_crop(frame)
                        frames.append(frame)
                        append_counter += 1
                    count += 1
                
                    if append_counter == max_frames:
                        break
            finally:
                cap.release()
            return np.transpose(np.array(frames), (1, 2, 0, 3))

        # ===============================================
        # label_list = sorted(os.listdir(self.data_dir))
        # for dir_ in label_list:
        #     videos_list = os.listdir(os.path.join(self.data_dir, dir_))
        #     for video_name in tqdm(videos_list):
        #         video_category = video_name[2:-12]
        #         print(video_category)
        #         video = load_video(os.path.join(self.data_dir, dir_, video_name))
        #         X.append(video)
        #         Y.append(self.labels_map[video_category])
        #     X = np.array(X)
        #     Y = np.array(Y)
        # ===============================================

        videos_list = os.listdir(os.path.join(self.data_dir))
        for video_name in tqdm(videos_list):
            video_category = video_name[2:-12]
            video = load_video(os.path.join(self.data_dir, video_name))
            X.append(video)
            Y.append(self.labels_map[video_category])
        X = np.array(X)
        Y = np.array(Y)

        # ===============================================

        #### Question (b): your implementation ends here (don't delete this line)
        return X, Y

    def model(self, input_shape):
        """
        Define Model Architecture:
        Input:
        - input_shape: (N, H, W, C)
            N - Number of inputs
            W - Weight of input
            H - Height of input
            C - Channel
        Return:
        - model: Defined Models
        """
        model = Sequential()
        
        #### Question (c): your implementation starts here (don't delete this line)
        model.add(Conv3D(64, 3, padding="same", input_shape=input_shape))
        model.add(MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1)))

        model.add(Conv3D(128, 3, padding="same"))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

        model.add(Conv3D(256, 3, padding="same"))
        model.add(Conv3D(256, 3, padding="same"))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

        model.add(Conv3D(512, 3, padding="same"))
        model.add(Conv3D(512, 3, padding="same"))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

        model.add(Conv3D(512, 3, padding="same"))
        model.add(Conv3D(512, 3, padding="same"))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
        model.add(ZeroPadding3D(padding=(1, 1, 0)))

        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Dense(4096))
        model.add(Dense(self.num_classes, activation="softmax"))   
       
        #### Question (c): your implementation ends here (don't delete this line)
        
        return model
    
def cate_crossentropy_loss(y_true, y_pred):
    #### Question (d): your implementation starts here (don't delete this line)
    # print(y_true.shape, y_pred.shape)
    y_true = tf.squeeze(tf.one_hot(tf.cast(y_true, tf.int32), 101), 1)
    return categorical_crossentropy(y_true, y_pred)
    
    #### Question (d): your implementation ends here (don't delete this line)
    # pass

def main():
    """
    Define three_dim with your parameter
    Define input data
        - X,Y
    Define model
        - model
    Make the training
        - history = model.fit()
    Make the evaluation
        - model.evalutate
    Make saving model
        - model.save_weights
    """
    data_dir = '/content/UCF101'
    input_dim = 32
    input_frames = 16
    output_model = './3dcnn_model/'
    batch_size = 30
    channel = 3
    num_epochs = 15
    num_classes = 101
    input_shape = (112, 112, 16, 3)

    # Example for three_dim:
    # three_dim = ThreeDimCNN(input_dim, 15, 3, 32, 20, 101,data_dir)
    
    #### Question (d): your implementation starts here (don't delete this line)
    three_dim = ThreeDimCNN(input_dim, input_frames, channel, batch_size, num_epochs, num_classes, data_dir)
    X, Y = three_dim.load_data()
    train_dataset, test_dataset = three_dim.get_loaders(X, Y)
    model = three_dim.model(input_shape)
    model.compile(Adam(), loss=cate_crossentropy_loss, metrics=['accuracy', 'sparse_categorical_accuracy'])
    history = model.fit(train_dataset, epochs=num_epochs)
    model.evaluate(test_dataset)
    model.save_weights('./3dcnn_model/trained_ucf101_model')
    
    #### Question (d): your implementation ends here (don't delete this line)
    
    """
    Plot Loss and Accuracy for Visualization (using history defined at the previous question)
    Using matplotlib
    """
    #### Question (e): your implementation starts here (don't delete this line)
    output_save_path = './result_visualization/'
    os.makedirs(output_save_path, exist_ok=True)
    
    acc = history.history['accuracy']
    loss = history.history['loss']

    plt.figure(figsize=(6, 8))
    plt.plot(acc, label='accuracy')
    plt.plot(valossl_acc, label='loss')
    plt.legend(loc='lower right')
    plt.title('Training Accuracy and Loss')
    plt.savefig(f"{output_save_path}images/loss_acc_ploy.png")
    
    #### Question (e): your implementation ends here (don't delete this line)
    
if __name__ == '__main__':
    main()

