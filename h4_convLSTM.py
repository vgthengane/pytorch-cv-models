import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, ConvLSTM2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import categorical_crossentropy
from tqdm import tqdm
# from tensorflow.python.ops.gen_nn_ops import data_format_dim_map_eager_fallback
class CLSTM(object):
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


    def label_list(self):
        label_list = []      
       	ucf_videos = sorted(os.listdir(self.data_dir))
        for video in ucf_videos:
            category = video[2:-12]
            if category not in label_list:
                label_list.append(category)
        return label_list


    def label2numeric(self):
        label_list = self.label_list()
        labels_map = {x: int(i) for i, x in enumerate(label_list)}
        return labels_map

    
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


    def load_data(self, data_dir):
        """
        Load video datas from dataset
        Input:
        - data_dir: data directory
        Return:
        - X: Frames captured from input video 
            Shape = (N,W,H,F,C)
            N - Number of inputs
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
        #### Question (a): your implementation starts here (don't delete this line)
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
            # return np.transpose(np.array(frames, dtype=np.float32), (0, 1, 2, 3))
            return np.array(frames, dtype=np.float32)

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
        # X = np.array(X)
        # Y = np.array(Y)
        # ===============================================

        videos_list = os.listdir(os.path.join(data_dir))
        for video_name in tqdm(videos_list[:500]):
            video_category = video_name[2:-12]
            video = load_video(os.path.join(data_dir, video_name))
            X.append(video)
            Y.append(self.labels_map[video_category])
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        
        #### Question (a): your implementation ends here (don't delete this line)
        return X, Y

    def model(self, input_shape):
        """
        Define Model Architecture:
        (The detailed model has been mentioned on pdf file)
        Input:
        - input_shape: (N, H, W, C)
            N - Number of inputs
            W - Weight of input
            H - Height of input
            C - Channel
        Return:
        - model: Defined Model
        """
        model = Sequential()
        
        #### Question (b): your implementation starts here (don't delete this line)
        model.add(ConvLSTM2D(filters=64, kernel_size=5, strides=(4, 4), padding="same", input_shape=input_shape))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(units=256))
        model.add(Dropout(0.5))
        model.add(Dense(units=101))
        
        #### Question (b): your implementation ends here (don't delete this line)
        
        return model
    
    def ano_model(self, input_shape):
        """
        Define different model from the main one (another model):
        (You can hyperpameter, add hidden layer, change filter, change kernel, ... anything)
        Input the same with the main model
        Returm:
        - model: New model
        """
        model = Sequential()
        #### Question (d): your implementation starts here (don't delete this line)
        model.add(ConvLSTM2D(filters=64, kernel_size=5, strides=2, padding="same", 
                                            input_shape=input_shape, return_sequences=True))
        model.add(ConvLSTM2D(filters=64, kernel_size=5, strides=2, padding="same"))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(units=256))
        model.add(Dropout(0.5))
        model.add(Dense(units=101))

        #### Question (d): your implementation ends here (don't delete this line)
        return model


def cate_crossentropy_loss(y_true, y_pred):
    #### Question (d): your implementation starts here (don't delete this line)
    # print(y_true.shape, y_pred.shape)
    y_true = tf.squeeze(tf.one_hot(tf.cast(y_true, tf.int32), 101), 1)
    return categorical_crossentropy(y_true, y_pred)


def main():
    """
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
    data_dir = './UCF-101'
    input_dim = 32
    input_frames = 16
    output_model = './convo_lstm_model'
    batch_size = 30
    channel = 3
    num_epochs = 15
    num_classes = 101
    input_shape = (16, 112, 112, 3)

    os.makedirs(output_model, exist_ok=True)

    # convo_lstm = CLSTM(input_dim, 15, 3, 32, 20, 101, data_dir)
    
    #### Question (c): your implementation starts here (don't delete this line)
    convo_lstm = CLSTM(input_dim, input_frames, channel, batch_size, num_epochs, num_classes, data_dir)
    X, Y = convo_lstm.load_data(data_dir)
    train_dataset, test_dataset = convo_lstm.get_loaders(X, Y)
    for x, y in train_dataset.take(1):
        print(x.shape, y.shape)
    
    """
    Comment c when you run d and Comment d when you run c
    """
    model = convo_lstm.model(input_shape) # line - c
    print(model.summary())
    # model = convo_lstm.ano_model(input_shape) # line - d
    
    model.compile(Adam(), loss=cate_crossentropy_loss, metrics=['accuracy', 'sparse_categorical_accuracy'])
    history = model.fit(train_dataset, epochs=num_epochs)
    model.evaluate(test_dataset)
    model.save_weights(f'{output_model}/trained_ucf101_model')

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
        
    #### Question (c): your implementation ends here (don't delete this line)
    
    #### Question (d): your implementation starts here (don't delete this line)
    # it is added above at line 250

    
    #### Question (d): your implementation ends here (don't delete this line)

    
if __name__ == '__main__':
    main()

