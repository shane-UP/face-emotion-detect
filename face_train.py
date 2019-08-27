import random
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, Conv2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
import numpy as np
from sklearn.model_selection import train_test_split
from load_dataset import load_dataset, resize_image, IMAGE_SIZE

import cv2

class Dataset:
    def __init__(self, path_name, nb_classes):
        self.train_images = None
        self.train_labels = None

        self.valid_images = None
        self.valid_labels = None

        self.test_images = None
        self.test_labels = None

        self.path_name = path_name

        self.input_shape = None

        self.nb_classes = nb_classes

    def load(self, img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE, img_channels = 3):
        images, labels = load_dataset(self.path_name)

        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3,
                                                                                 random_state=0)
        test_images, valid_images, test_labels, valid_labels = train_test_split(test_images, test_labels, test_size=0.5, random_state=0)



        train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
        valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
        test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
        self.input_shape = (img_rows, img_cols, img_channels)

        #训练集、验证集、测试集的数量
        print(train_images.shape[0], 'train samples')
        print(valid_images.shape[0], 'valid samples')
        print(test_images.shape[0], 'test samples')

        #模型使用categorical_crossentropy作为损失函数，因此需要类别数量nb_classes
        #将类别向量（从0到nb_classes的整数向量）映射为二值类别矩阵
        train_labels = np_utils.to_categorical(train_labels, self.nb_classes)
        valid_labels = np_utils.to_categorical(valid_labels, self.nb_classes)
        test_labels = np_utils.to_categorical(test_labels, self.nb_classes)

        #使像素浮点化方便归一化
        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')
        test_images = test_images.astype('float32')

        train_images /= 255
        valid_images /= 255
        test_images /= 255

        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_labels = test_labels

class Model:
    def __init__(self):
        self.model = None

    def build_model(self, dataset):
        self.model = Sequential()
        #
        # self.model.add(Conv2D(32, (1, 1), strides=1, padding='same', input_shape=dataset.input_shape))
        # self.model.add(Activation('relu'))
        # self.model.add(Conv2D(32, (5, 5), padding='same'))
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.25))
        #
        # self.model.add(Conv2D(32, (3, 3), padding='same'))
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #
        # self.model.add(Conv2D(64, (5, 5), padding='same'))
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.25))
        #
        # self.model.add(Flatten())
        # self.model.add(Dense(2048))
        # self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.5))
        # self.model.add(Dense(1024))
        # self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.5))
        # self.model.add(Dense(dataset.nb_classes))
        # self.model.add(Activation('softmax'))
        # self.model.summary()

        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # self.model.add(Conv2D(64, (3, 3), activation='relu'))
        # self.model.add(Conv2D(64, (3, 3), activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(dataset.nb_classes, activation='softmax'))
        self.model.summary()

    def train(self, dataset, batch_size=32, nb_epoch=10, data_augmentation=True):
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  #支持动量参数，支持学习衰减率，支持Nesterov动量
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) #编译模型以供训练 compile()

        if not data_augmentation:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(dataset.valid_images, dataset.valid_labels),
                           shuffle=True)  #模型训练函数fit()
        else:             #数据提升貌似没啥用
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=False
            )
            datagen.fit(dataset.train_images)

            hist_fit = self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels, batch_size=batch_size),
                                                steps_per_epoch=dataset.train_images.shape[0]/batch_size,
                                                epochs=nb_epoch,
                                                verbose=1,
                                                validation_data=(dataset.valid_images, dataset.valid_labels))

            hist_val = self.model.evaluate_generator(datagen.flow(dataset.valid_images, dataset.valid_labels, batch_size=batch_size),
                                                     verbose=1,
                                                     steps=dataset.test_images.shape[0]/batch_size)
            with(open('./face_model_fit_log.txt', 'w+')) as f:
                f.write(str(hist_fit.history))
            with(open('./face_model_val_log.txt', 'w+')) as f:
                f.write(str(hist_val))


    def save_model(self):
        self.model_json = self.model.to_json()
        with open('./model/face_model_json.json', 'w+') as json_file:
            json_file.write(self.model_json)
        self.model.save_weights('./model/face_model_weight.h5')
        self.model.save('./model/face_model.h5')
        print('save finished')

    def load_model(self):
        json_file = open('./model/face_model_json.json')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights('./model/face_model_weight.h5')


    def face_predict(self, image):
        face_labels = ['zhuang', 'deng', 'others', 'shang']

        if image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

        result = self.model.predict(image)
        print('result:', result[0])

        result = self.model.predict_classes(image)
        faceID = face_labels[result[0]]

        return faceID

if __name__ == '__main__':
    dataset = Dataset('./data/', 4)
    dataset.load()

    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.save_model()


