from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
import cv2



class Model:
    def __init__(self):
        self.model = None
        self.batch_size = 128
        self.nb_classes = 7
        self.nb_epoch = 90
        self.img_size = 48

    def build_model(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, (1, 1), strides=1, padding='same', input_shape=(self.img_size, self.img_size, 1)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(32, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(2048))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.nb_classes))
        self.model.add(Activation('softmax'))
        self.model.summary()

    def train_model(self):
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        val_datagen = ImageDataGenerator(
            rescale=1./255
        )
        test_datagen = ImageDataGenerator(
            rescale=1./255
        )

        train_generator = train_datagen.flow_from_directory(
            './train',
            target_size=(self.img_size, self.img_size),
            color_mode='grayscale',
            batch_size=self.batch_size)

        val_generator = val_datagen.flow_from_directory(
            './val',
            target_size=(self.img_size, self.img_size),
            color_mode='grayscale',
            batch_size=self.batch_size)

        test_generator = test_datagen.flow_from_directory(
            './test',
            target_size=(self.img_size, self.img_size),
            color_mode='grayscale',
            batch_size=self.batch_size)

        EarlyStopping(monitor='loss', patience=3)

        hist_fit = self.model.fit_generator(train_generator,
                                 steps_per_epoch=28709/128,
                                 epochs=self.nb_epoch,
                                 verbose=1,
                                 validation_data=val_generator,
                                 validation_steps=28709/128,
                                 )
        hist_predict = self.model.predict_generator(test_generator,
                                     steps=3589/128,
                                     verbose=1)

        with(open('./emotion_model_fit_log.txt', 'w+')) as f:
            f.write(str(hist_fit.history))
        with(open('./emotion_model_predict_log.txt', 'w+')) as f:
            f.write(str(hist_predict))

        print('train finished')

    def save_model(self):
        model_json = self.model.to_json()
        with open('./model/emotion_model_json.json', 'w+') as json_file:
            json_file.write(model_json)
        self.model.save_weights('./model/emotion_model_weight.h5')
        self.model.save('./model/emotion_model.h5')
        print('saved finished')

    def load_model(self):
        json_file = open('./model/emotion_model_json.json')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights('./model/emotion_model_weight.h5')

    def emotion_predict(self, image):
        emo_labels = ['angry', 'disgust:', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        if image.shape != (1, self.img_size, self.img_size, 3):
            image = cv2.resize(image, (self.img_size, self.img_size))
            image = image.reshape((1, self.img_size, self.img_size, 1))

        result = self.model.predict(image)
        print('result: ', result)

        result = self.model.predict_classes(image)
        emo = emo_labels[result[0]]
        print('result: ', emo)

        return emo

if __name__ == '__main__':
    model = Model()
    model.build_model()
    print('model built')
    model.train_model()
    print('model trained')
    model.save_model()
    print('model saved')
