from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.initializers import RandomUniform
from keras.layers import Dense, Activation
from keras.optimizers import SGD


TRN_SIZE = 60000
TST_SIZE = 10000
BATCH_SIZE = 100
CLASSES = 10
EPOCHS = 5
IMAGE_SIZE = 28



(trn_images, trn_labels), (tst_images, tst_labels) = mnist.load_data()

trn_images = trn_images.reshape(TRN_SIZE, IMAGE_SIZE**2)
tst_images = tst_images.reshape(TST_SIZE, IMAGE_SIZE**2)
tst_labels = to_categorical(tst_labels, CLASSES)
trn_labels = to_categorical(trn_labels, CLASSES)
trn_images = trn_images.astype("float32")
tst_images = tst_images.astype("float32")
trn_images /= 255
tst_images /= 255

ru = RandomUniform(minval=-0.1, maxval=0.1, seed=None)
model = Sequential()
model.add(Dense(300, input_shape=(IMAGE_SIZE**2,),
                kernel_initializer=ru, bias_initializer=ru))
model.add(Activation("relu"))
model.add(Dense(CLASSES))
model.add(Activation("softmax"))

model.summary()

model.compile(optimizer=SGD(lr=0.05, momentum=0, decay=0.0001, nesterov=False),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(trn_images, trn_labels,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(tst_images, tst_labels))
score = model.evaluate(tst_images, tst_labels)
print("final accuracy: %f" % score[1])