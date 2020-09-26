import tensorflow as tf
from tensorflow.keras import layers
import keras
from keras import Model
import PIL
from PIL import Image, ImageFilter
import numpy as np
import os
import pathlib
import glob

print("Tensorflow version: ", tf.__version__)
print("GPU Availability: ", tf.config.list_physical_devices('GPU'))

gpus = tf.config.experimental.list_physical_devices('GPU')
# You can limit GPU memory usage here
limit_mem = True
if gpus and limit_mem:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7500)])
    except RuntimeError as e:
        print(e)

# I used this callback to see how my model's doing as it trains
# It's not super robust or useful though, there are definitely better
# ways to monitor this type of model's progress.
class AEMonitor(keras.callbacks.Callback):
    def __init__(self, orig_prediction_frames, outdir):
        super(AEMonitor)
        self.orig_prediction_frames = orig_prediction_frames
        self.frames_to_predict = orig_prediction_frames
        self.outdir = outdir

        for i in range(5):
            frame = orig_prediction_frames[0,:,:,i*3:(i+1)*3]
            Image.fromarray((frame*255).astype('uint8')).save(os.path.join(self.outdir, 'frame%04d.jpg'%(4-i)))

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.frames_to_predict)[0]
        Image.fromarray((prediction*255).astype('uint8')).save(os.path.join(self.outdir, 'frame{epoch:04d}.jpg'.format(epoch=epoch)))

# Define the model
# Model is basically shaped like an autoencoder, with a squeeze down to an
# encoded latent space and then an expansion from that space
''' NOTE:
Almost all of the model's architecture was chosen arbitrarily.
Lots of room to explore how many filters to use, how deep to go,
kernel size, activation functions, etc. etc. etc.
'''
# Input Shape = (WIDTH, HEIGHT, CHANNELS * Number of frames to predict from)
inputs = layers.Input((512,512,15))
x = layers.Conv2D(64, (7,7), padding='same', activation='relu')(inputs)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.Conv2D(32, (5,5), padding='same', activation='relu')(x)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.Conv2D(16, (3,3), padding='same', activation='relu')(x)
skip_one = x
x = layers.MaxPooling2D(pool_size=(2,2))(x)


x = layers.BatchNormalization(axis=-1)(x)
x = layers.Conv2D(32, (7,7), padding='same', activation='relu')(x)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.Conv2D(32, (5,5), padding='same', activation='relu')(x)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.Conv2D(16, (3,3), padding='same', activation='relu')(x)
skip_two = x
x = layers.MaxPooling2D(pool_size=(2,2))(x)



x = layers.BatchNormalization(axis=-1)(x)
x = layers.Conv2D(32, (5,5), padding='same', activation='relu')(x)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
x = layers.BatchNormalization(axis=-1)(x)
encoded = layers.Conv2D(16, (1,1), padding='same', activation='relu')(x)


x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(encoded)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.Conv2DTranspose(32, (5,5), padding='same', strides=2, activation='relu')(x)
x = layers.BatchNormalization(axis=-1)(x)

x = layers.Concatenate(axis=-1)([x, skip_two])
x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.Conv2D(32, (5,5), padding='same', activation='relu')(x)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.Conv2DTranspose(32, (7,7), padding='same', strides=2, activation='relu')(x)
x = layers.BatchNormalization(axis=-1)(x)

x = layers.Concatenate(axis=-1)([x, skip_one])
x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.Conv2D(32, (5,5), padding='same', activation='relu')(x)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.Conv2D(32, (5,5), padding='same', activation='relu')(x)
x = layers.BatchNormalization(axis=-1)(x)
decoded = layers.Conv2D(3, (7,7), padding='same', activation='sigmoid', dtype='float32')(x)

ae = keras.Model(inputs, decoded)

# These functions tell our model how to read our tfrecord file.
def _parse_function(proto):
    # both input & output are saved as binary strings
    keys_to_features = {
        'image':tf.io.FixedLenFeature([], tf.string),
        'label':tf.io.FixedLenFeature([], tf.string)
    }

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    # Decode the binary strings
    # ***IMPORTANT***
    # Make sure the datatype here matches whatever you used to write the data
    # to the tfrecord
    parsed_features['image'] = tf.io.decode_raw(parsed_features['image'], tf.float64)
    parsed_features['label'] = tf.io.decode_raw(parsed_features['label'], tf.float64)

    # Reshape the tensors
    parsed_features['image'] = tf.reshape(parsed_features['image'], [512,512,15])
    parsed_features['label'] = tf.reshape(parsed_features['label'], [512,512,3])

    return parsed_features['image'], parsed_features['label']

# This function gets the dataset
def get_dataset(path):
    dataset = tf.data.TFRecordDataset(path)

    dataset = dataset.map(_parse_function, 1)
    # Fills a shuffle buffer
    dataset = dataset.shuffle(150)
    # Makes the dataset repeat indefinitely
    dataset = dataset.repeat()
    # Prefectes data
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # Batches data
    dataset = dataset.batch(2)

    return dataset

# Get the dataset
ds = get_dataset('/home/tom/data/road_1frame_predictor/stacked_roadframes.tfrecord')



# Load up data to monitor model training with
sampled_frames = glob.glob('/home/tom/data/frames/road_frames_sampled_holder/road_frames_sampled/*')
sampled_frames.sort()
prediction_array = np.zeros([512,512,15])
for i in range(5):
    img = Image.open(sampled_frames[100+i]).resize([512,512])
    prediction_array[:,:,i*3:(i+1)*(3)] = np.array(img)/255

# Make callback
monitor = AEMonitor(np.expand_dims(prediction_array,0), 'logs/monitor')

# Make another callback to save weights periodically
# NOTE: save_freq counts training steps, not epochs
#   (unless you're not using the training_steps parameter)
weight_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                        'logs/models/small_resid_road_predictor_{epoch:04d}',
                                        monitor="val_loss",
                                        verbose=1,
                                        save_best_only=False,
                                        save_weights_only=True,
                                        mode="auto",
                                        save_freq=256,
                                        options=None
                                    )

# Compile model
opt = keras.optimizers.Nadam(learning_rate=0.0001)
loss_fn = keras.losses.mean_squared_error

ae.compile(opt, loss_fn)
print(ae.summary())
ae.load_weights('logs/models/resid_road_predictor_0054')

# Train it
ae.fit(ds, epochs = 100, callbacks=[monitor,weight_checkpoint], steps_per_epoch=256)
