
# After grid search for 32 pixel images train, two models were selected and modified to work on 128 pixel images,
# This code created by alperenlcr@gmail.com and sevval.bulburu@std.yildiz.edu.tr


###############
### IMPORTS ###
###############

import os, sys, cv2
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
from datetime import datetime
from time import time
from skimage.color import rgb2lab, lab2rgb
from tensorflow.keras.layers import ( # type: ignore
    Activation, AveragePooling2D, BatchNormalization, Conv2D, Conv2DTranspose,
    Dense, Dropout, Flatten, Input, LeakyReLU, ReLU, UpSampling2D, MaxPool2D)
from tensorflow.keras.models import Sequential, Model # type: ignore


#################
### CONSTANTS ###
#################

IMAGE_SIZE = 128
EPOCHS = 150
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 25
WORKDIR = "/home/alperenlcr/bitirme/"


######################################
### DATASET LOAD AND PREPROCESSING ###
######################################

# Load dataset and convert images to LAB color space
def generate_dataset(images, debug=False):
    X = []
    Y = []

    for i in images:
        lab_image_array = rgb2lab(i / 255)
        x = lab_image_array[:, :, 0]
        y = lab_image_array[:, :, 1:]
        y /= 128  # normalize

        if debug:
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(i / 255)

            fig.add_subplot(1, 2, 2)
            plt.imshow(lab2rgb(np.dstack((x, y * 128))))
            plt.show()

        X.append(x.reshape(IMAGE_SIZE, IMAGE_SIZE, 1))
        Y.append(y)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    return X, Y

# Load data which created from video2dataset class
def load_data(force=False):
    is_saved_arrays_exist = os.path.isfile(os.path.join(WORKDIR+'dataset/', 'X_train.npy'))

    if not is_saved_arrays_exist or force:
        train_images = [cv2.imread(WORKDIR+'dataset/' + 'train/rgb/'+f) for f in os.listdir(WORKDIR+'dataset/' + 'train/rgb/')]
        test_images = [cv2.imread(WORKDIR+'dataset/' + 'test/rgb/'+f) for f in os.listdir(WORKDIR+'dataset/' + 'test/rgb/')]
        X_train, Y_train = generate_dataset(train_images)
        X_test, Y_test = generate_dataset(test_images)
        print('Saving processed data to computer')
        np.save(os.path.join(WORKDIR+'dataset/', 'X_train.npy'), X_train)
        np.save(os.path.join(WORKDIR+'dataset/', 'Y_train.npy'), Y_train)
        np.save(os.path.join(WORKDIR+'dataset/', 'X_test.npy'), X_test)
        np.save(os.path.join(WORKDIR+'dataset/', 'Y_test.npy'), Y_test)
    else:
        print('Loading processed data from computer')
        X_train = np.load(os.path.join(WORKDIR+'dataset/', 'X_train.npy'))
        Y_train = np.load(os.path.join(WORKDIR+'dataset/', 'Y_train.npy'))
        X_test = np.load(os.path.join(WORKDIR+'dataset/', 'X_test.npy'))
        Y_test = np.load(os.path.join(WORKDIR+'dataset/', 'Y_test.npy'))

    return X_train, Y_train, X_test, Y_test


# X_train, Y_train, X_test, Y_test = load_data()

# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
# test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

# train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
# test_dataset = test_dataset.batch(BATCH_SIZE)

def alternative():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    rgb_images = os.listdir(WORKDIR+'dataset/train/rgb/')
    # create a l and ab folders next to rgb folder
    for folder in ['l', 'ab']:
        os.makedirs(WORKDIR+'dataset/train/'+folder, exist_ok=True)
    
    # Dataset loading
    train_dataset_l = tf.keras.preprocessing.image_dataset_from_directory(
        WORKDIR+'dataset/train/rgb', image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=False, label_mode=None
    )
    train_dataset_l = train_dataset_l.map(lambda rgb: tfio.experimental.color.rgb_to_lab(rgb/255)[:, :, :, :1])
    

    train_dataset_ab = tf.keras.preprocessing.image_dataset_from_directory(
        WORKDIR+'dataset/train/rgb', image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=False, label_mode=None
    )
    train_dataset_ab = train_dataset_ab.map(lambda rgb: tfio.experimental.color.rgb_to_lab(rgb/255)[:, :, :, 1:]/128)

    train_dataset = tf.data.Dataset.zip((train_dataset_l, train_dataset_ab))

    return train_dataset

train_dataset = alternative()

X_test = np.load(os.path.join(WORKDIR+'dataset/', 'X_test.npy'))
Y_test = np.load(os.path.join(WORKDIR+'dataset/', 'Y_test.npy'))

##############
### MODELS ###
##############

#disc for shape 128x128
def discriminator():
    model = Sequential()
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(.25))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(.2))
    model.add(BatchNormalization())
    model.add(Dropout(.3))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

def generator2(): # autoencoder generator model
    def downsample(filters, kernel_size, apply_batchnorm=True):
        initializer = tf.random_uniform_initializer(0, 0.02)
        model = Sequential()
        model.add(Conv2D(filters, kernel_size, strides=2, padding='same',
                        kernel_initializer=initializer, use_bias=False))
        
        if apply_batchnorm:
            model.add(BatchNormalization())

        model.add(LeakyReLU())
        return model

    def upsample(filters, kernel_size, apply_dropout=False):
        initializer = tf.random_uniform_initializer(0, 0.02)
        model = Sequential()
        model.add(Conv2DTranspose(filters, kernel_size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))
        model.add(BatchNormalization())

        if apply_dropout:
            model.add(Dropout(0.5))

        model.add(ReLU())
        return model

    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

    # Downsampling layers
    # 1: (BATCH_SIZE, 16, 16, 32)
    # 2: (BATCH_SIZE, 8, 8, 64)
    # 3: (BATCH_SIZE, 4, 4, 128)
    # 4: (BATCH_SIZE, 2, 2, 256)
    # 5: (BATCH_SIZE, 1, 1, 256)

    downstack = [
        downsample(128, 4, apply_batchnorm=False),
        downsample(256, 4),
        downsample(512, 4),
        downsample(1024, 4),
        downsample(1024, 4)
    ]

    # Upsampling layers
    # 1: (BATCH_SIZE, 1, 1, 256)
    # 2: (BATCH_SIZE, 1, 1, 128)
    # 3: (BATCH_SIZE, 1, 1, 64)
    # 4: (BATCH_SIZE, 1, 1, 32)
    
    upstack = [
        upsample(1024, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
    ]

    initializer = tf.random_uniform_initializer(0, 0.02)
    output_layer = Conv2DTranspose(2, 3, strides=2, padding='same',
                                   kernel_initializer=initializer,
                                   activation='tanh')
    
    x = inputs

    # Downsampling layers
    skips = []
    for dm in downstack:
        x = dm(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling layers
    for um, skip in zip(upstack, skips):
        x = um(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    x = output_layer(x)

    return Model(inputs=inputs, outputs=x)

######################
### LOSS FUNCTIONS ###
######################

LAMBDA = 100
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = cross_entropy(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = cross_entropy(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


################
### TRAINING ###
################

def train(generator, discriminator, generator_optimizer, discriminator_optimizer, batch_size, csv_file):
    BATCH_SIZE = batch_size
    print('Training started')
    checkpoint_dir = os.path.join(csv_file[:-4]+'_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    if generator_optimizer['name'] == 'adam':
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=generator_optimizer['learning_rate'], beta_1=generator_optimizer['beta_1'])
    if discriminator_optimizer['name'] == 'adam':
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=discriminator_optimizer['learning_rate'], beta_1=discriminator_optimizer['beta_1'])
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    summary_log_file = os.path.join(
        WORKDIR, 'tf-summary', datetime.now().strftime("%Y%m%d-%H%M%S"))
    summary_writer = tf.summary.create_file_writer(summary_log_file)

    @tf.function(experimental_relax_shapes=True)
    def train_step(input_image, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            tf.keras.layers.concatenate([input_image, target])
            disc_real_output = discriminator(tf.keras.layers.concatenate([input_image, target]), training=True)
            disc_generated_output = discriminator(tf.keras.layers.concatenate([input_image, gen_output]), training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
                disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)
        
        return gen_total_loss, disc_loss


    checkpoint.restore(manager.latest_checkpoint)
    skip_training = False
    if manager.latest_checkpoint:
        print('Restored from {}'.format(manager.latest_checkpoint))
        skip_training = True
    else:
        print('Initializing from scratch')


    if not skip_training:
        for e in range(EPOCHS):
            start_time = time()
            gen_loss_total = disc_loss_total = 0
            for input_image, target in train_dataset:
                gen_loss, disc_loss = train_step(input_image, target, e)
                gen_loss_total += gen_loss
                disc_loss_total += disc_loss

            time_taken = time() - start_time

            if (e + 1) % 10 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
            
            print('Epoch {}: gen loss: {}, disc loss: {}, time: {:.2f}s'.format(
                e + 1, gen_loss_total / BATCH_SIZE, disc_loss_total / BATCH_SIZE, time_taken))
            
            with open(csv_file, 'a') as f:
                f.write(f'{e+1},{gen_loss_total / BATCH_SIZE},{disc_loss_total / BATCH_SIZE},{time_taken}\n')

    # plot and save
    sample_count = 20
    Y_hat = generator(X_test[:sample_count])

    for idx, (x, y, y_hat) in enumerate(zip(X_test[:sample_count], Y_test[:sample_count], Y_hat)):

        # Original RGB image
        orig_lab = np.dstack((x, y * 128))
        orig_rgb = lab2rgb(orig_lab)

        # Grayscale version of the original image
        grayscale_lab = np.dstack((x, np.zeros((IMAGE_SIZE, IMAGE_SIZE, 2))))
        grayscale_rgb = lab2rgb(grayscale_lab)

        # Colorized image
        predicted_lab = np.dstack((x, y_hat * 128))
        predicted_rgb = lab2rgb(predicted_lab)

        # print(idx)
        # convert to cv2 format
        grayscale_rgb = (grayscale_rgb * 255).astype(np.uint8)
        orig_rgb = (orig_rgb * 255).astype(np.uint8)
        predicted_rgb = (predicted_rgb * 255).astype(np.uint8)
        # concat grayscale_rgb, orig_rgb, predicted_rgb
        img = np.concatenate((grayscale_rgb, orig_rgb, predicted_rgb), axis=1)
        os.makedirs(csv_file[:-4], exist_ok=True)
        cv2.imwrite(csv_file[:-4]+f'/{idx}.png', img)


######################
### TESTING MODELS ###
######################

train(generator2(), discriminator(), {'name': 'adam', 'learning_rate': 0.0002, 'beta_1': 0.5}, {'name': 'adam', 'learning_rate': 0.0002, 'beta_1': 0.5}, 128, WORKDIR+'results.csv')
