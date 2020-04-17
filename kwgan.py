# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import tensorflow as tf
import pandas as pd
import numpy as np 
import scipy.io.wavfile as wav
from os import listdir
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.layers import UpSampling1D, Conv1D, UpSampling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
from IPython import display
from tqdm.autonotebook import tqdm

#------------------------------------------------------------------------------
# Training, model save, and audio export variables
#------------------------------------------------------------------------------

epochs_number          = 40001
model_save_interval    = 1000
audio_export_interval  = 400
audio_export_per_epoch = 3

audio_samplerate       = 16000

TRAIN_BUF  = 2048 # 128
TEST_BUF   = 128  # 64
BATCH_SIZE = 256  # 32
LATENT_DIM = 128
DIMS       = (2**14,1)

gen_learning_rate     = 0.0001
disc_learning_rate    = 0.0002
number_of_disc_layers = 22

#------------------------------------------------------------------------------
# paths and filenames
#------------------------------------------------------------------------------

node_path="/users/PAS1309/fdch"
code_path="keras-wavegan"
dataset_path="sc09"
model_train_path=node_path+"/"+dataset_path+"/train"
model_test_path=node_path+"/"+dataset_path+"/test"
model_save_path=node_path+"/"+code_path+"/saved_model"
audio_save_path=node_path+"/"+code_path+"/audio"
audio_prefix="aud-" # audio filename prefix for audio export

#------------------------------------------------------------------------------
# initialize random seed
#------------------------------------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

#------------------------------------------------------------------------------
# convert audio file to numpy array
#------------------------------------------------------------------------------

def audio_to_numpy(path):
  a=[]
  for i in listdir(path):
    u=np.zeros((2**14,1))
    _, y= wav.read(path+"/"+i)
    u[:y.size,0]=y
    a.append(u)
  a = np.asarray(a,dtype='float32')
  a /= 32768.
  return a

#------------------------------------------------------------------------------
# sample audio routine
#------------------------------------------------------------------------------

def sample_audio(e,z,gen):
    g = gen.predict(z)
    g *= 2**15
    for i in range(audio_export_per_epoch):
        f=audio_save_path+"/"+audio_prefix+str(e)+"-"+str(i)+".wav"
        wav.write(f,audio_samplerate,g[i])

#------------------------------------------------------------------------------
# model saving routine
#------------------------------------------------------------------------------

def save_model(m, n):
    p = model_save_path+"/%s.json" % n
    w = model_save_path+"/%s_weights.hdf5" % n
    options = {"file_arch": p, "file_weight": w}
    json_string = m.to_json()
    open(options['file_arch'], 'w').write(json_string)
    m.save_weights(options['file_weight'])

#------------------------------------------------------------------------------
# generator model
#------------------------------------------------------------------------------

def get_generator():
  model = Sequential()
  model.add(Dense(4*4*16*64, activation='relu', input_dim=LATENT_DIM))
  model.add(Reshape((2**10,2**4)))
  model.add(BatchNormalization(momentum=0.8))
  # output (2**11, 2**4)
  model.add(UpSampling1D())
  model.add(Conv1D(2**6, kernel_size=25,strides=4,activation="relu",padding="same"))
  model.add(BatchNormalization(momentum=0.8))
  # 
  model.add(UpSampling1D())
  model.add(Conv1D(2**8, kernel_size=25,strides=4,activation="relu",padding="same"))
  model.add(BatchNormalization(momentum=0.8))
  # 
  model.add(UpSampling1D())
  model.add(Conv1D(2**7, kernel_size=25,strides=1,activation="relu",padding="same"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(UpSampling1D())
  # 
  model.add(UpSampling1D())
  model.add(Conv1D(2**6, kernel_size=25,strides=1,activation="relu",padding="same"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(UpSampling1D())
  #
  model.add(UpSampling1D())
  model.add(Conv1D(1, kernel_size=25,strides=1,activation="relu",padding="same"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(UpSampling1D())
  #
  model.add(Activation("tanh"))
  model.summary()
  return model

#------------------------------------------------------------------------------
# discriminator model
#------------------------------------------------------------------------------

def get_discriminator():
  model = Sequential()
  # 16234x1
  model.add(Conv1D(2**6, kernel_size=25, strides=4, input_shape=DIMS, padding="same"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  # 4096x64
  model.add(Conv1D(2**7, kernel_size=25, strides=4, padding="same"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  # 1024x128
  model.add(Conv1D(2**8, kernel_size=3, strides=4, padding="same"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  # 256x256
  model.add(Conv1D(2**9, kernel_size=3, strides=4, padding="same"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  # 64x512
  model.add(Conv1D(2**11, kernel_size=3, strides=4, padding="same"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  # 16x2048
  model.add(Flatten())
  model.add(Dense(1))

  model.summary()
  return model

#------------------------------------------------------------------------------
# build models
#------------------------------------------------------------------------------

generator = get_generator()
discriminator = get_discriminator()

# RMSprop   in order to test where the error comes from
generator_optimizer = tf.keras.optimizers.RMSprop(gen_learning_rate)     
discriminator_optimizer = tf.keras.optimizers.RMSprop(disc_learning_rate)

#------------------------------------------------------------------------------
# loss function
#------------------------------------------------------------------------------

@tf.function
def compute_loss(train_x):
  x  = tf.random.normal([train_x.shape[0], LATENT_DIM])

  real_output = discriminator(train_x)
  fake_output = discriminator(generator(x))
  disc_loss = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
  gen_loss = -tf.reduce_mean(fake_output)

  return disc_loss, gen_loss

#------------------------------------------------------------------------------
# GAN training step
#------------------------------------------------------------------------------

@tf.function
def train_step(train_x,n_steps=4):
  x = tf.random.normal([train_x.shape[0], LATENT_DIM])
  for i in range(n_steps):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      real_output = discriminator(train_x)
      fake_output = discriminator(generator(x))
      
      disc_loss = -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)

      
      #if tf.math.is_nan(disc_loss) == False:
      gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
      discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
      t=0
      for t in range(number_of_disc_layers):
        y = tf.clip_by_value(discriminator.trainable_weights[t],clip_value_min=-0.05,clip_value_max=0.05,name=None)
        discriminator.trainable_weights[t].assign(y)

        #tf.print(discriminator.trainable_weights[1])

    
      if i == (n_steps-1) :
        fake_training_data = generator(x)
        fake_output = discriminator(fake_training_data)
        gen_loss = -tf.reduce_mean(fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

#------------------------------------------------------------------------------
# batch datasets
#------------------------------------------------------------------------------

tf.print("Making datasets...")

batch_dataset_start = time.time()

x_train=audio_to_numpy(model_train_path)
x_test=audio_to_numpy(model_test_path)

train_dataset = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .shuffle(TRAIN_BUF)
    .batch(BATCH_SIZE)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(x_test)
    .shuffle(TEST_BUF)
    .batch(BATCH_SIZE)
)

batch_dataset_end = time.time()-batch_dataset_start

tf.print("Finished makind datasets.")

tf.print ('Batch Dataset time is {} sec,'.format( batch_dataset_end ))

#------------------------------------------------------------------------------
# z0 noise same latent vector 
#------------------------------------------------------------------------------

z0 = np.random.normal(0, 1, (audio_export_per_epoch, LATENT_DIM))

#------------------------------------------------------------------------------
# a pandas dataframe to save the loss information to
#------------------------------------------------------------------------------

losses = pd.DataFrame(columns = ['disc_loss', 'gen_loss'])

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# start training
#------------------------------------------------------------------------------#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

N_TRAIN_BATCHES = int(TRAIN_BUF/BATCH_SIZE)
N_TEST_BATCHES  = int(TEST_BUF/BATCH_SIZE)

tf.print("Begin training...")

start = time.time()

for epoch in range(epochs_number):
    
    # train
    for batch, train_x in tqdm(
        zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES
    ):
        train_step(train_x)

    # test on holdout
    loss = []
    for batch, test_x in tqdm(
        zip(range(N_TEST_BATCHES), test_dataset), total=N_TEST_BATCHES
    ):
        loss.append(compute_loss(train_x))
    losses.loc[len(losses)] = np.mean(loss, axis=0)
    
    # plot results
    
    display.clear_output()
    print(
        "Epoch: {} | disc_loss: {} | gen_loss: {}".format(
            epoch, losses.disc_loss.values[-1], losses.gen_loss.values[-1]
        )
    )
    
    if epoch != 0:
      # sample audios at export interval (not 0)
      if epoch % audio_export_interval == 0:
          sample_audio(epoch,z0,generator)
      # save the model at save interval (not 0)
      if epoch % model_save_interval  == 0:
          save_model(generator,     "KW_gen")
          save_model(discriminator, "KW_dis")

time_to_train_gan = time.time()-start

tf.print("Finished training.")

tf.print ('Training time is {} sec,'.format( time_to_train_gan ))
