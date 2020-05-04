# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import time
import tensorflow as tf
import numpy as np 
import scipy.io.wavfile as wav
from os import listdir
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.layers import UpSampling1D, Conv1D, UpSampling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop

#------------------------------------------------------------------------------
# Training, model save, and audio export variables
#------------------------------------------------------------------------------

epochs_number          = 40001
model_save_interval    = 1000
audio_export_interval  = 400
audio_export_per_epoch = 5

audio_samplerate       = 16000

TRAIN_BUF  = 2048 # 128
TEST_BUF   = 128  # 64
BATCH_SIZE = 256  # 32
LATENT_DIM = 128
DIMS       = (2**14,1)


gen_learning_rate     = 0.0001
disc_learning_rate    = 0.0001
number_of_disc_layers = 22
n_discriminator = 5 # Extra Number of times the discriminator is trained per generator train
weight_clip = 0.05 # Weight clip parameter as in WGAN

#------------------------------------------------------------------------------
# paths and filenames
#------------------------------------------------------------------------------

node_path="/users/PAS1309/fdch"
code_path="keras-wavegan"
dataset_path="sc09"
model_train_path=node_path+"/"+dataset_path+"/train"
model_test_path=node_path+"/"+dataset_path+"/test"
model_save_path=node_path+"/"+code_path+"/saved_model"
audio_save_path=node_path+"/"+code_path+"/audio/train-10"
audio_prefix="kwg-" # audio filename prefix for audio export

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

#  I would say no longer necessary!
# #------------------------------------------------------------------------------
# # set allow growth flag 
# # issue here https://github.com/tensorflow/tensorflow/issues/36025
# # and https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
# #------------------------------------------------------------------------------

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

#------------------------------------------------------------------------------
# check gpu
#------------------------------------------------------------------------------


print("tf.test.is_built_with_cuda():")
print(tf.test.is_built_with_cuda())
print("-----------------------")

print("tf.config.experimental.get_device_policy():")
print(tf.config.experimental.get_device_policy())
print("-----------------------")
testgpu=tf.test.is_gpu_available()
print("tf.test.is_gpu_available():")
print(testgpu)
print("-----------------------")

if not testgpu:
  print("Sorry, I have to go now. Good bye.")
  quit()


#------------------------------------------------------------------------------
# build models
#------------------------------------------------------------------------------

generator = get_generator()
discriminator = get_discriminator()

# We from RMSprop to Adam just to change it up a bit
generator_optimizer = tf.keras.optimizers.Adam(gen_learning_rate)     
discriminator_optimizer = tf.keras.optimizers.Adam(disc_learning_rate)

#------------------------------------------------------------------------------
# loss functions
#------------------------------------------------------------------------------

@tf.function
def generator_loss(z):
  fake_output = discriminator(generator(z))
  gen_loss = -tf.reduce_mean(fake_output)

  return gen_loss

@tf.function
def discriminator_loss(x_train,z):
  fake_output = discriminator(generator(z))
  real_output = discriminator(x_train)
  dis_loss = tf.reduce_mean(real_output)-tf.reduce_mean(fake_output)

  return dis_loss

#------------------------------------------------------------------------------
# GAN training step
#------------------------------------------------------------------------------
@tf.function
def train_discriminator_step(train_x,z):
  with tf.GradientTape() as disc_tape:
      disc_loss = discriminator_loss(x_train, z)
  discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))
  for t in range(number_of_disc_layers):
      y = tf.clip_by_value(discriminator.trainable_weights[t],clip_value_min=-weight_clip,clip_value_max=weight_clip,name=None)
      discriminator.trainable_weights[t].assign(y)

@tf.function
# Taken away training more on discriminator that generator
def train_step(train_x,z):
  with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:  
    gen_loss = generator_loss(z)
    disc_loss = discriminator_loss(x_train, z)
  
  generator_gradients = gen_tape.gradient(gen_loss,generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
  generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))
  
  return disc_loss, gen_loss

#------------------------------------------------------------------------------
# a pandas dataframe to save the loss information to
#------------------------------------------------------------------------------

#No longer using it


# losses = pd.DataFrame(columns = ['disc_loss', 'gen_loss'])


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
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Fit Function (v 2.0 no longer using tqsm)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def fit(train_dataset, epochs_number, test_dataset):
  tf.print("Begin training...")
  for epoch in range(epochs_number):
    start = time.time()
    # Train Loss
    loss=[]
    for n, train_x in train_dataset.enumerate():
      print('.', end='')
      #trainning the generator more times
      z=tf.random.normal([train_x.shape[0], LATENT_DIM])
      for k in range(n_discriminator):
        train_discriminator_step(train_x,z)
      
      disc_loss, gen_loss = train_step(train_x,z)
      loss.append([disc_loss, gen_loss])
    tf.print(epoch,'Train Losses: ' , np.mean(loss,axis=0))
    # Test Loss
    loss=[]
    for n, test_x in test_dataset.enumerate():
      z=tf.random.normal([test_x.shape[0], LATENT_DIM])
      loss.append([discriminator_loss(test_x,z),generator_loss(z)])
    tf.print(epoch,'Test Losses: ' , np.mean(loss,axis=0))

    if epoch != 0:
      # sample audios at export interval (not 0)
      if epoch % audio_export_interval == 0:
          sample_audio(epoch,z0,generator)
      # save the model at save interval (not 0)
      if epoch % model_save_interval  == 0:
          save_model(generator,     "KW_gen")
          save_model(discriminator, "KW_dis")
    time_to_train_epoch = time.time()-start
    tf.print (epoch, ': Training time {} sec,'.format( time_to_train_epoch ))
  tf.print("Finished training.")

fit(train_dataset, epochs_number, test_dataset)
