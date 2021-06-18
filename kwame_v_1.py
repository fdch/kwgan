# -*- coding: utf-8 -*-
"""Kwame v.1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14pwwDOqpz5Tuh_JpXdnOmxcxs_bPf3dS
"""

# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import time, sys
import tensorflow as tf
import numpy as np 
import scipy.io.wavfile as wav
from os import listdir
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.layers import UpSampling1D, Conv1D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop, Adam

#------------------------------------------------------------------------------
# Training, model save, and audio export variables
#------------------------------------------------------------------------------

epochs_number          = 1001
model_save_interval    = 100
audio_export_interval  = 100
audio_export_per_epoch = 5
audio_samplerate       = 16000

BATCH_SIZE = 64
LATENT_DIM = 128
DIMS       = (2**14, 1)

wgan_dim        = 64
wgan_dim_mul    = 16
wgan_kernel_len = 25

# Number of times discr. is trained per generator train
n_discriminator = 5 

LAMBDA = 10
AUTOTUNE = tf.data.experimental.AUTOTUNE

#------------------------------------------------------------------------------
# paths and filenames
#------------------------------------------------------------------------------

job_suffix=sys.argv[1]

node_path="/users/PAS1309/fdch"
code_path="kwgan"
audio_prefix="kwg-9-" # audio filename prefix for audio export

dataset_path="sc09"
model_train_path = node_path + "/" + dataset_path + "/train"
model_test_path  = node_path + "/" + dataset_path + "/test"
model_save_path  = node_path + "/" + code_path + "/saved_model"
audio_save_path  = node_path + "/" + code_path + "/audio/train-9-"+str(job_suffix)

# # uncomment if not on google drive
# model_train_path = "/content/drive/MyDrive/Datasets/sc09/train"
# model_test_path  = "/content/drive/MyDrive/Datasets/sc09/test"
# model_save_path  = "/content/saved_model"
# audio_save_path  = "/content/audio"

Path(audio_save_path).mkdir(parents=True, exist_ok=True)
Path(model_save_path).mkdir(parents=True, exist_ok=True)

#------------------------------------------------------------------------------
# Making datasets
#------------------------------------------------------------------------------
def decode_audio(audio_binary):
  
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1)
    pad = (1 + DIMS[0] - len(audio)) // 2
    if pad > 0:
        paddings = [[pad, pad]]
        audio = tf.pad(audio, paddings, mode='CONSTANT')
    return  audio[:DIMS[0]]

def get_waveform(file_path):
#   label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform #, label

print("-"*80)
print("Making datasets...")
print("-"*80)

batch_dataset_start = time.time()

print(model_train_path)
print(model_test_path)
# find the filenames
train_files = tf.io.gfile.glob(model_train_path + "/*.wav")
test_files  = tf.io.gfile.glob(model_test_path  + "/*.wav")
print(f"Using {len(train_files)} files for training")
print(f"Using {len(test_files)} files for validation")
# shuffle filenames
train_files = tf.random.shuffle(train_files)
test_files  = tf.random.shuffle(test_files)
# make the dataset from tensor slices
train_files_ds = tf.data.Dataset.from_tensor_slices(train_files)
test_files_ds  = tf.data.Dataset.from_tensor_slices(test_files)

# map the function to read filenames into decoded arrays
train_ds = train_files_ds.map(get_waveform, num_parallel_calls=AUTOTUNE)
test_ds = test_files_ds.map(get_waveform, num_parallel_calls=AUTOTUNE)

# batch into dataset
train_ds = train_ds.batch(BATCH_SIZE)
val_ds  = test_ds.batch(BATCH_SIZE)

# cache and prefetch
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

print(f"Done making datasets. It took: {time.time()-batch_dataset_start} seconds.")

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
  dim = wgan_dim
  dim_mul = wgan_dim_mul
  kernel_len = wgan_kernel_len
  # Noise input
  z = Input(shape=(LATENT_DIM,), name='noise')
  output = z
  # WaveGAN arquitecture
  output = Dense(4*4*dim*dim_mul,activation='relu')(output)
  output = Reshape([1,16, dim * dim_mul])(output)
  # output = BatchNormalization()(output)
  dim_mul //= 2
  output = Conv2DTranspose(dim * dim_mul, (1,kernel_len), (1,4), padding='same')(output)
  # output = BatchNormalization()(output)
  output = tf.nn.relu(output)
  dim_mul //= 2
  output = Conv2DTranspose(dim * dim_mul, (1,kernel_len), (1,4), padding='same')(output)
  # output = BatchNormalization()(output)
  output = tf.nn.relu(output)
  dim_mul //= 2
  output = Conv2DTranspose(dim * dim_mul, (1,kernel_len), (1,4), padding='same')(output)
  # output = BatchNormalization()(output)
  output = tf.nn.relu(output)
  dim_mul //= 2
  output = Conv2DTranspose(dim * dim_mul, (1,kernel_len), (1,4), padding='same')(output)
  # output = BatchNormalization()(output)
  output = tf.nn.relu(output)
  output = Conv2DTranspose(1, (1,kernel_len), (1,4), padding='same')(output)
  output = tf.nn.tanh(output)
  output = Reshape(DIMS)(output)

  return tf.keras.Model(z, output)

#------------------------------------------------------------------------------
# phaseshuffle = lambda x: apply_phaseshuffle(x)
#------------------------------------------------------------------------------

def phaseshuffle(x, rad=2, pad_type='reflect'):
  b, x_len, nch = x.get_shape().as_list()

  phase = tf.random.uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
  pad_l = tf.maximum(phase, 0)
  pad_r = tf.maximum(-phase, 0)
  phase_start = pad_r
  x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)
  x = x[:, phase_start:phase_start+x_len]
  x.set_shape([b, x_len, nch])

  return x

#------------------------------------------------------------------------------
# discriminator model
#------------------------------------------------------------------------------

def get_discriminator():
  dim=wgan_dim
  kernel_len=wgan_kernel_len
  # Noise input
  x = Input((DIMS),name='audio')
  output = x
  # WaveGAN arquitecture
  output = Conv1D(dim, kernel_len, 4, padding='SAME')(output)
  output = phaseshuffle(output)
    # output = BatchNormalization()(output)
  output = tf.nn.leaky_relu(output)
  output = Conv1D(dim*2, kernel_len, 4, padding='SAME')(output)
  output = phaseshuffle(output)
    # output = BatchNormalization()(output)
  output = tf.nn.leaky_relu(output)
  output = Conv1D(dim*4, kernel_len, 4, padding='SAME')(output)
  output = phaseshuffle(output)
    # output = BatchNormalization()(output)
  output = tf.nn.leaky_relu(output)
  output = Conv1D(dim*8, kernel_len, 4, padding='SAME')(output)
    # output = BatchNormalization()(output)
  output = tf.nn.leaky_relu(output)
  output = Conv1D(dim*16, kernel_len, 4, padding='SAME')(output)
    # output = BatchNormalization()(output)
  output = tf.nn.leaky_relu(output)
  output = Reshape([DIMS[0]])(output)
  output = Dense(1)(output)

  return tf.keras.Model(x, output)

#------------------------------------------------------------------------------
# # set allow growth flag 
# # issue here https://github.com/tensorflow/tensorflow/issues/36025
# # and https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
# #------------------------------------------------------------------------------

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    tf.print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    tf.print(e)

#------------------------------------------------------------------------------
# build models
#------------------------------------------------------------------------------

generator = get_generator()
discriminator = get_discriminator()

# Adam optimizer with parameters from WAVEGAN
generator_optimizer = Adam(learning_rate=1e-4,beta_1=0.5,beta_2=0.9)   
discriminator_optimizer = Adam(learning_rate=1e-4,beta_1=0.5,beta_2=0.9)

#------------------------------------------------------------------------------
# loss functions
#------------------------------------------------------------------------------

@tf.function
def generator_loss(z):
    tf.print("*"*80)
    tf.print("Generator Loss")
    tf.print("*"*80)
    tf.print("Input shape: ", z.shape)
    fake = discriminator(generator(z))
    tf.print("Fake disc shape: ", fake.shape)
    gen_loss = -tf.reduce_mean(fake)
    return gen_loss

@tf.function
def discriminator_loss(x, z):
    tf.print("_"*80)
    tf.print("Discriminator Loss")
    tf.print("_"*80)
    epsilon = tf.random.uniform(shape=(x.shape[0], x.shape[1]), minval=0., maxval=1.)
    tf.print("epsilon shape:", epsilon.shape)
    gen = generator(z)
    tf.print("generated shape:", gen.shape)
    x_hat = epsilon * x + (1 - epsilon) * gen[:,0] 
    tf.print("x_hat shape:", x_hat.shape)
    d_hat = discriminator(x_hat)
    tf.print("d_hat shape:", d_hat.shape)
    ddx = tf.gradients(d_hat, x_hat)[0]
    ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
    ddx = tf.reduce_mean(tf.square(ddx - 1.0))
    tf.print("ddx shape", ddx.shape)
    fake = discriminator(generator(z))
    real = discriminator(x)
    dis_loss = (tf.reduce_mean(real) - tf.reduce_mean(fake) + LAMBDA) * ddx
    return dis_loss

#------------------------------------------------------------------------------
# GAN training step
#------------------------------------------------------------------------------
@tf.function
def train_discriminator_step(x):
    with tf.GradientTape() as disc_tape:
        _, _, _, disc_loss, _ = step(x)

    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))

@tf.function
def step(x):
    tf.print("="*30)
    tf.print("STEP")
    tf.print("="*30)
    tf.print("Input Shape: ", x.shape)
    # Match batch sizes with input batch size
    z = tf.random.normal(shape=(x.shape[0], LATENT_DIM))
    tf.print("Latent Z Shape: ", z.shape)
    # discriminating the real audio
    real = discriminator(x)
    tf.print("Real Disc Shape: ", real.shape)
    # discriminating the generated audio (fake)
    fake = discriminator(generator(z))
    tf.print("Fake Disc Shape: ", fake.shape)
    # take the losses
    gen_loss = generator_loss(z)
    tf.print("Generator Loss Shape: ", gen_loss.shape)
    disc_loss = discriminator_loss(x,z)
    tf.print("Discriminator Loss Shape: ", disc_loss.shape)
    return real, fake, gen_loss, disc_loss, z

@tf.function
def train_step(x):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        real, fake, gen_loss, disc_loss, z = step(x)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Fit Function (v 2.0 no longer using tqsm)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def fit(train_ds, epochs, test_ds):
    
  start_time = time.time()
  print("="*80)
  print(f"Begin at {start_time}")
  print("="*80)

  print("Epoch, Tst_Dl, Tst_Gl, Trn_Dl, Trn_Gl, Time")

  for epoch in range(epochs):
    start = time.time()
    train_loss = []
    test_loss  = []
    
    # trainning the discriminator more times
    for _ in range(n_discriminator):
      for audio_batch in train_ds:         
        train_discriminator_step(audio_batch)

    # train generator
    for audio_batch in train_ds:  
      disc_loss, gen_loss = train_step(audio_batch)
      train_loss.append([disc_loss, gen_loss])

    for audio_batch in test_ds:
      z = tf.random.normal([audio_batch.shape[0], LATENT_DIM])
      generated, real = generator(z), discriminator(audio_batch)
      fake = discriminator(generated)
      disc_loss = discriminator_loss(real, fake, generated)
      gen_loss = generator_loss(z)
      test_loss.append([disc_loss, gen_loss])
    
    tr_loss = np.asarray(np.mean(train_loss, axis=0))
    te_loss = np.asarray(np.mean(test_loss,  axis=0))
    
    if epoch != 0:
      # sample audio at export interval (not 0)
      if epoch % audio_export_interval == 0:
        sample_audio(epoch,z0,generator)

    # save the model at save interval (not 0)
    if epoch % model_save_interval  == 0:
      save_model(generator,     "KW_gen")
      save_model(discriminator, "KW_dis")
    
    time_to_train_epoch = time.time() - start
    
    tf.print(epoch,tr_loss[0],tr_loss[1],te_loss[0],te_loss[1],time_to_train_epoch)
  
  print("="*80)
  print(f"Ended at {time.time()}, took: {time.time()-start_time} secs.")
  print("="*80)

discriminator.summary()
generator.summary()

fit(train_ds, epochs_number, test_ds)