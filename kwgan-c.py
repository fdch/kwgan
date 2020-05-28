# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import time, sys
import tensorflow as tf
import numpy as np 
import scipy.io.wavfile as wav
from os import listdir
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.layers import UpSampling1D, Conv1D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop, Adam

#------------------------------------------------------------------------------
# Training, model save, and audio export variables
#------------------------------------------------------------------------------

epochs_number          = 40001
model_save_interval    = 100
audio_export_interval  = 25
audio_export_per_epoch = 5

audio_samplerate       = 16000

TRAIN_BUF  = 512
TEST_BUF   = 128
BATCH_SIZE = 64
LATENT_DIM = 128
DIMS       = (2**14,1)

wgan_dim        = 64
wgan_dim_mul    = 16
wgan_kernel_len = 25

n_discriminator = 5 # Number of times discr. is trained per generator train

#------------------------------------------------------------------------------
# paths and filenames
#------------------------------------------------------------------------------

job_suffix=str(sys.argv[1])

audio_prefix="ckwg-" # audio filename prefix for audio export


node_path="/users/PAS1309/fdch"
code_path="kwgan"
dataset_path="sc09"
model_train_path=node_path+"/"+dataset_path+"/train"
model_test_path=node_path+"/"+dataset_path+"/test"
model_save_path=node_path+"/"+code_path+"/saved_model"
audio_save_path=node_path+"/"+code_path+"/audio/train-"+job_suffix

Path(audio_save_path).mkdir(parents=True, exist_ok=True)

#------------------------------------------------------------------------------
# convert audio file to numpy array
#------------------------------------------------------------------------------
def indexed_audio_to_numpy(path):
  numbers= ['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine']
  allpaths=listdir(path)
  path_by_number = [[i for i in allpaths if number in i] for number in numbers]
  a=[]
  b=[]
  for k in range(10):
    for i in path_by_number[k]:
      u=np.zeros((2**14,1))
      _, y= wav.read(path+"/"+i)
      u[:y.size,0]=y
      a.append(u)
      b.append(k)
  a = np.asarray(a,dtype='float32')
  b = np.asarray(b)
  a /= 32768.
  return a, b

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

def sample_audio(e,z,gen,*y):
    if y:
      g = gen.predict([z,y])
    else:
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
# up/downsampling
#------------------------------------------------------------------------------

def upsample(filters, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(UpSampling1D())
  result.add(tf.keras.layers.Conv1D(filters, kernel_size=25,strides=1,padding='same'))
  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def downsample(filters, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv1D(filters, kernel_size=25,strides=8,padding='same'))
  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

#------------------------------------------------------------------------------
# phase shuffling
#------------------------------------------------------------------------------

# phaseshuffle = lambda x: apply_phaseshuffle(x)

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
# generator model
#------------------------------------------------------------------------------

# <<<<<<<< from old conditional

# def define_generator():
#   initializer = tf.random_normal_initializer(0., 0.02)
#   # label input
#   label1 = Input(shape=(1,),name='label')
#   label2 = Dense(2**2)(label1)
#   label3 = Dense(2**4)(label2)
#   label4 = Dense(2**6)(label3)
#   label5 = Dense(2**7)(label4)
#   # Noise input
#   z = Input(shape=(LATENT_DIM,),name='noise')
#   # Concatenate
#   x=tf.keras.layers.Concatenate()([z, label5]) # (bs, 2^8)
#   x=Reshape((2**8,1))(x)
#   up_stack = [
#     upsample(2**10, apply_dropout=True), # (bs, 2^9, 2^10)
#     upsample(2**8, apply_dropout=True), # (bs, 2^10, 2^8)
#     upsample(2**6, apply_dropout=True), # (bs, 2^11, 2^6)
#     upsample(2**4), # (bs, 2^12, 2^4)
#     upsample(2**2), # (bs, 2^13, 2^2)
#     upsample(2**1), # (bs, 2^14, 2)
#   ]
  
#   # Upsampling 
#   for up in up_stack:
#     x = up(x)
#   last = tf.keras.layers.Conv1D(1, kernel_size=25,strides=1,padding='same',kernel_initializer=initializer,activation='tanh')
#   x = last(x)

#   return tf.keras.Model([z,label1], outputs=x)


def get_generator():
  dim=wgan_dim
  dim_mul=wgan_dim_mul
  kernel_len=wgan_kernel_len
  # Noise input
  z = Input(shape=(LATENT_DIM,),name='noise')
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
# ================

#------------------------------------------------------------------------------
# discriminator model
#------------------------------------------------------------------------------



# <<<<<<<<<<<<<< from old conditional
# 
# def define_discriminator():
#   initializer = tf.random_normal_initializer(0., 0.02)
#   # Label input
#   label1 = Input(shape=(1,),name='Label')
#   label2 = Dense(2**4)(label1)
#   label3 = Dense(2**7)(label2)
#   label4 = Dense(2**9)(label3)
#   label5 = Dense(2**13)(label4)
#   # Sound input
#   x = Input(shape=(2**14,),name='Image')
#   # Concatenate
#   y=tf.keras.layers.Concatenate()([x, label5]) # (bs, 2^14, 2^14)
#   y=Reshape((2**14+2**13,1))(y) # (bs, 2^15,1)
  
#   down_stack = [
#     downsample(2**9, apply_dropout=True), # (bs, 2^11, 2^9)
#     downsample(2**7, apply_dropout=True), # (bs, 2^8, 2^7)
#     downsample(2**5, apply_dropout=True), # (bs, 2^5, 2^5)
#     downsample(2**3), # (bs, 2^3, 2^3)
#     downsample(2**1), # (bs, 1, 2^1)
#   ]
  
#   # Upsampling 
#   for down in down_stack:
#     y = down(y)
#   last = Dense(1)
#   y = last(y)

#   return tf.keras.Model([x,label1], outputs=y)


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
# ================


#------------------------------------------------------------------------------
# check gpu
#------------------------------------------------------------------------------

print("tf.test.is_built_with_cuda():")
print(tf.test.is_built_with_cuda())
print("-----------------------")

testgpu=tf.test.is_gpu_available()

print("tf.test.is_gpu_available():")
print(testgpu)
print("-----------------------")

if not testgpu:
  print("Sorry, I have to go now: cannot use GPU. Good bye.")
  quit()

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
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

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
  # <<<<<<< old conditional
  # fake_output = discriminator([generator([z,label]),label])
  fake_output = discriminator(generator(z))
  # ============
  gen_loss = -tf.reduce_mean(fake_output)

  return gen_loss

@tf.function
def discriminator_loss(x,z):
  fake_output = discriminator(generator(z))
  real_output = discriminator(x)
  dis_loss = tf.reduce_mean(fake_output)-tf.reduce_mean(real_output)

  epsilon = tf.random.uniform(shape=[x.shape[0], 1, 1], minval=0., maxval=1.)
  x_hat = epsilon * x + (1 - epsilon) * generator(z)
  d_hat = discriminator(x_hat)
  ddx = tf.gradients(d_hat, x_hat)[0]
  ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
  ddx = tf.reduce_mean(tf.square(ddx - 1.0))
  LAMBDA = 10
  dis_loss += LAMBDA * ddx
  return dis_loss

#------------------------------------------------------------------------------
# GAN training step
#------------------------------------------------------------------------------
@tf.function
# <<<<<<< old conditional
# def train_discriminator_step(train_x,z,label):
def train_discriminator_step(x,z):
# =======
  with tf.GradientTape() as disc_tape:
  # <<<<<<< old conditional
      # disc_loss = discriminator_loss(train_x, z, label)
      disc_loss = discriminator_loss(x, z)
  # ============
  discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))

@tf.function
# <<<<<<< old conditional
# def train_discriminator_step(train_x,z,label):
def train_step(x,z):
# =======
  with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:  
# <<<<<<< old conditional
    # gen_loss = generator_loss(z,label)
    # disc_loss = discriminator_loss(train_x, z, label)
    gen_loss = generator_loss(z)
    disc_loss = discriminator_loss(x, z)
# =======
  generator_gradients = gen_tape.gradient(gen_loss,generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
  generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))
  
  return disc_loss, gen_loss

#------------------------------------------------------------------------------
# batch datasets
#------------------------------------------------------------------------------

tf.print("Making datasets...")

batch_dataset_start = time.time()
# <<<<<<< old conditional

# x_train, y_train =indexed_audio_to_numpy(model_train_path)
# x_test, y_test =indexed_audio_to_numpy(model_test_path)

x_train=audio_to_numpy(model_train_path)
x_test=audio_to_numpy(model_test_path)
# =======

train_dataset = (
# <<<<<<< old conditional
    # tf.data.Dataset.from_tensor_slices((x_train,y_train))
    tf.data.Dataset.from_tensor_slices(x_train)
# =======
    .shuffle(TRAIN_BUF)
    .batch(BATCH_SIZE)
)

test_dataset = (
# <<<<<<< old conditional
    # tf.data.Dataset.from_tensor_slices((x_test,y_test))
    tf.data.Dataset.from_tensor_slices(x_test)
# =======
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
y0 = 9*np.ones(audio_export_per_epoch)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Fit Function (v 2.0 no longer using tqsm)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def fit(train_dataset, epochs_number, test_dataset):
  
  tf.print("Begin training...")

  tf.print("Epoch:  Tst_Dl . . ,Tst_Gl . . ,Trn_Dl . . ,Trn_Gl . . , Time")

  for epoch in range(epochs_number):
    start = time.time()

    train_loss=[]
    for n, train_x in train_dataset.enumerate():     
      #  trainning the generator more times
      z=tf.random.normal([train_x.shape[0], LATENT_DIM])      
      for k in range(n_discriminator):
# <<<<<<< old conditional
        # train_discriminator_step(train_x, z, label)
        train_discriminator_step(train_x,z)
        
      # disc_loss, gen_loss = train_step(train_x,z,label)
      disc_loss, gen_loss = train_step(train_x,z)
# =======
      train_loss.append([disc_loss, gen_loss])
    
    # Test Loss
    test_loss=[]
    for n, test_x in test_dataset.enumerate():
      z=tf.random.normal([test_x.shape[0], LATENT_DIM])
# <<<<<<< old conditional
      
      # loss.append([discriminator_loss(test_x,z,label),generator_loss(z,label)])
      test_loss.append([discriminator_loss(test_x,z),generator_loss(z)])
# =======
    tr_loss= np.asarray(np.mean(train_loss,axis=0))
    te_loss=np.asarray(np.mean(test_loss,axis=0))
    if epoch != 0:
      # sample audio at export interval (not 0)
      if epoch % audio_export_interval == 0:
        sample_audio(epoch,z0,generator,y0)

    # save the model at save interval (not 0)
    if epoch % model_save_interval  == 0:
      save_model(generator,     "c-KW_gen-"+job_suffix)
      save_model(discriminator, "c-KW_dis-"+job_suffix)
    
    time_to_train_epoch = time.time()-start
    
    tf.print(epoch,':',tr_loss[0],tr_loss[1],te_loss[0],te_loss[1],time_to_train_epoch)
  
  tf.print("Finished training.")



fit(train_dataset, epochs_number, test_dataset)

