# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import time, sys
import tensorflow as tf
import numpy as np 
import scipy.io.wavfile as wav
from os import listdir
from pathlib import Path
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

n_discriminator = 5 # Number of times discr. is trained per generator train
weight_clip     = 0.05 # Weight clip parameter as in WGAN

quitOnCPU=0

#------------------------------------------------------------------------------
# paths and filenames
#------------------------------------------------------------------------------

job_suffix=sys.argv[1]

node_path="/users/PAS1309/fdch"
code_path="kwgan"
dataset_path="sc09"
model_train_path=node_path+"/"+dataset_path+"/train"
model_test_path=node_path+"/"+dataset_path+"/test"
model_save_path=node_path+"/"+code_path+"/saved_model"
audio_save_path=node_path+"/"+code_path+"/audio/train-"+str(job_suffix)
audio_prefix="kwg-" # audio filename prefix for audio export

Path(audio_save_path).mkdir(parents=True, exist_ok=True)

#------------------------------------------------------------------------------
# convert audio file to numpy array
#------------------------------------------------------------------------------

def audio_to_numpy(path):
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
      np.append(a,u)
      np.append(b,k)
    a = np.asarray(a,dtype='float32')
    b = np.asarray(b)
    a /= 32768.
  return a, k

#------------------------------------------------------------------------------
# sample audio routine
#------------------------------------------------------------------------------

def sample_audio(e,z,y,gen):
    g = gen.predict([z,y])
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

def define_generator():
  initializer = tf.random_normal_initializer(0., 0.02)
  # label input
  label1 = Input(shape=(1,),name='label')
  label2 = Dense(2**2)(label1)
  label3 = Dense(2**4)(label2)
  label4 = Dense(2**6)(label3)
  label5 = Dense(2**7)(label4)
  # Noise input
  z = Input(shape=(LATENT_DIM,),name='noise')
  # Concatenate
  x=tf.keras.layers.Concatenate()([z, label5]) # (bs, 2^8)
  x=Reshape((2**8,1))(x)
  up_stack = [
    upsample(2**10, apply_dropout=True), # (bs, 2^9, 2^10)
    upsample(2**8, apply_dropout=True), # (bs, 2^10, 2^8)
    upsample(2**6, apply_dropout=True), # (bs, 2^11, 2^6)
    upsample(2**4), # (bs, 2^12, 2^4)
    upsample(2**2), # (bs, 2^13, 2^2)
    upsample(2**1), # (bs, 2^14, 2)
  ]
  
  # Upsampling 
  for up in up_stack:
    x = up(x)
  last = tf.keras.layers.Conv1D(1, kernel_size=25,strides=1,padding='same',kernel_initializer=initializer,activation='tanh')
  x = last(x)

  return tf.keras.Model([z,label1], outputs=x)
#------------------------------------------------------------------------------
# discriminator model
#------------------------------------------------------------------------------

def downsample(filters, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv1D(filters, kernel_size=25,strides=8,padding='same'))
  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def define_discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)
  # Label input
  label1 = Input(shape=(1,),name='Label')
  label2 = Dense(2**4)(label1)
  label3 = Dense(2**7)(label2)
  label4 = Dense(2**9)(label3)
  label5 = Dense(2**13)(label4)
  # Sound input
  x = Input(shape=(2**14,),name='Image')
  # Concatenate
  y=tf.keras.layers.Concatenate()([x, label5]) # (bs, 2^14, 2^14)
  y=Reshape((2**14+2**13,1))(y) # (bs, 2^15,1)
  
  down_stack = [
    downsample(2**9, apply_dropout=True), # (bs, 2^11, 2^9)
    downsample(2**7, apply_dropout=True), # (bs, 2^8, 2^7)
    downsample(2**5, apply_dropout=True), # (bs, 2^5, 2^5)
    downsample(2**3), # (bs, 2^3, 2^3)
    downsample(2**1), # (bs, 1, 2^1)
  ]
  
  # Upsampling 
  for down in down_stack:
    y = down(y)
  last = Dense(1)
  y = last(y)

  return tf.keras.Model([x,label1], outputs=y)

#------------------------------------------------------------------------------
# GPU Test
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

if not testgpu and quitOnCPU:
  print("Sorry, I have to go now: not using GPU. Good bye.")
  quit()


#------------------------------------------------------------------------------
# build models
#------------------------------------------------------------------------------

generator = define_generator()
discriminator = define_discriminator()
generator.summary()
discriminator.summary()

# We from RMSprop to Adam just to change it up a bit
generator_optimizer = tf.keras.optimizers.Adam(gen_learning_rate)     
discriminator_optimizer = tf.keras.optimizers.Adam(disc_learning_rate)

#------------------------------------------------------------------------------
# loss functions
#------------------------------------------------------------------------------

@tf.function
def generator_loss(z,label):
  fake_output = discriminator([generator([z,label]),label])
  gen_loss = -tf.reduce_mean(fake_output)

  return gen_loss

@tf.function
def discriminator_loss(x_train,z,label):
  fake_output = discriminator([generator([z,label]),label])
  real_output = discriminator([x_train,label])
  dis_loss = tf.reduce_mean(real_output)-tf.reduce_mean(fake_output)

  return dis_loss

#------------------------------------------------------------------------------
# GAN training steps
#------------------------------------------------------------------------------
@tf.function
def train_discriminator_step(train_x,z,label):
  with tf.GradientTape() as disc_tape:
      disc_loss = discriminator_loss(train_x, z, label)
  discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))
  for t in range(number_of_disc_layers):
      y = tf.clip_by_value(discriminator.trainable_weights[t],clip_value_min=-weight_clip,clip_value_max=weight_clip,name=None)
      discriminator.trainable_weights[t].assign(y)

@tf.function
# Taken away training more on discriminator that generator
def train_step(train_x,z,label):
  with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:  
    gen_loss = generator_loss(z,label)
    disc_loss = discriminator_loss(train_x, z, label)
  
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

x_train, y_train =audio_to_numpy(model_train_path)
x_test, y_test =audio_to_numpy(model_test_path)


train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train,y_train))
    .shuffle(TRAIN_BUF)
    .batch(BATCH_SIZE)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices((x_test,y_test))
    .shuffle(TEST_BUF)
    .batch(BATCH_SIZE)
)

batch_dataset_end = time.time()-batch_dataset_start

tf.print("Finished makind datasets.")

tf.print ('Batch Dataset time is {} sec,'.format( batch_dataset_end ))

quit()

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
  for epoch in range(epochs_number):
    start = time.time()
    # Train Loss
    loss=[]
    for n, (train_x, label) in train_dataset.enumerate():
      print('.', end='')
      #trainning the generator more times
      z=tf.random.normal([train_x.shape[0], LATENT_DIM])
      for k in range(n_discriminator):
        train_discriminator_step(train_x, z, label)
      
      disc_loss, gen_loss = train_step(train_x,z,label)
      loss.append([disc_loss, gen_loss])
    tf.print(epoch,'Train Losses: ' , np.mean(loss,axis=0))
    # Test Loss
    loss=[]
    for n, (test_x,label) in test_dataset.enumerate():
      z=tf.random.normal([test_x.shape[0], LATENT_DIM])
      loss.append([discriminator_loss(test_x,z,label),generator_loss(z,label)])
    tf.print(epoch,'Test Losses: ' , np.mean(loss,axis=0))
    if epoch != 0:
      # sample audios at export interval (not 0)
      if epoch % audio_export_interval == 0:
          sample_audio(epoch,z0,y0,generator)
      # save the model at save interval (not 0)
      if epoch % model_save_interval  == 0:
          save_model(generator,     "KW_gen")
          save_model(discriminator, "KW_dis")
    time_to_train_epoch = time.time()-start
    tf.print (epoch, ': Training time {} sec,'.format( time_to_train_epoch ))
  tf.print("Finished training.")

fit(train_dataset, epochs_number, test_dataset)
