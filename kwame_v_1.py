# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import time, sys
import tensorflow as tf
import numpy as np 
import scipy.io.wavfile as wav
import glob

from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import activations
from tensorflow.keras.optimizers import RMSprop, Adam

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
# PhaseShuffle Class
#------------------------------------------------------------------------------

class PhaseShuffle(layers.Layer):

  def __init__(self, rad=2, pad_type='reflect'):
    super(PhaseShuffle, self).__init__()
    self.rad = rad
    self.pad_type = pad_type

  def build(self, shape):
    ph_init = tf.random_uniform_initializer(minval=-rad, maxval=rad+1)
    self.phase = tf.Variable(ph_init(shape=shape[-1],dtype=tf.int32))
    self.pad_l = tf.maximum(self.phase, 0)
    self.pad_r = tf.maximum(-self.phase, 0)
    self.phase_start = self.pad_r
    self.x_len = shape[1]

  def call(self, inputs):  # Defines the computation from inputs to outputs
    out = tf.pad(inputs, 
          [[0,0], [self.pad_l, self.pad_r], [0,0]], 
          mode=self.pad_type)

    return out[:, self.phase_start:self.phase_start+self.x_len]

#------------------------------------------------------------------------------
# loss functions
#------------------------------------------------------------------------------

@tf.function
def G_loss(z):
    # tf.print("*"*80)
    # tf.print("Generator Loss")
    # tf.print("*"*80)
    # tf.print("Input shape: ", z.shape)
    fake = D(G(z))
    # tf.print("Fake disc shape: ", fake.shape)
    gen_loss = -tf.reduce_mean(fake)
    return gen_loss

@tf.function
def D_loss(x, z):
    # tf.print("_"*80)
    # tf.print("Discriminator Loss")
    # tf.print("_"*80)
    epsilon = tf.random.uniform(
      shape=(x.shape[0], x.shape[1]), 
      minval=0., 
      maxval=1.)
    # tf.print("epsilon shape:", epsilon.shape)
    gen = G(z)
    # tf.print("generated shape:", gen.shape)
    x_hat = epsilon * x + (1 - epsilon) * gen[:,0] 
    # tf.print("x_hat shape:", x_hat.shape)
    d_hat = D(x_hat)
    # tf.print("d_hat shape:", d_hat.shape)
    ddx = tf.gradients(d_hat, x_hat)[0]
    ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
    ddx = tf.reduce_mean(tf.square(ddx - 1.0))
    # tf.print("ddx shape", ddx.shape)
    fake = D(G(z))
    real = D(x)
    dis_loss = (tf.reduce_mean(real) - tf.reduce_mean(fake) + LAMBDA) * ddx
    return dis_loss

#------------------------------------------------------------------------------
# steps
#------------------------------------------------------------------------------

@tf.function
def step(x):
    z = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    D_loss = D_loss(x, z)
    G_loss = G_loss(z)
    return G_loss, D_loss

@tf.function
def D_step(x):
    z = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    with tf.GradientTape() as D_tape:
        D_loss = D_loss(x, z)

    D_g = D_tape.gradient(D_loss, D.trainable_variables)
    D_opt.apply_gradients(zip(D_g, D.trainable_variables))

@tf.function
def train_step(x):
    with tf.GradientTape(persistemt=True) as tape:
        G_loss, D_loss, z = step(x)

    G_g = tape.gradient(G_loss, G.trainable_variables)
    D_g = tape.gradient(D_loss, D.trainable_variables)

    G_opt.apply_gradients(zip(G_g, G.trainable_variables))
    D_opt.apply_gradients(zip(D_g, D.trainable_variables))

    return G_loss, D_loss

#------------------------------------------------------------------------------
# useful globals
#------------------------------------------------------------------------------

EPOCHS = 1001       # number of epochs to train model
SAVE_INTERVAL = 100 # number of epochs to wait before saving the model
AUDIO_EXPORTS = 5   # number of audio to export per save interval
BATCH_SIZE = 64     # batches for training
LATENT_DIM = 128    # Input vector for generator (short noise)
SAMPLERATE = 16000  # Audio samplerate
DIMS = (2**14, 1)   # Input dimensions - one sec file aprox (16384, 1)
D_TRAIN = 5         # Number of times discr. is trained per generator train
DB_PERCENT = 5      # percentage of the database to use
LAMBDA = 10         # loss function param

#------------------------------------------------------------------------------
# Hyperparameters - taken mostly from the WaveGAN arquitecture
#------------------------------------------------------------------------------

filt   = 64  # dimension of the convolution filters
fmult  = 16  # filter dimension multiplier
size   = 25  # size or length of the kernel
strd   = 4   # strides
moment = 0.8 # momentum for the moving avg of the batch normalization
alpha  = 0.2 # leaky relu alpha parameter
rad    = 2   # range for phase shuffling (-rad, rad+1)
pad    = 'causal'  # pading for up/down convolution layers
pad_s  = 'reflect' # padding type for phase shuffling

# parameters for the generator's relu activation
relu = {
  "max": None, 
  "slope":0,
  "thresh":0
}

AUTOTUNE = tf.data.experimental.AUTOTUNE

if 4 * 4 * filt * fmult != DIMS[0]:
  print("Warning, wrong dims.")

#------------------------------------------------------------------------------
# Generator model
#------------------------------------------------------------------------------

G = models.Sequential([
  layers.InputLayer(shape=(LATENT_DIM,), name="G_Input"),

  # (LATENT_DIM,1) -> (16384, 1)
  layers.Dense(4 * 4 * filt * fmult, activation='relu', name="G_Dense"),
  
  # (16384, 1) -> [BATCH_SIZE, 16, 1024]
  layers.Reshape([BATCH_SIZE, 16, filt * fmult], name="G_Reshape_Input"),
  # batch size is assumed heretofore
  # (16, 1024) --> (64, 512)
  layers.Conv2DTranspose(filt*fmult//2, (1,size), (1,strd), padding=pad, name="G_UpConv-1"),
  layers.BatchNormalization(momentum=moment,name="G_Norm-1"),
  layers.ReLU(max_value=relu['max'], negative_slope=relu['slope'], threshold=relu['thresh'], name="G_Relu-1"),
  
  # (64, 512) --> (256, 256)
  layers.Conv2DTranspose(filt*fmult//4, (1,size), (1,strd), padding=pad, name="G_UpConv-2"),
  layers.BatchNormalization(momentum=moment,name="G_Norm-2"),
  layers.ReLU(max_value=relu['max'], negative_slope=relu['slope'], threshold=relu['thresh'], name="G_Relu-2"),
  
  # (256, 256) --> (1024, 128)
  layers.Conv2DTranspose(filt*fmult//8, (1,size), (1,strd), padding=pad, name="G_UpConv-3"),
  layers.BatchNormalization(momentum=moment,name="G_Norm-3"),
  layers.ReLU(max_value=relu['max'], negative_slope=relu['slope'], threshold=relu['thresh'], name="G_Relu-3"),
  
  # (1024, 128) --> (4096, 64)
  layers.Conv2DTranspose(filt*fmult//16, (1,size), (1,strd), padding=pad, name="G_UpConv-4"),
  layers.BatchNormalization(momentum=moment,name="G_Norm-4"),
  layers.ReLU(max_value=relu['max'], negative_slope=relu['slope'], threshold=relu['thresh'], name="G_Relu-4"),
  
  # (4096, 64) --> (16384,1)
  layers.Conv2DTranspose(1, (1,size), (1,strd), padding=pad, name="G_UpConv-5", activation=activations.tanh)

  ], name="Generator")


#------------------------------------------------------------------------------
# discriminator model
#------------------------------------------------------------------------------


D = models.Sequential([
  layers.InputLayer(shape=DIMS, name="D_Input"),

  # (16384,1) --> (4096, 64)
  layers.Conv1D(filt, size, strides=strd, padding=pad, name="D_DownConv-1"),
  PhaseShuffle(rad=rad, pad_type=pad_s, name="D_PhaseShuffle-1"),
  layers.BatchNormalization(momentum=moment,name="D_Norm-1"),
  layers.LeakyReLU(alpha,name="D_leaky-1"),

  # (4096, 64) --> (1024, 128)
  layers.Conv1D(filt, size, strides=strd, padding=pad, name="D_DownConv-2"),
  PhaseShuffle(rad=rad, pad_type=pad_s, name="D_PhaseShuffle-2"),
  layers.BatchNormalization(momentum=moment,name="D_Norm-2"),
  layers.LeakyReLU(alpha,name="D_leaky-2"),

  # (1024, 128) --> (256, 256)
  layers.Conv1D(filt, size, strides=strd, padding=pad, name="D_DownConv-3"),
  PhaseShuffle(rad=rad, pad_type=pad_s, name="D_PhaseShuffle-3"),
  layers.BatchNormalization(momentum=moment,name="D_Norm-3"),
  layers.LeakyReLU(alpha,name="D_leaky-3"),

  # (256, 256) --> (64, 512)
  layers.Conv1D(filt, size, strides=strd, padding=pad, name="D_DownConv-4"),
  PhaseShuffle(rad=rad, pad_type=pad_s, name="D_PhaseShuffle-4"),
  layers.BatchNormalization(momentum=moment,name="D_Norm-4"),
  layers.LeakyReLU(alpha,name="D_leaky-4"),
  
  # (64, 512) --> (16, 1024)
  layers.Conv1D(filt, size, strides=strd, padding=pad, name="D_DownConv-5"),
  PhaseShuffle(rad=rad, pad_type=pad_s, name="D_PhaseShuffle-5"),
  layers.BatchNormalization(momentum=moment,name="D_Norm-5"),
  layers.LeakyReLU(alpha,name="D_leaky-5"),

  layers.Dense(1, name="D_DenseLogit")

  ], name="Discriminator")


#------------------------------------------------------------------------------
# Adam optimizer with parameters from WAVEGAN
#------------------------------------------------------------------------------

G_opt = optimizers.Adam(learning_rate=1e-4,beta_1=0.5,beta_2=0.9)   
D_opt = optimizers.Adam(learning_rate=1e-4,beta_1=0.5,beta_2=0.9)


#------------------------------------------------------------------------------
# paths and filenames
#------------------------------------------------------------------------------


job_suffix = sys.argv[1]
db = "sc09"

if True:
  # in node
  PATH_NODE    = Path("/users/PAS1309/fdch")
  PATH_TRAIN   = PATH_NODE / db / "train"
  PATH_TEST    = PATH_NODE / db / "test"
  PATH_MODEL   = PATH_NODE / "kwgan" / "saved_model"
  PATH_AUDIO   = PATH_NODE / "kwgan" / "audio" 
else:
  # in colab
  PATH_TRAIN  = f"/content/drive/MyDrive/Datasets/{db}/train"
  PATH_TEST   = f"/content/drive/MyDrive/Datasets/{db}/test"
  PATH_MODEL  = "/content/saved_model"
  PATH_AUDIO  = "/content/audio"


print("-"*80)
print("Making datasets...")
print("-"*80)

Path(PATH_AUDIO).mkdir(parents=True, exist_ok=True)
Path(PATH_MODEL).mkdir(parents=True, exist_ok=True)

START_TIME = time.time()

# find the filenames
train_files = [ i for i in PATH_TRAIN.glob("/*.wav") ]
test_files  = [ i for i in PATH_TEST.glob("/*.wav") ]

train_files = train_files[:len(train_files) // 100 * DB_PERCENT]
test_files  = test_files[:len(test_files)   // 100 * DB_PERCENT]

print(f"Using {len(train_files)} files for training: {PATH_TRAIN.as_posix()}")
print(f"Using {len(test_files) } files for validation:{PATH_TEST.as_posix()}")

# shuffle filenames
train_files = tf.random.shuffle(train_files)
test_files  = tf.random.shuffle(test_files)

# make the dataset from tensor slices
train_files_ds = tf.data.Dataset.from_tensor_slices(train_files)
test_files_ds  = tf.data.Dataset.from_tensor_slices(test_files[:ts_split])

# map the function to read filenames into decoded arrays
DS_TRAIN = train_files_ds.map(get_waveform, num_parallel_calls=AUTOTUNE)
DS_TEST  = test_files_ds.map(get_waveform, num_parallel_calls=AUTOTUNE)

# batch into dataset, cache and prefetch
DS_TRAIN = DS_TRAIN.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
DS_TEST  = DS_TEST.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

print(f"Done making datasets in {time.time()-START_TIME} seconds.")


D.summary()
G.summary()

CHECKPOINT = tf.train.Checkpoint(G_opt=G_opt, D_opt=D_opt, G=G, D=D)

#------------------------------------------------------------------------------
# fit
#------------------------------------------------------------------------------

print("="*80)
print(f"Begin at {start_time}")
print("="*80)

print("Epoch, Tst_Dl, Tst_Gl, Trn_Dl, Trn_Gl, Time")

z0 = tf.random.normal([AUDIO_EXPORTS, LATENT_DIM])

for epoch in range(EPOCHS):
  start = time.time()
  train_loss = []
  test_loss  = []
  
  # trainning the discriminator more times
  for _ in range(D_TRAIN):
    for x in DS_TRAIN:
      D_step(x)

  # train generator
  for x in DS_TRAIN:
    G_loss, D_loss = train_step(x)
    train_loss.append([G_loss, D_loss])

  for x in DS_TEST:
    G_loss, D_loss = test_step(x)
    test_loss.append([G_loss, D_loss])
  
  tr_loss = np.array(np.mean(train_loss, axis=0))
  te_loss = np.array(np.mean(test_loss,  axis=0))
  
  # save the model at save interval (not 0)
  if epoch % SAVE_INTERVAL  == 0:
    tf.print("Saving model checkpoint")
    checkpoint.save(PATH_MODEL.resolve().as_posix())
    if epoch:
      tf.print("Exporting audio files")
      generated = G(z0, training=False)
      for i, audio in enumerate(generated):
        a = tf.audio.encode(audio, SAMPLERATE, name=f"Encode-{str(i)}")
        PATH_FILE = PATH_AUDIO / f"kw_aud-{str(epoch)}-{str(i)}.wav"
        wav.write(PATH_FILE.resolve.as_posix(), SAMPLERATE, a)
  
  time_to_train_epoch = time.time() - start
  
  tf.print(epoch, tr_loss, te_loss, time_to_train_epoch)

print("="*80)
print(f"Ended at {time.time()}, took: {time.time()-START_TIME} secs.")
print("="*80)
