import scipy.io.wavfile as wav
import tensorflow as tf
#from tf.keras.models import model_from_json
#import keras.backend as K
#from keras.optimizers import RMSprop
import numpy as np

model_from_json=tf.keras.models.model_from_json

n=100     # number of audio files to generate
dim=128  # dimensions
sr=16000 # sample rate

# define wasserstein loss function
def wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)


# model compilation arguments

mloss=wasserstein_loss # 'binary_crossentropy'
moptimizer=tf.keras.optimizers.RMSprop(lr=0.00005)
mmetrics=['accuracy']

# audio files and paths

afprefix="aud-kwg" # audio filename prefix for audio export
afformat="wav" # audio file format
afsubdir="audio" # audio sub directory

# model file and paths

msubdir="saved_model" # sub directory with saved models
mfilename="KW_gen" # model generator filename
mfformat="json" # model generator file format
wsuffix="_weights.hdf5" # model generator weights suffix

# load the json file
with open(msubdir+"/"+mfilename+"."+mfformat,"r") as f:
	lm_json = f.read()

# load json file as keras model
lm = model_from_json(lm_json)

# load model weights
lm.load_weights(msubdir+"/"+mfilename+wsuffix)

# compile the model
lm.compile(loss=mloss, optimizer=moptimizer, metrics=mmetrics)

# make a random array
noise = np.random.normal(0,1,(n,dim))

# make a prediction
audio = lm.predict(noise)

# write wav files
for i in range(n):
    a=[]
    fname=afsubdir+"/"+afprefix+"-"+str(i)+"."+afformat
    print("outputting",fname)
    a = audio[i]
    a *= 2**15
    wav.write(fname,sr,a)
