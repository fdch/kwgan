# KWGAN

A GAN architecture to generate raw audio based on a dataset. This work is based on [wavegan](https://github.com/chrisdonahue/wavegan) and it is made using Tensorflow and Keras.

## Usage

For now just edit the variables within the python script to match your needs. 

```
epochs_number          = 40001
model_save_interval    = 1000
audio_export_interval  = 400
audio_export_per_epoch = 3

audio_samplerate       = 16000

TRAIN_BUF  = 2048
TEST_BUF   = 128
BATCH_SIZE = 256
LATENT_DIM = 128
DIMS       = (2**14,1)

gen_learning_rate     = 0.0001
disc_learning_rate    = 0.0002
number_of_disc_layers = 22

```

Then, simply use `python kwgan.py` to begin training. GPU usage is a must. This work was tested using *pitzer* at [https://www.osc.edu](https://www.osc.edu)