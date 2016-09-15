import os
import sys
import time
import numpy as np
from keras.optimizers import SGD
from scipy.io.wavfile import read, write
from keras.models import Model, Sequential
from keras.layers import Convolution1D, AtrousConvolution1D, Flatten, Dense, \
    Input, Lambda, merge


def get_discriminative_model(input_size):
    model = Sequential()
    model.add(Dense(input_size, activation='relu', input_dim=input_size))
    model.add(Dense(1, activation='sigmoid'))
    return model


def get_generative_model(output_size):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=100))
    model.add(Dense(output_size, activation='tanh'))
    return model


def get_generator_containing_disciminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def get_audio(filename):
    sr, audio = read(filename)
    audio = audio.astype(float)
    audio = audio - audio.min()
    audio = audio / (audio.max() - audio.min())
    audio = (audio - 0.5) * 2
    return sr, audio


def get_training_data(frame_size, frame_shift):
    sr, audio = get_audio('sample.wav')
    X_train = []
    base = 0
    n_possible_examples = int((len(audio) - frame_size) / float(frame_shift))
    # print 'Total number of possible samples:', n_possible_examples
    while len(X_train) < 10000:
        frame = audio[base:base+frame_size]
        X_train.append(frame)
        base += frame_shift
    X_train = np.array(X_train)
    return sr, np.array(X_train)


def get_uniform_noise(n):
    return np.random.uniform(0, 1, (n, 100))


if __name__ == '__main__':
    n_epochs = 10
    batch_size = 200
    frame_shift = 100
    frame_size = 4000
    n_audios_to_dump = 10
    model_dumping_freq = 5
    sr, X_train = get_training_data(frame_size, frame_shift)
    generator = get_generative_model(frame_size)
    discriminator = get_discriminative_model(frame_size)
    generator_containing_disciminator = get_generator_containing_disciminator(generator, discriminator)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer='sgd')
    generator_containing_disciminator.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    n_minibatches = int(X_train.shape[0]/batch_size)
    for i in range(n_epochs):
        print 'Epoch:', i+1
        d_losses = []
        g_losses = []
        for index in range(n_minibatches):
            noise = get_uniform_noise(batch_size)
            generated_audio = generator.predict(noise)
            audio_batch = X_train[index*batch_size:(index+1)*batch_size]
            X = np.concatenate((audio_batch, generated_audio))
            y = [1] * batch_size + [0] * batch_size
            d_loss = discriminator.train_on_batch(X, y)
            d_losses.append(d_loss)
            discriminator.trainable = False
            g_loss = generator_containing_disciminator.train_on_batch(noise, [1]*batch_size)
            g_losses.append(g_loss)
            discriminator.trainable = True
            sys.stdout.write(' + minibatch: ' + str(index+1) + '/' + str(n_minibatches) + '\r')
            sys.stdout.flush()
        mean_dloss = round(np.mean(d_losses), 2)
        mean_gloss = round(np.mean(g_losses), 2)
        print '\n + d_loss:', mean_dloss
        print ' + g_loss:', mean_gloss
        if i>0 and (i+1)%model_dumping_freq==0:
            str_timestamp = str(int(time.time()))
            gen_model_savepath = os.path.join('saved_models', 'gen_' + str_timestamp + '_' + str(mean_dloss) + '_' + str(mean_gloss) +'.h5')
            dis_model_savepath = os.path.join('saved_models', 'dis_' + str_timestamp + '_' + str(mean_dloss) + '_' + str(mean_gloss) +'.h5')
            print ' + saving models:', gen_model_savepath, dis_model_savepath
            generator.save(gen_model_savepath)
            discriminator.save(dis_model_savepath)
            print ' + generating audio samples:'
            gend_audio_dirpath = os.path.join('generated_audios', str_timestamp)
            os.makedirs(gend_audio_dirpath)
            counter = 0
            while counter < n_audios_to_dump:
                noise = get_uniform_noise(1)
                gend_audio = generator.predict(noise)
                if discriminator.predict(gend_audio)[0] > 0.5:
                    gend_audio = gend_audio[0]
                    gend_audio *= 2**15
                    outfile = str(counter)+'.wav'
                    outfilepath = os.path.join(gend_audio_dirpath, outfile)
                    print '   + ', outfilepath
                    write(outfilepath, sr, gend_audio.astype(np.int16))
                    counter += 1
