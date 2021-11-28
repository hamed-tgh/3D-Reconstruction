
import numpy as np
from random import seed
from random import random
import matplotlib.pyplot as plt
from keras.models import Model ,load_model
from keras import backend as K
from keras.layers import *
from keras.layers.merge import concatenate as concat
import numpy as np
import scipy.io as sio
import h5py
from keras.callbacks import EarlyStopping
from keras.losses import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import keras
from os import listdir,makedirs
from os.path import isfile, join
from keras.callbacks import ModelCheckpoint
from PIL import Image
import sys
import tensorflow as tf
from sklearn.metrics import mean_squared_error as msqe


# seed random number generator
seed(1)
trdata_path_landmark = ''
trdata_path_alpha = ''
trdata_path_x3d = ''

tstdata_path_landmark = ''
tstdata_path_alpha = ''
tstdata_path_x3d = ''
   

data_path =''

file_list = [f for f in listdir(trdata_path_landmark) if isfile(join(trdata_path_landmark, f))]

file_list_test = [f for f in listdir(tstdata_path_landmark) if isfile(join(tstdata_path_landmark, f))]


batch_size=32
input_shape = (144,)

def Generator_direct():
    while True:
        landmark_files=np.random.choice(file_list, size=batch_size, replace=False)
        landmarks = [sio.loadmat(trdata_path_landmark+str(f))['landmarks_2d'] for f in landmark_files]
        landmarks = np.array(landmarks)
        landmarks = np.squeeze(landmarks)
        landmarks =  np.reshape(landmarks, (batch_size,144))
        for j in range(batch_size):    
            landmarks[j] = (landmarks[j]  - np.min(landmarks[j] ))/(np.max(landmarks[j] )-np.min(landmarks[j] ))
        # landmarks = (landmarks+ 85000)/(80000+85000)
        alpha = [sio.loadmat(trdata_path_alpha+str(f))['alpha'] for f in landmark_files]
        alpha = np.array(alpha)
        alpha = np.squeeze(alpha)
       
        yield (landmarks, alpha)


def Generator_direct_valid():
    while True:
        landmark_files=np.random.choice(file_list_test, size=batch_size, replace=False)
        landmarks = [sio.loadmat(tstdata_path_landmark+str(f))['landmarks_2d'] for f in landmark_files]
        landmarks = np.array(landmarks)
        landmarks = np.squeeze(landmarks)
        landmarks =  np.reshape(landmarks, (batch_size,144))
        for j in range(batch_size):    
            landmarks[j] = (landmarks[j]  - np.min(landmarks[j] ))/(np.max(landmarks[j] )-np.min(landmarks[j] ))
        # landmarks = (landmarks+ 85000)/(80000+85000)
        alpha = [sio.loadmat(tstdata_path_alpha+str(f))['alpha'] for f in landmark_files]
        alpha = np.array(alpha)
        alpha = np.squeeze(alpha)
       
        yield (landmarks, alpha)


input_landmark = Input(input_shape,name='landmark') 
statistical_test_run_num = 1

for j in range(statistical_test_run_num):

    dense1 = Dense(100, name = 'ff_h1', activation = 'linear')
    d1_bn = BatchNormalization()
    
    dense2 = Dense(50, name = 'ff_h2', activation = 'linear')
    d2_bn = BatchNormalization()
    
    dense3 = Dense(30, name = 'ff_h3', activation = 'linear')
    d3_bn = BatchNormalization()
    
    dense4 = Dense(50, name = 'ff_h4', activation = 'linear')
    d4_bn = BatchNormalization()
    
    dense5 = Dense(100, name = 'ff_h5', activation = 'linear')
    d5_bn = BatchNormalization()
    
    dense6 = Dense(199, name = 'alpha', activation = 'linear')
    d6_bn = BatchNormalization()
    
#############################################################################

    x2 = dense1(input_landmark)
    x2 = keras.layers.LeakyReLU(alpha=0.3)(x2)
    x2 = d1_bn(x2)
    
    x2 = dense2(x2)
    x2 = keras.layers.LeakyReLU(alpha=0.3)(x2)
    x2_d2 = d2_bn(x2)
    
    x2 = dense3(x2_d2)
    x2 = keras.layers.LeakyReLU(alpha=0.3)(x2)
    x2 = d3_bn(x2)
    
    x2 = dense4(x2)
    x2 = keras.layers.LeakyReLU(alpha=0.3)(x2)
    x2_d4 = d4_bn(x2)
    
    x2 = concatenate([x2_d2,x2_d4])
    x2 = dense5(x2)
    x2 = keras.layers.LeakyReLU(alpha=0.3)(x2)
    x2 = d5_bn(x2)
    
    x_alpha2 = dense6(x2)
    x2 = d6_bn(x_alpha2)
    
    
    inverse_renderer_test = Model(inputs = input_landmark , outputs = x_alpha2)
    inverse_renderer_test.summary()
    
    inverse_renderer_test.compile(optimizer = 'Adam' , loss = 'mean_squared_error')
    
    inverse_renderer_test.load_weights(data_path+'DM_alpha.h5')
    # history_IR_z = inverse_renderer_test.fit_generator(Generator_direct_valid(), steps_per_epoch =1000, epochs =50,
    #                                                    validation_data=Generator_direct_valid(), validation_steps=100)#,callbacks=[checkpointer])
    # inverse_renderer_test.save_weights(data_path+'DM_alpha.h5')
    
    ####################################################################################
    #                          Testing
    #####################################################################################
    # tstdata_path_landmark = 'E:\\thesis_phd\MDS\\landmark_alpha_test\\x2d\\'
    # tstdata_path_alpha = 'E:\\thesis_phd\MDS\\landmark_alpha_test\\alpha\\'
    # tstdata_path_x3d = 'E:\\thesis_phd\MDS\\landmark_alpha_test\\x3d\\'
    
        
    file_list_test = [f for f in listdir(tstdata_path_landmark) if isfile(join(tstdata_path_landmark, f))]

    res_path = ''
    pred_distance=[]
    for k in range(len(file_list_test)):
        x2d = sio.loadmat(tstdata_path_landmark+file_list_test[k])['landmarks_2d']
        x2d = np.squeeze(np.array(x2d))
        x2d =  np.reshape(x2d, (144,))
        x2d = (x2d - np.min(x2d))/(np.max(x2d)-np.min(x2d)) 
        x2d = np.expand_dims(x2d,axis=0)
        pred_alpha=np.squeeze(inverse_renderer_test.predict(x2d))
        
        sio.savemat(res_path+file_list_test[k],{'a': pred_alpha})
        
