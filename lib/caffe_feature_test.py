
# coding: utf-8

# Reference:
# * https://github.com/fede1024/caffe-experiments/blob/master/memo.txt
# * https://github.com/TZstatsADS/Spr2016-Proj3-Grp3/blob/master/output/extractfeature.ipynb 
# * https://cdn.rawgit.com/TZstatsADS/ADS_Teaching/master/Spring2016/Tutorials/wk7-image_analysis/advanced_image_analysis.html

# ### 0. Input
# 
# * caffepath indicates the root path of the `caffe` package
# * inputpath indicates the folder which saves all the training images
# * inputpath_test indicates the folder which saves all the testing images
# * outputpath indicates the foler which the features extracted should be saved in

# In[2]:

caffepath = '/Users/YaqingXie/caffe'
inputpath = '/Users/YaqingXie/Desktop/BDA_Project/test_imgs'
outputpath = '/Users/YaqingXie/Desktop/BDA_Project/output/feature'


# ### 1. Setup
# 
# * First, set up Python, `numpy`, `panda`, `datetime` and `matplotlib`.

# In[3]:

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime


# * Load `caffe`.

# In[4]:

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys

if not caffepath.endswith('/'):
    caffepath = caffepath + '/'
if not inputpath.endswith('/'):
    inputpath = inputpath + '/'
if not outputpath.endswith('/'):
    outputpath = outputpath + '/'

caffe_root = caffepath
sys.path.insert(0, caffe_root + 'python')

import caffe


# * If needed, download the reference model ("CaffeNet", a variant of AlexNet).

# In[7]:

import os
if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print('CaffeNet found.')
else:
    print('Downloading pre-trained CaffeNet model...')
    get_ipython().system('../scripts/download_model_binary.py ../models/bvlc_reference_caffenet')


# ### 2. Load net and set up input preprocessing
# 
# * Set Caffe to CPU mode and load the net from disk.

# In[8]:

caffe.set_mode_cpu()

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


# * Set up input preprocessing. (We'll use Caffe's `caffe.io.Transformer` to do this, but this step is independent of other parts of Caffe, so any custom preprocessing code may be used).
# 
#     Our default CaffeNet is configured to take images in BGR format. Values are expected to start in the range [0, 255] and then have the mean ImageNet pixel value subtracted from them. In addition, the channel dimension is expected as the first (_outermost_) dimension.
#     
#     As matplotlib will load images with values in the range [0, 1] in RGB format with the channel as the _innermost_ dimension, we are arranging for the needed transformations here.

# In[9]:

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print('mean-subtracted values:', zip('BGR', mu))

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


# In[10]:

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227


# ### 3. Training Feature Extraction

# In[11]:

from os import listdir
from os.path import isfile, join
name_list = [f for f in listdir(inputpath) if isfile(join(inputpath, f)) and f.endswith('.jpg')]


# In[12]:

name_list = []
id_list = []
for path, subdirs, files in os.walk(inputpath):
    for name in files:
        if name.endswith('.jpg'):
            name_list.append(os.path.join(path, name))
            id_list.append(name)        


# In[14]:

data7_pca = pd.read_csv('/Users/YaqingXie/Desktop/BDA_Project/data/feature7_pca_eigen.csv')
data8_pca = pd.read_csv('/Users/YaqingXie/Desktop/BDA_Project/data/feature8_pca_eigen.csv')
data9_pca = pd.read_csv('/Users/YaqingXie/Desktop/BDA_Project/data/feature9_pca_eigen.csv')


# In[78]:

for i in range(0,len(id_list)):
    image = caffe.io.load_image(str(name_list[i]))
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    net.forward()
    feature7 = np.reshape(net.blobs['fc7'].data[0], 4096, order='C')
    feature8 = np.reshape(net.blobs['fc8'].data[0], 1000, order='C')
    feature9 = np.reshape(net.blobs['prob'].data[0], 1000, order='C')
    data7 = pd.DataFrame(feature7).transpose()
    data8 = pd.DataFrame(feature8).transpose()
    data9 = pd.DataFrame(feature9).transpose()
    data7.to_csv(outputpath+id_list[i]+"_feature7.csv", index=False)
    data8.to_csv(outputpath+id_list[i]+"_feature8.csv", index=False)
    data9.to_csv(outputpath+id_list[i]+"_feature9.csv", index=False)
    data7 = pd.DataFrame(np.dot(data7_pca, feature7)).transpose()
    data8 = pd.DataFrame(np.dot(data8_pca, feature8)).transpose()
    data9 = pd.DataFrame(np.dot(data9_pca, feature9)).transpose()
    data7.to_csv(outputpath+id_list[i]+"_feature7_pca.csv", index=False)
    data8.to_csv(outputpath+id_list[i]+"_feature8_pca.csv", index=False)
    data9.to_csv(outputpath+id_list[i]+"_feature9_pca.csv", index=False)


# In[ ]:



