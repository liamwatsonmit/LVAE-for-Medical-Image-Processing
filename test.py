import h5py
import numpy as np
import h5py
import numpy as np
import numpy as np
import tensorflow.keras.backend as K
from tensorflow import keras
from models import Sampling, gen_mu_sigma_d, kl_loss_layer, dice_loss_sum_3d, load_3ddata, \
                   dice_loss_1, dice_loss_2, dice_loss_3, dice_loss_4, dice_loss_5, dice_loss_6, \
                   dice_loss_7, dice_loss_8, dice_loss_9, dice_loss_10, init
                   #dice_loss_mean_3d

input_slices = list(range(0, 10, 2))
output_slices = list(range(0, 10, 1))
batch_size = 10

inputs3d = init(len(input_slices))

# loading data
_, _, tsdata = load_3ddata()
x = tsdata[...,input_slices]
y = tsdata

def dice_mean_3d(y_true, y_pred, smooth=0.0001):#y_pred.shape -> (batch,80,80,10)
    y_true=K.cast(y_true,'float32')
    union        = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    intersection = K.sum(y_true * y_pred, axis=[1,2, 3])
    dice         = K.mean( (2. * intersection + smooth) / (union + smooth), axis=[0,1])
    return dice.numpy()

# loading model
objs = {'Sampling': Sampling, 'gen_mu_sigma_d': gen_mu_sigma_d, 'kl_loss_layer': kl_loss_layer,
        'dice_loss_sum_3d': dice_loss_sum_3d}
for i in range(1,11):exec('objs["dice_loss_%d"]=dice_loss_%d'%(i,i))
# objs['dice_loss_mean_3d']=dice_loss_mean_3d
model = keras.models.load_model('./result/model3d.h5',custom_objects=objs)

# testing
y_ = model.predict(x)
# y_ is probablity of each point whether true or false
# but voxel is classified either the point is there or not
# so its's 0 or 1 not probablity
cls_pt = np.where(y_>0.5, 1, 0)
#model.save('./result/model3d.h5')
DSC = dice_mean_3d(y, (y_>0.5).astype('float32'))
print( "DSC: ",DSC )
#f = h5py.File("test.hdf5","w")
#d1= f.create_dataset("dset1",data=cls_pt)

#f.close()

