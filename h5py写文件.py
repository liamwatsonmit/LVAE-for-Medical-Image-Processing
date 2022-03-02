#####5. h5py存放数据的容器
import h5py
import numpy as np

#写文件（默认二进制） w w+ 
f 		= h5py.File("data.hdf5","w")
d1 		= f.create_dataset("dset1",data=np.arange(20))
d2 		= f.create_dataset("dset2",(3,4),dtype='int') #'i'和'int'等价
f.close()

#读文件
#查询文件字典keys
of 		= h5py.File("data.hdf5","r")
keys	=list(of.keys())
print("keys:\n", keys)#['dset1', 'dset2']	
print("方法1\n", 	of['dset1'][:])
print("方法1\n",		of['dset2'][:])
print("方法2\n",		of['dset1'].value)
print("方法2\n",		of['dset2'].value)
of.close()



