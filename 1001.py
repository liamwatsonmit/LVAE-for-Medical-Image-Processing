import h5py

def load_data():
    PATH = './datas/segmentation_test.h5'
    File = h5py.File(PATH, 'r')
    ts_Data = File['test'][:]
    File.close()
    return ts_Data

datas = load_data()

from matplotlib import pyplot as plt

plt.imshow(datas[1,40,...,5])

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(datas[1,...,5], edgecolor='k');
plt.show()