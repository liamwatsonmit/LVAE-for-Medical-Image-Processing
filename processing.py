from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import h5py

def show_result_picture(model, ts_Data_x, ts_Data_y, showfigure=True):
    y   = model.predict(ts_Data_x)[0]
    color_map   = np.stack([ts_Data_y[0], y, y*ts_Data_y[0]],axis=-1)
    plt.figure()
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(color_map[:,:,i,:])
        plt.axis('off')
        plt.title(i)#kl_score: 1 to 10
    if showfigure:plt.show()

def show_figure(loss_log,index=None):
    pd.DataFrame(loss_log).plot(figsize=(5, 5))
    plt.grid(True)
    plt.gca().set_ylim(1e-2, 1e0)  # y轴
    plt.gca().set_xlim(0, 140)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if index:plt.savefig("./result/dice(%d).png"%index)
    else: plt.show()
    #plt.clf()
    #plt.show()

def load_3ddata():
    Path_to_trainingDatas = "./datas/segmentation_train.h5"
    Path_to_validationDatas = "./datas/segmentation_eval.h5"
    Path_to_testingDatas = "./datas/segmentation_test.h5"
    
    tr_File = h5py.File(Path_to_trainingDatas, 'r')
    #tv_File = h5py.File(Path_to_validationDatas, 'r')
    ts_File = h5py.File(Path_to_testingDatas, 'r')

    tr_Data = tr_File['train'][:]
    #tv_Data = tv_File['eval'][:]
    tv_Data = None
    ts_Data = ts_File['test'][:]

    # to close the file
    tr_File.close()
    #tv_File.close()
    ts_File.close()
    
    return tr_Data, tv_Data, ts_Data

def show_result(loss_log):
    pd.DataFrame(loss_log).plot(figsize=(5, 5))
    plt.grid(True)  # 方格
    plt.gca().set_ylim(0, 1e0)  # y轴
    #plt.gca().set_xlim(0, 140)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig("./result/dice.png")
    plt.show()