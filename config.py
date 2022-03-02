from tensorflow import keras

stoper = keras.callbacks.EarlyStopping(
	monitor='loss',
	patience=15,  #当early stop被激活（如发现loss相比上一个epoch训练没有下降），则经过patience个epoch后停止训练。
	verbose=0,
	mode='auto'
	)


#lr_conf={'lr' : [1e-3,0.5e-3,1e-4,0.5e-4], 'boundary':[30,60,90]} # dice = 0.2171
lr_conf={'lr' : [1e-3,0.5e-3,1e-4,0.5e-4], 'boundary':[40,70,100]}
def lr_schedule(epoch):
    if epoch < lr_conf['boundary'][0]:lr = lr_conf['lr'][0]
    elif epoch < lr_conf['boundary'][1]:lr = lr_conf['lr'][1]
    elif epoch < lr_conf['boundary'][2]:lr = lr_conf['lr'][2]
    else: lr = lr_conf['lr'][3]
    return lr

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='loss',  # 被监测的量
    factor=0.5,  # 学习率将以lr = lr*factor的形式被减少
    patience=15,  # 当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
    verbose=0,
    mode='auto',  # ‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
    epsilon=0.1,  # 阈值，用来确定是否进入检测值的“平原区”
    cooldown=0,  # 学习率减少后，会经过cooldown个epoch才重新进行正常操作
    min_lr=1e-5  # 学习率的下限
)

Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)

boundaries=[40, 80, 100, 120]  # 以 0-20 20-40 40-60 60-80 80-inf 为分段
values=[1e-3, 0.5e-3, 1e-4, 0.5e-4, 1e-5]  # 各个分段学习率的值
piece_wise_constant_decay = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries, values=values, name=None)
Adam2 = keras.optimizers.Adam(piece_wise_constant_decay)