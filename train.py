#import os, sys
#sys.path.append(os.path.dirname(__file__))

import json
from tkinter import Y
import keras
from models import init, cnn_encoder,cnn_decoder, lvae_model, load_3ddata, \
                   dice_loss_sum_3d, dice_loss_every_frame, Adam, stoper, reduce_lr, show_result
                   #dice_loss_mean_3d


input_slices = list(range(0, 10, 2))
output_slices = list(range(0, 10, 1))
batch_size = 10

inputs3d = init(len(input_slices))

# model 初期化
encoder     = cnn_encoder(len(input_slices))
lvae        = lvae_model(add_kl_loss=True)
decoder     = cnn_decoder(len(output_slices))
output      = decoder(lvae( encoder(inputs3d) ))

model = keras.Model(inputs=inputs3d, outputs=output,name='model')
model.summary()


# data
trdata, valdata, tsdata = load_3ddata()
x = trdata[...,input_slices]
y = trdata[...,output_slices]

#val_x = valdata[...,input_slices]
#val_y = valdata[...,output_slices]
#val_x = x
#val_y = y

# train
model.compile(loss=dice_loss_sum_3d, metrics=dice_loss_every_frame, optimizer=Adam)  # dice_loss_every_frame
h = model.fit(x=x, y=y, batch_size=batch_size, epochs=300, callbacks=[stoper, reduce_lr], 
#validation_data=(x, y)
)
'''
for i in range(num_epochs):
    tr_x, tr_y, val_x, val_y = ??(i)
    model.fit(x=tr_x, y=tr_y, batch_size=batch_size, epochs=1, callbacks=[stoper, reduce_lr], validation_data=(val_x, val_y))
'''
h.history.pop('loss')
h.history.pop('lr')

# save
model.save('./result/model3d.h5')
with open('./result/history3d.json', 'w') as f: json.dump(h.history, f)
show_result(h.history)