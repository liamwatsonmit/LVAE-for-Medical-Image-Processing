from tensorflow import keras


#Cnn encoder
def cnn_encoder(num_input_slices):
    encoder = keras.models.Sequential([
        keras.layers.Reshape([80,80,80,num_input_slices], input_shape=[80,80,80,5]),
        keras.layers.Conv3D(8 ,3,strides=2,padding='same',activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3D(16,3,strides=2,padding='same',activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3D(32,3,strides=2,padding='same',activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3D(64,3,strides=2,padding='same', activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3D(2,1,strides=1,padding='same', activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.Flatten()
    ],name='cnn_encoder')
    return encoder

#cnn decoder
def cnn_decoder(num_output_slices):
    decoder = keras.models.Sequential([
        keras.layers.Reshape(target_shape=[5, 5, 5, 2],input_shape=[250]),
        keras.layers.Conv3DTranspose(64,3,strides=1,padding='same',activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3DTranspose(64,3,strides=2,padding='same',activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3DTranspose(32,3,strides=2,padding='same',activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3DTranspose(32,3,strides=2,padding='same',activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3DTranspose(28,3,strides=2,padding='same',activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3DTranspose(20,3,strides=1,padding='same',activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3DTranspose(num_output_slices,3,strides=1,padding='same',activation='sigmoid',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.Reshape([80,80,80,num_output_slices])
    ],name='cnn_decoder')
    return decoder
#conv_decoder.summary()

def cnn_model():
    cnn = keras.models.Sequential([cnn_encoder(), cnn_decoder()],name='cnn')
    return cnn

if __name__ == '__main__' :
        print('\n--------------in model cnn.py-------------\n')
        encoder, decoder, model = cnn_encoder(), cnn_decoder(), cnn_model()
        encoder.summary()
        decoder.summary()
        model.summary()
        print('\n-------------end of model cnn.py-------------\n')
