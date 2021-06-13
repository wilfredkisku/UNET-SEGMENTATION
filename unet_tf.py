import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2,2))(x)
    return x, p

def decode_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(inputs)
    x = Concatenate()([x,skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    inputs = Input(input_shape)

    """Encoder"""
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    """Bridge"""
    b1 = conv_block(p4, 1024)
    
    """Decoder"""
    d1 = decode_block(b1, s4, 512)
    d2 = decode_block(d1, s3, 256)
    d3 = decode_block(d2, s2, 128)
    d4 = decode_block(d3, s1, 64)

    outputs = Conv2D(3, (1,1), padding='same', activation='sigmoid')(d4)
    
    model = Model(inputs, outputs, name='U_Net')

    return model

if __name__ == "__main__":
    #code
    input_shape = (512, 512, 9)
    model = build_unet(input_shape)
    model.summary()
    
    #compile, fit and train the model
    model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model_history =  model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, validation_data=test_dataset, callbacks=[DisplayCallback()])

    #after data plotting
    #plot_model(model, to_file='data/diag.png', show_shapes=True)
    #loss = model_history.history['loss']
    #val_loss = model_history.history['val_loss']

    #plt.figure()
    #plt.plot(model_history.epoch, loss, 'r', label='Training loss')
    #plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
    #plt.title('Training and Validation Loss')
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss Value')
    #plt.ylim([0, 1])
    #plt.legend()
    #plt.show()
