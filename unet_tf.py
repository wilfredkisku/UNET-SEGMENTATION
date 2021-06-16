import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
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

def load_vgg(mainModel):

    selectedLayers = [1,2,9,10,17,18]
    
    vggModel = tf.keras.applications.vgg16.VGG16(include_top = False, weights='imagenet', input_shape = (512, 512, 3))
    vggModel.trainable = False
    for layer in vggModel.layers:
        layer.trainable = False
    
    selectedOutputs = [vggModel.layers[i].output for i in selectedLayers]
    lossModel = Model(vggModel.inputs,selectedOutputs)
    lossModelOutputs = lossModel(mainModel.outputs)

    fullModel = Model(mainModel.inputs, lossModelOutputs)

    return fullModel

def my_generator(batch, img_dir):
    dirs = glob.glob(img_dir + '/*')
    counter = 0
    while True:
        input_images = np.zeros((batch, width, height, 1*4))
        output_images = np.zeros((batch, width, height, 1))
        random.shuffle(dirs)
        if (counter+batch >= len(dirs)):
            counter = 0
        for i in range(batch):
            input_imgs = sorted(glob.glob(dirs[counter + i] + '/*'))
            imgs = []
            for j in range(len(input_imgs)-1):
                imgs.append(cv2.imread(input_imgs[j],0).reshape(width, height, 1))
            input_images[i] = np.concatenate(imgs,axis=2)
            output_images[i] = cv2.imread(input_imgs[4],0).reshape(width, height, 1)

            input_images[i] /= 255.
            output_images[i] /= 255.

        yield(input_images, output_images)
        counter += batch

def model_evaluate():

    model_new = create_model()
    adam = tf.keras.optimizers.Adam(learning_rate=tf.Variable(0.001),beta_1=tf.Variable(0.9),beta_2=tf.Variable(0.999),epsilon=tf.Variable(1e-7),decay = tf.Variable(0.0),)
    adam.iterations
    model_new.compile(optimizer=adam, loss=ssim_loss)
    model_new.load_weights(checkpoint_path)

    predict_path = '/home/wilfred/Downloads/github/Python_Projects/videoPrediction/data'
    dirs = sorted([f for f in glob.glob(predict_path+'/*') if os.path.isdir(f)])

    for d in dirs:
        input_images = np.zeros((1, width, height, 1 * 4))
        output_image = np.zeros((1, width, height, 1))
        input_imgs = sorted(glob.glob(d + '/*'))

        imgs = []

        for j in range(4):
            im = cv2.imread(input_imgs[j],0)
            if im.shape[0] % 2 == 1:
                w_c = im.shape[1]
                h_c = im.shape[0] - 1
                im = im[:h_c,w_c-h_c:w_c]
            else:
                w_c = im.shape[1]
                h_c = im.shape[0]
                im = im[:,w_c-h_c:w_c]
            im = cv2.resize(im, (120,120), interpolation = cv2.INTER_AREA)
            imgs.append(im.reshape(width, height, 1))

        input_images[0] = np.concatenate(imgs,axis=2)
        input_images[0] /= 255.

        output_image = model_new.predict(input_images)
        arr = output_image[0]
        new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
        cv2.imwrite(d+'/predicted.jpg',new_arr)
    return None

if __name__ == "__main__":
    #code
    input_shape = (512, 512, 9)
    model = build_unet(input_shape)
    model.summary()
    load_vgg(model)
    
    #es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
    #model.compile(optimizer = 'adam', loss = 'mse', metric = ['mse', 'mae'])

    #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)

    #history = model.fit(my_generator(batch_size,train_dir), steps_per_epoch=steps_per_epoch//4, validation_steps = validation_steps//4, epochs= num_epochs, validation_data = my_generator(batch_size, val_dir), callbacks=[es_callback, cp_callback], verbose = 1)

    #history_df = pd.DataFrame(history.history)
    #history_df.to_csv(saved_path+'/model-history.csv')
    #model.save(saved_path+'/model.h5')
    print('End of training ...')

    #model_evaluate()
    
    print('End of Evaluation ...')

    ################## another half ######################
    #compile, fit and train the model
    #model.compile(optimizer = 'adam', loss = 'mse' , metrics=['mse', 'mae'])
    #model_history =  model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, validation_data=test_dataset, callbacks=[DisplayCallback()])

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
