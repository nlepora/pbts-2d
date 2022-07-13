#### Compatible with tensorflow v2.1.0 tensorflow-gpu v2.8.0 ########

import pandas as pd, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

for gpu in tf.config.experimental.list_physical_devices('GPU'): 
    tf.config.experimental.set_memory_growth(gpu, True)


# decorator for multi-output layer generators
class MultiOutputGenerator:
    def __init__(self, gen):
        self._gen = gen
    
    def __iter__(self):
        return self
    
    def __next__(self):
        ret = self._gen.next()
        x, y = ret
        y = [y_col for y_col in y.T] if y.ndim>1 else [y]
        return x, y
    
    @property
    def n(self):
        return self._gen.n
    
    @property
    def batch_size(self):
        return self._gen.batch_size
  
    
class CNNmodel:
    def __init__(self):
        self._model = None
        
    # build CNN model
    def build_model(self, 
                    num_conv_layers=5,
                    num_conv_filters=256,
                    num_dense_layers=1,
                    num_dense_units=64,
                    activation='relu', 
                    batch_norm=False,
                    dropout=0.0,
                    kernel_l1=0.0, 
                    kernel_l2=0.0,
                    target_names=['pose_2', 'pose_6'], 
                    size=(128,128),
                    lr=1e-4,
                    decay=1e-6,
                    **kwargs):
    
        input_dims = (size[0], size[1], 1)  # reverse order from cv2
        output_dim = len(target_names)
        num_hidden_layers = num_conv_layers + num_dense_layers
    
        # convolutional layers
        inp = tf.keras.Input(shape=input_dims, name='input')
        x = inp
        for i in range(num_conv_layers):
            x = layers.Conv2D(filters=num_conv_filters, kernel_size=3)(x)
            if batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            x = layers.MaxPooling2D()(x)
    
        # dense layers
        x = layers.Flatten()(x)
        for i in range(num_conv_layers, num_hidden_layers):
            if dropout > 0:
                x = layers.Dropout(dropout)(x)
            x = layers.Dense(units=num_dense_units,
                kernel_regularizer=regularizers.l1_l2(l1=kernel_l1, l2=kernel_l2))(x)         
            x = layers.Activation(activation)(x)            
    
        # Output layers
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
        out = [layers.Dense(units=1,
                    name=(f'output_{i+1}'),
                    kernel_regularizer=regularizers.l1_l2(l1=kernel_l1, l2=kernel_l2))(x)
                    for i in range(output_dim)]

        # Define and compile model   
        self._model = models.Model(inp, out)
        self._model.compile(optimizer=optimizers.Adam(learning_rate=lr, decay=decay),
                            loss=['mse',] * output_dim,
                            metrics=['mae']) 
    
    def fit_model(self, 
                  train_df_file,
                  train_image_dir, 
                  target_names, 
                  size, 
                  epochs,
                  batch_size=16,
                  poses_rng=None,
                  valid_df_file=None,
                  valid_image_dir=None, 
                  valid_split=0.0,
                  patience=10,
                  model_file='model.h5',
                  verbose=0,
                  **kwargs):      
        size = (size[1], size[0]) # reversed - strange bug

        target_df = pd.read_csv(train_df_file)
        
        if valid_image_dir is not None and valid_df_file is not None:
            train_df = target_df
            valid_df = pd.read_csv(valid_df_file)
        else:
            train_df, valid_df = train_test_split(target_df, test_size=valid_split)  
            valid_image_dir = train_image_dir

        pose_ = [f"pose_{_+1}" for _ in range(6)]
        self._inds = [pose_.index(t) for t in target_names]
        self._scale = [[poses_rng[_][i] for _ in reversed(range(2))] for i in range(6)] 
        
        for t, i in zip(target_names, self._inds):
            train_df[t] = np.interp(train_df[t], self._scale[i], (-1,1)) 
            valid_df[t] = np.interp(valid_df[t], self._scale[i], (-1,1))

        train_datagen = ImageDataGenerator(rescale=1./255.)   
        train_generator = train_datagen.flow_from_dataframe(
                                dataframe=train_df,
                                directory=train_image_dir,
                                x_col='image_name',
                                y_col=target_names,
                                batch_size=batch_size,
                                seed=42,
                                shuffle=True,
                                class_mode='raw',
                                target_size=size,
                                color_mode='grayscale')
        
        valid_datagen = ImageDataGenerator(rescale=1./255.)    
        valid_generator = valid_datagen.flow_from_dataframe(
                                dataframe=valid_df,
                                directory=valid_image_dir,
                                x_col='image_name',
                                y_col=target_names,
                                batch_size=batch_size,
                                seed=42,
                                shuffle=True,
                                class_mode='raw',
                                target_size=size,
                                color_mode='grayscale')        
        
        train_callbacks = [
            callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1),
            callbacks.ModelCheckpoint(filepath=model_file, monitor='val_loss', save_best_only=True)]                  
        
        history =  self._model.fit_generator(
                      generator=MultiOutputGenerator(train_generator),
                      steps_per_epoch=train_generator.n // train_generator.batch_size,
                      epochs=epochs,
                      verbose=verbose,
                      callbacks=train_callbacks,           
                      validation_data=MultiOutputGenerator(valid_generator),
                      validation_steps=valid_generator.n // valid_generator.batch_size)
        return history.history
    
    def predict(self, image, 
                      batch_size=None):     
        pred = self._model.predict(x=image, verbose=1, batch_size=batch_size)
        pred = [np.interp(p, (-1,1), self._scale[i]) for p,i in zip(pred,self._inds)]
        pred = np.array([np.squeeze(pred_array) for pred_array in pred]) 
        return pred.T

    def predict_from_file(self, test_image_dir, 
                          test_df_file,
                          target_names,
                          batch_size,
                          size, 
                          **kwargs):
        size = (size[1], size[0]) # reversed - strange bug                          
    
        test_datagen = ImageDataGenerator(rescale=1./255.)
        test_generator = test_datagen.flow_from_dataframe(
                            dataframe=pd.read_csv(test_df_file),
                            directory=test_image_dir,
                            x_col='image_name',
                            y_col=target_names,
                            batch_size=batch_size,
                            seed=42,
                            shuffle=False,
                            class_mode='raw',
                            target_size=size,
                            color_mode='grayscale')
            
        pred = self._model.predict_generator(generator=test_generator, verbose=1)  
        pred = [np.interp(p, (-1,1), self._scale[i]) for p,i in zip(pred,self._inds)]
        pred = np.array([np.squeeze(pred_array) for pred_array in pred])
        return pred.T
         
    def save_model(self, model_file, **kwargs):
        self._model.save(model_file)
            
    def load_model(self, model_file, 
                        target_names = ['pose_2', 'pose_6'], 
                        poses_rng = [[0,5,0,0,0,45],[0,-5,0,0,0,-45]], 
                        **kwargs):
        self._model =  models.load_model(model_file)
    
        pose_ = [f"pose_{_+1}" for _ in range(6)]
        self._inds = [pose_.index(t) for t in target_names]
        self._scale = [[poses_rng[_][i] for _ in reversed(range(2))] for i in range(6)] 

    def print_model_summary(self):
        self._model.summary()
    

def main():
    pass
    
if __name__ == '__main__':
    main() 