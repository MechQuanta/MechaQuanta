import tensorflow as tf
import datetime
mnist = keras.datasets.mnist
(X,train,Y_train),(X_test,Y_test) = mnist.load_data()
X_train,X_test = X_train/255.0,X_test/255.0
def create_model():
	model = keras.models.Sequential([keras.layers.Input(shape = (28,28),name = 'layers_input'),
	keras.layers.Flatten(name= 'layers_flatten'),
	keras.layers.Dense(512,activation='relu',name='layer_dense'),
	keras.layers.Dropout(0.2,name='layers_dropout'),
	keras.layers.Dense(10,activation='softmax',name='layer_dense_2')])
	return model
if __name__ == "__main__":
	model  = create_mode()
	model.compile(optimizer = 'adam',loss = 'sparse_categorical_cross_entropy',metrics = ['accuracy'])
	log_dir = "logs"+datetime.datetime.now().strftime("%Y%m%d")
	tensorboard_callback = keras.callbacks.Tensorboard(log_dir = log_dir,histogram_freq=1)

	model.fit(x = X_train,y=Y_train,epochs = 5,validation_data=(X_test,Y_test),callback = [tensorboard_callback])
