import os
import tensorflow as tf
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Layer, Dense, LayerNormalization, Dropout,Flatten
from keras.models import Model,Sequential
from keras import Input
import visualkeras
from sklearn.preprocessing import MinMaxScaler
from keras.metrics import RootMeanSquaredError
from keras.callbacks import TensorBoard


gpus = tf.config.list_physical_devices('GPU')
if gpus:
	try:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu,True)
		logical_gpus = tf.config.list_logical_devices('GPU')
		print(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} logical GPUs')
	except RuntimeError as e:
		print(f"error setting up gpus: {e}")
else:
	print('no gpus')
class MultiHeadSelfAttention(Layer):
	def __init__(self,embed_dim,num_heads = 8):
		super(MultiHeadSelfAttention,self).__init__()
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.projection_dim = embed_dim//num_heads
		self.query_dense = Dense(embed_dim)
		self.key_dense= Dense(embed_dim)
		self.value_dense= Dense(embed_dim)
		self.combine_heads = Dense(embed_dim)
	def attention(self,query,key,value):
		score = tf.matmul(query,key,transpose_b = True)
		dim_key = tf.cast(tf.shape(key)[-1],tf.float32)
		scaled_score = score/tf.math.sqrt(dim_key)
		weights = tf.nn.softmax(scaled_score,axis=-1)
		output = tf.matmul(weights,value)
		return output,weights
	def split_heads(self,x,batch_size):
		x = tf.reshape(x,(batch_size,-1,self.num_heads,self.projection_dim))
		return tf.transpose(x,perm=[0,2,1,3])
	def call(self,inputs):
		batch_size = tf.shape(inputs)[0]
		query = self.query_dense(inputs)
		key = self.key_dense(inputs)
		value = self.value_dense(inputs)
		query = self.split_heads(query,batch_size)
		key = self.split_heads(key,batch_size)
		value = self.split_heads(value,batch_size)
		attention_output,_=self.attention(query,key,value)
		attention_output = tf.transpose(attention_output,perm=[0,2,1,3])
		concat_attention = tf.reshape(attention_output,(batch_size,-1,self.embed_dim))
		return self.combine_heads(concat_attention)
class TransformBlock(Layer):
	def __init__(self,embed_dim,num_heads,ff_dim,rate=0.1):
		super(TransformBlock,self).__init__()
		self.att = MultiHeadSelfAttention(embed_dim,num_heads)
		self.ffn = Sequential([Dense(ff_dim,activation='relu'),Dense(embed_dim),])
		self.layernorm1 = LayerNormalization(epsilon = 1e-6)
		self.layernorm2 = LayerNormalization(epsilon=1e-6)
		self.dropout1 = Dropout(rate)
		self.dropout2 = Dropout(rate)
	def call(self,inputs,training=False):
		attn_output = self.att(inputs)
		attn_output = self.dropout1(attn_output,training=training)
		out1 = self.layernorm1(inputs+attn_output)
		ffn_output = self.ffn(out1)
		return self.layernorm2(out1+ffn_output)
class TransformerEncoder(Layer):
	def __init__(self,num_layers,embed_dim,num_heads,ff_dim,rate=0.1):
		super(TransformerEncoder,self).__init__()
		self.enc_layers = [TransformBlock(embed_dim,num_heads,ff_dim) for _ in range(num_layers)]
		self.dropout = Dropout(rate)
	def call(self,inputs,training=False):
		x = inputs
		x = self.dropout(x,training=training)
		for layer in self.enc_layers:
			x = layer(x,training=training)
		return x
def create_dataset(data,time_step=1):
	X,Y = [],[]
	for i in range(len(data)-time_step-1):
		X.append(data[i:(i+time_step),0])
		Y.append(data[(i+time_step),0])
	return np.array(X),np.array(Y)
def build_model(time_step,embed_dim=128,num_heads=8,ff_dim = 512,num_layers=4,dropout_rate=0.1):
	inputs = Input(shape=(time_step,1))
	x = Dense(embed_dim)(inputs)
	encoder = TransformerEncoder(num_layers,embed_dim,num_heads,ff_dim,rate=dropout_rate)
	x = encoder(x)
	x = Flatten()(x)
	x = Dropout(dropout_rate)(x)
	outputs = Dense(1)(x)
	return Model(inputs,outputs)
	
def main():
	np.random.seed(42)
	data_length = 2000
	trend1 = np.linspace(100,200,data_length)
	trend1 = np.sin(trend1)
	#trend2 = np.linspace(40,80,data_length)
	noise = np.random.normal(0,1,data_length)
	synthetic_data = trend1+noise
	df = pd.DataFrame(synthetic_data,columns=['Close'])
	print(df.head())
	scaler = MinMaxScaler(feature_range=(0,1))
	scaled_data = scaler.fit_transform(df[['Close']].values)
	time_step = 100
	X,Y = create_dataset(scaled_data,time_step)
	print(X.shape,Y.shape)
	X = X.reshape((X.shape[0],X.shape[1],1))
	model = build_model(time_step,embed_dim=128,num_heads=8,ff_dim=512,num_layers=4,dropout_rate = 0.1)
	model.compile(optimizer='Adam',loss='mse',metrics=['mae',RootMeanSquaredError(name='rmse')])
	model.summary()
	logdir = os.path.join("logs",datetime.now().strftime("%Y%m%d-%H%M%S"))
	os.makedirs(logdir,exist_ok=True)
	arch_path = os.path.join(logdir,'model_visualkeras.png')
	visualkeras.layered_view(model,to_file=arch_path,legend=True,draw_volume=False,scale_xy=1.5,scale_z=1,spacing=20)
	with tf.summary.create_file_writer(logdir).as_default():
		img = tf.io.read_file(arch_path)
		img = tf.image.decode_png(img,channels=4)
		tf.summary.image("Model_visualization",tf.expand_dims(img,0),step=0)
	tensorboard_cb = TensorBoard(log_dir = logdir,histogram_freq = 1,write_graph=True,write_images=True,update_freq='epoch',profile_batch=1)
	print(f"Tensorboard logs in : {os.path.abspath(logdir)}")
	print(f"Run: tensorboard--logdir{logdir}")
	history = model.fit(X,Y,epochs=20,batch_size = 32,validation_split=0.1,callbacks = [tensorboard_cb])
	loss = model.evaluate(X,Y)
	#print("Test loss (MSE): {:.6f}".format(loss))
	predictions = model.predict(X)
	predictions = scaler.inverse_transform(predictions)
	plt.figure(figsize=(10,6))
	plt.plot(df['Close'],label='True data')
	plt.plot(np.arange(time_step,time_step+len(predictions)),predictions,label = 'Predictions')
	plt.title('Transformer Time Series Forecasting')
	plt.xlabel('Time')
	plt.ylabel('synthetic data')
	plt.legend()
	plot_path=os.path.join(logdir,'prediction_plot.png')
	plt.savefig(plot_path,dpi=300,bbox_inches='tight')
	print(f"Plot saved to: {plot_path}")
	
if __name__=='__main__':
	main()
