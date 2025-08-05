import os
import tensorflow as tf
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Layer, Dense, LayerNormalization, Dropout, Flatten
from keras.models import Model, Sequential
from keras import Input
from sklearn.preprocessing import MinMaxScaler
from keras.metrics import RootMeanSquaredError
from keras.callbacks import TensorBoard

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} logical GPUs')
    except RuntimeError as e:
        print(f"error setting up gpus: {e}")
else:
    print('no GPUs')

# MultiHeadSelfAttention Layer
class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        attention_output, _ = self.attention(query, key, value)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.embed_dim))
        return self.combine_heads(concat_attention)

# Transformer Block
class TransformBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = Sequential([Dense(ff_dim, activation='relu'), Dense(embed_dim),])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

# Transformer Encoder
class TransformerEncoder(Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.enc_layers = [TransformBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self, inputs, training=False):
        x = inputs
        x = self.dropout(x, training=training)
        for layer in self.enc_layers:
            x = layer(x, training=training)
        return x

# Create Dataset
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step)])
        Y.append(data[(i + time_step), 2])  # Targeting 'cr_pos' index
    return np.array(X), np.array(Y)

# Build Model
def build_model(time_step, embed_dim=128, num_heads=8, ff_dim=512, num_layers=4, dropout_rate=0.1):
    inputs = Input(shape=(time_step, 3))
    x = Dense(embed_dim)(inputs)
    encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, rate=dropout_rate)
    x = encoder(x)
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    return model

def main():
    np.random.seed(42)
    df_read = pd.read_csv("sim_keff_data.csv")

    # Use first 3000 rows for training
    df = pd.DataFrame()
    df['Time'] = df_read['time'].iloc[0:3000]
    df['keff'] = df_read['keff'].iloc[0:3000]
    df['cr_pos'] = df_read['rod_position'].iloc[0:3000]
    print(df.head())

    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Time', 'keff', 'cr_pos']].values)
    time_step = 20  # Increased time_step to capture more context
    X, Y = create_dataset(scaled_data, time_step)

    # Reshape for model
    X = X.reshape((X.shape[0], X.shape[1], 3))

    # Build and compile model
    model = build_model(time_step, embed_dim=128, num_heads=8, ff_dim=512, num_layers=6, dropout_rate=0.1)
    model.compile(optimizer='Adam', loss='mae', metrics=['mae', RootMeanSquaredError(name='rmse')])  # Changed to MAE for step-like data
    model.summary()

    # TensorBoard logging
    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir, exist_ok=True)
    tensorboard_cb = TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', profile_batch=1)
    print(f"Tensorboard logs in : {os.path.abspath(logdir)}")
    print(f"Run: tensorboard --logdir {logdir}")

    # Train model
    history = model.fit(X, Y, epochs=50, batch_size=32, validation_split=0.1, callbacks=[tensorboard_cb], verbose=1)

    # Prepare test data
    df_test = pd.DataFrame()
    noise = np.random.normal(0,1,4000)
    df_test['Time'] = df_read['time'].iloc[0:4000] +noise
    df_test['keff'] = df_read['keff'].iloc[0:4000] +noise
    df_test['cr_pos'] = df_read['rod_position'].iloc[0:4000] +noise
    scaler_test = MinMaxScaler(feature_range=(0, 1))
    scaled_data_test = scaler_test.fit_transform(df_test[['Time', 'keff', 'cr_pos']].values)
    X_test, Y_test = create_dataset(scaled_data_test, time_step)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 3))

    # Predictions
    predictions = model.predict(X_test)
    predictions = scaler_test.inverse_transform(np.hstack((np.zeros((predictions.shape[0], 2)), predictions)))[:, 2]
    Y_test_reshaped = Y_test.reshape(-1, 1)
    dummy_array = np.zeros((Y_test_reshaped.shape[0], 3))
    dummy_array[:, 2] = Y_test_reshaped.ravel()
    y_inverse = scaler_test.inverse_transform(dummy_array)[:, 2]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(y_inverse, label='True data')
    plt.plot(predictions, label='Predictions')
    plt.title('Transformer Time Series Forecasting')
    plt.xlabel('1000_DATA_STEP')
    plt.ylabel('CR_DATA')
    plt.legend()
    plot_path = os.path.join(logdir, 'prediction_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

if __name__ == '__main__':
    main()
