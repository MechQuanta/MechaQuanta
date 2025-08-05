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
from keras.callbacks import TensorBoard, EarlyStopping

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

# Positional Encoding Layer
class PositionalEncoding(Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]  # (max_len, 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model))  # (d_model // 2,)
        
        pe[:, 0::2] = np.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = np.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        
        pe = pe[np.newaxis, :, :]  # shape: (1, max_len, d_model)
        self.pe = tf.Variable(pe, trainable=False, name='positional_encoding')
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        # Scale the positional encoding to be smaller relative to the input embeddings
        return inputs + self.pe[:, :seq_len, :] * 0.1

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
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Transformer Encoder
class TransformerEncoder(Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.enc_layers = [TransformBlock(embed_dim, num_heads, ff_dim, rate) for _ in range(num_layers)]
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
        Y.append(data[i + time_step, 2])  # Targeting 'cr_pos' index
    return np.array(X), np.array(Y)

# Build Model
def build_model(time_step, embed_dim=128, num_heads=8, ff_dim=512, num_layers=6, dropout_rate=0.1):
    inputs = Input(shape=(time_step, 3))
    x = Dense(embed_dim)(inputs)
    
    # Add positional encoding with proper max_len
    pos_encoding = PositionalEncoding(d_model=embed_dim, max_len=max(time_step, 100))
    x = pos_encoding(x)
    
    encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, rate=dropout_rate)
    x = encoder(x, training=True)
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    return model

def main():
    np.random.seed(42)

    # Load or generate data
    try:
        df_read = pd.read_csv("sim_keff_data.csv")
        df = pd.DataFrame()
        noise = np.random.normal(0,2.2,3000)
        df['Time'] = df_read['time'].iloc[0:3000] +noise
        df['keff'] = df_read['keff'].iloc[0:3000] +noise
        df['cr_pos'] = df_read['rod_position'].iloc[0:3000] +noise
    except FileNotFoundError:
        print("sim_keff_data.csv not found. Using synthetic data instead.")
        # Synthetic data (uncomment to use)
        time_inv = np.linspace(0, 365, 3000)
        time = time_inv[::-1]
        keff_inv = np.linspace(0, 2, 3000)
        keff = keff_inv[::-1]
        cr_pos_inv = np.linspace(0, 29, 3000)
        cr_pos = cr_pos_inv[::-1]
        df = pd.DataFrame({'Time': time, 'keff': keff, 'cr_pos': cr_pos})

    print(df.head())

    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Time', 'keff', 'cr_pos']].values)
    time_step = 20
    X, Y = create_dataset(scaled_data, time_step)

    # Reshape for model
    X = X.reshape((X.shape[0], X.shape[1], 3))

    # Build and compile model
    model = build_model(time_step, embed_dim=128, num_heads=8, ff_dim=512, num_layers=6, dropout_rate=0.1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mae', metrics=['mae', RootMeanSquaredError(name='rmse')])
    model.summary()

    # TensorBoard logging
    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir, exist_ok=True)
    tensorboard_cb = TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', profile_batch=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    print(f"Tensorboard logs in : {os.path.abspath(logdir)}")
    print(f"Run: tensorboard --logdir {logdir}")

    # Train model
    history = model.fit(X, Y, epochs=100, batch_size=32, validation_split=0.1, callbacks=[tensorboard_cb, early_stopping], verbose=1)
    model.save_weights('saved_model/model.weights.h5')

    # Prepare test data
    df_test = pd.DataFrame()
    noise = np.random.normal(0,1,4000)
    df_test['Time'] = df_read['time'].iloc[0:4000] if 'df_read' in locals() else df['Time'].iloc[0:4000]
    df_test['keff'] = df_read['keff'].iloc[0:4000] if 'df_read' in locals() else df['keff'].iloc[0:4000]
    df_test['cr_pos'] = df_read['rod_position'].iloc[0:4000] if 'df_read' in locals() else df['cr_pos'].iloc[0:4000]
    scaled_data_test = scaler.transform(df_test[['Time', 'keff', 'cr_pos']].values)  # Use same scaler
    X_test, Y_test = create_dataset(scaled_data_test, time_step)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 3))

    # Predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.hstack((np.zeros((predictions.shape[0], 2)), predictions.reshape(-1, 1))))[:, 2]
    Y_test_reshaped = Y_test.reshape(-1, 1)
    dummy_array = np.zeros((Y_test_reshaped.shape[0], 3))
    dummy_array[:, 2] = Y_test_reshaped.ravel()
    y_inverse = scaler.inverse_transform(dummy_array)[:, 2]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(y_inverse, label='True data')
    plt.plot(predictions, label='Predictions')
    plt.title('Transformer Time Series Forecasting')
    plt.xlabel('Data Step')
    plt.ylabel('CR Data')
    plt.legend()
    plot_path = os.path.join(logdir, 'prediction_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(logdir, 'loss_plot.png'))
    plt.show()

if __name__ == '__main__':
    main()
