import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.layers import Layer, Dense, LayerNormalization, Dropout, Flatten, Input
from keras.models import Model, load_model

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

# Build Model
def build_model(time_step, embed_dim=128, num_heads=8, ff_dim=512, num_layers=6, dropout_rate=0.1):
    inputs = Input(shape=(time_step, 3))
    x = Dense(embed_dim)(inputs)
    encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, rate=dropout_rate)
    x = encoder(x)
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    return model

# Define create_dataset function
def create_dataset(data, time_step=1):
    X, Y, Z = [], [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step)])
        Y.append(data[i + time_step, 2]) 
        Z.append(data[i+time_step,1])
    return np.array(X), np.array(Y), np.array(Z)

# Generate sample data
noise = np.random.normal(0,1,300)
time_inv = np.linspace(0, 364, 300) 
time = time_inv[::-1]
keff_inv = np.linspace(0, 2, 300) 
keff = keff_inv[::-1]
cr_pos_inv = np.linspace(0, 18, 300) 
cr_pos = cr_pos_inv[::-1]

# Create DataFrame
df = pd.DataFrame()
df['t'] = time
df['k'] = keff
df['cr'] = cr_pos

# Scale data
scaler_in = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler_in.fit_transform(df[['t', 'k', 'cr']].values)

# Create dataset
time_step = 20
X, Y ,Z= create_dataset(scaled_data, time_step)
X = X.reshape((X.shape[0], X.shape[1], 3))

# Load pre-trained model
model = build_model(time_step, embed_dim=128, num_heads=8, ff_dim=512, num_layers=6, dropout_rate=0.1)
model.load_weights('saved_model/model.weights.h5')

# Model prediction
predict = model.predict(X)
predict = scaler_in.inverse_transform(np.hstack((np.zeros((predict.shape[0], 2)), predict)))[:, 2]

# Inverse transform Y
Y_reshaped = Y.reshape(-1, 1)
dummy_array = np.zeros((Y_reshaped.shape[0], 3))
dummy_array[:, 2] = Y_reshaped.ravel()
y = scaler_in.inverse_transform(dummy_array)[:, 2]
# Inverse transform Z
Z_reshaped = Z.reshape(-1, 1)
dummy_array_Z = np.zeros((Z_reshaped.shape[0], 3))
dummy_array_Z[:, 2] = Z_reshaped.ravel()
z = scaler_in.inverse_transform(dummy_array)[:, 2]


# Print shapes
print(predict.shape)
predict = np.array(predict)  # Redundant but harmless
print(predict.shape)
fig,ax = plt.subplots()
#print(y,"pre",predict)
ax.plot(z,predict,label='prediction')
# Plot results
ax.plot(z,y, label='actual')
ax.set_xlim(0,12.5)
plt.legend()
plt.savefig("prediction_plot.png")
plt.show()
