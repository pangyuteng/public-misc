import tensorflow as tf

MultiHeadAttention= tf.keras.layers.MultiHeadAttention
layer = MultiHeadAttention(num_heads=2, key_dim=2)
target = tf.keras.Input(shape=[128,128,3])
source = tf.keras.Input(shape=[128,128,3])
output_tensor = layer(target, source)
output_tensor, weights = layer(target, source,return_attention_scores=True)
print(output_tensor.shape)
#(None, 8, 16)
print(weights.shape)
#(None, 2, 8, 4)

print(output_tensor.shape)
