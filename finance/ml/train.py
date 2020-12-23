import os
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
import scipy
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
import tensorflow as tf
import time

BUFFER_SIZE = 20000
BATCH_SIZE = 64

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert self.d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        #print(scaled_attention.get_shape(),'scaled_attention###',mask.get_shape() if mask is not None else None)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights



def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)


    def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2
    

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        #self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(self.d_model,self.d_model)

        
        #self.conv = tf.keras.layers.Conv1D(1, 5, activation='relu',padding='same')
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        #x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        #x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        #x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        #self.embedding = tf.keras.layers.Embedding(d_model, d_model)
        #self.pos_encoding = positional_encoding(d_model, d_model)
        
        #self.conv = tf.keras.layers.Conv1D(1, 5, activation='relu',padding='same')
        
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        #x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        #x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        #x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        #x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1,target_seq_len=20):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, rate)
        
        self.final_layer0 = tf.keras.layers.Dense(target_seq_len,activation='tanh')
        self.final_layer1 = tf.keras.layers.Dense(target_seq_len,activation='tanh')
        self.final_layer2 = tf.keras.layers.Dense(target_seq_len,activation='tanh')
        self.final_layer3 = tf.keras.layers.Dense(target_seq_len,activation='tanh')
        
        
    def call(self, inp, tar, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        final_output0 = self.final_layer0(dec_output[:,:,0])
        final_output1 = self.final_layer1(dec_output[:,:,1])
        final_output2 = self.final_layer2(dec_output[:,:,2])
        final_output3 = self.final_layer3(dec_output[:,:,3])
        final_output = tf.stack([final_output0,final_output1,final_output2,final_output3],axis=-1)
        
        return final_output, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def debug():
    
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    print(out.shape, attn.shape)
    
    ####################
    
    sample_ffn = point_wise_feed_forward_network(512, 2048)
    print(sample_ffn(tf.random.uniform((64, 50, 512))).shape)
    
    ####################
    
    # d_model, num_heads, dff
    sample_encoder_layer = EncoderLayer(512, 8, 2048)
    sample_encoder_layer_output = sample_encoder_layer(
        tf.random.uniform((64, 43, 512)), False, None)
    print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)
    
    ####################
    
    sample_decoder_layer = DecoderLayer(512, 8, 2048)
    sample_decoder_layer_output, _, _ = sample_decoder_layer(
        tf.random.uniform((64, 50, 512)), sample_encoder_layer_output, 
        False, None, None)
    print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)
    
    ####################
    
    print("------------------")
    input_seq_len = 80
    target_seq_len = 20
    batch_size = 32
    
    num_layers = 4
    d_model = 4
    dff = 4
    num_heads = 4
    dropout_rate = 0.1
    
    
    temp_mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    y = tf.random.uniform((1, input_seq_len, d_model))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    print(out.shape, attn.shape)
    
    ####################
    
    sample_ffn = point_wise_feed_forward_network(d_model, 2048)
    print(sample_ffn(tf.random.uniform((batch_size, input_seq_len, d_model))).shape)
    
    ####################
    
    sample_encoder_layer = EncoderLayer(d_model, num_heads, 2048)
    sample_encoder_layer_output = sample_encoder_layer(
        tf.random.uniform((batch_size, input_seq_len, d_model)), False, None)
    print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)
    
    ####################
    
    sample_decoder_layer = DecoderLayer(d_model, num_heads, 2048)
    sample_decoder_layer_output, _, _ = sample_decoder_layer(
        tf.random.uniform((batch_size, target_seq_len, d_model)), sample_encoder_layer_output, 
        False, None, None)
    print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)
    
    ####################
    
    sample_transformer = Transformer(num_layers, d_model,num_heads,dff)
    temp_input = tf.random.uniform((batch_size, input_seq_len, d_model), dtype=tf.float32, minval=-1, maxval=1)
    temp_target = tf.random.uniform((batch_size, target_seq_len, d_model), dtype=tf.float32, minval=-1, maxval=1)

    fn_out, _ = sample_transformer(temp_input, temp_target, training=False, 
                                   enc_padding_mask=None, 
                                   look_ahead_mask=None,
                                   dec_padding_mask=None)

    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
    
    ####################

    #loss_object = tf.keras.losses.Huber(delta=2.0)
    loss_object = tf.keras.losses.MeanSquaredError()
    def loss_function(real, pred):
        loss_ = loss_object(real, pred)
        return tf.reduce_sum(loss_)

    #"acc..."
    def accuracy_function(real, pred):
        accuracies = tf.keras.losses.MSE(real,pred)
        return tf.reduce_sum(accuracies)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    
    #############

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                         epsilon=1e-9)

    temp_learning_rate_schedule = CustomSchedule(d_model)

    ########
    
    transformer = Transformer(num_layers, d_model, num_heads, dff ,rate=dropout_rate)

    ########
    
    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    '''
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
    '''

    EPOCHS = 20

    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    train_step_signature = [
        tf.TensorSpec(shape=(None, input_seq_len, d_model), dtype=tf.float32),
        tf.TensorSpec(shape=(None, target_seq_len, d_model), dtype=tf.float32),

    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        
        #_inp = tf.random.uniform((batch_size, input_seq_len, 1), dtype=tf.float32, minval=8, maxval=8)
        #_tar = tf.random.uniform((batch_size, target_seq_len, 1), dtype=tf.float32, minval=-8, maxval=8)
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp[:,:,0], tar[:,:,0])

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar, 
                                         training=True, 
                                         enc_padding_mask=enc_padding_mask, 
                                         look_ahead_mask=look_ahead_mask, 
                                         dec_padding_mask=dec_padding_mask)
            
            loss = loss_function(tar, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)    
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar, predictions))
        
    X_train = tf.random.uniform((batch_size*10, input_seq_len, d_model), dtype=tf.float32, minval=8, maxval=8)
    y_train = tf.random.uniform((batch_size*10, target_seq_len, d_model), dtype=tf.float32, minval=-8, maxval=8)
    X_test = tf.random.uniform((batch_size*2, input_seq_len, d_model), dtype=tf.float32, minval=8, maxval=8)
    y_test = tf.random.uniform((batch_size*2, target_seq_len, d_model), dtype=tf.float32, minval=-8, maxval=8)

    #X_train, X_test, y_train, y_test = train_test_split(temp_input, temp_target, test_size=0.33, random_state=42)
    #print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

        if batch % 50 == 0:
            print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
              epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            pass
            #ckpt_save_path = ckpt_manager.save()
            #print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
            #                                                     ckpt_save_path))

        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                    train_loss.result(), 
                                                    train_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

def etl(history):
    df = pd.DataFrame()
    df['price'] = history.Close
    df.index = history.index
    df['log_ret'] = np.log(df.price) - np.log(df.price.shift(1))
    df['ret_mean'] = df.log_ret.rolling(21).mean()
    df['hist_vol'] = df.log_ret.rolling(21).std()*np.sqrt(252)*100
    df = df.dropna()
    df['z_vol']=np.clip(scipy.stats.zscore(df.hist_vol)/10,-1,1)
    df['z_ret']=np.clip(scipy.stats.zscore(df.ret_mean)/10,-1,1)
    df['month']=df.index.month.values/12
    df['day']=df.index.day.values/31
    # add in interest rate.
    return np.stack([df.z_vol.values,df.z_ret.values,df.month.values,df.day.values],axis=-1)

look_back=125
look_forward=8
total_days = look_back+look_forward-1
def chunckify(arr):
    tmp_list = []
    for x in np.arange(total_days,arr.shape[0]-total_days,5):
        tmp = arr[x:x+total_days]
        if tmp.shape != (total_days,4):
            continue
        x,y = tmp[:-1*look_forward,:],tmp[-1*look_forward:,:]
        tmp_list.append((x,y))
    return tmp_list

def train():
    data_exists = os.path.exists('X_train.npy')
    if data_exists:
        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')

    else:
        final_list = []
        whole_list_symbols = ['IWM','SPY','QQQ','GLD','SLV']
        df=pd.read_csv('https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv')
        whole_list_symbols.extend(list(df.Symbol.values))
        for x in np.arange(0,len(whole_list_symbols),100):
            try:
                symbols = whole_list_symbols[x:x+100]
                print(symbols)
                ticker_list = yf.Tickers(' '.join(symbols))
                for ticker in ticker_list.tickers:
                    try:
                        history = ticker.history(period="max")
                        print(ticker.ticker,history.shape)
                        arr = etl(history)
                        if arr.shape[0] > total_days:
                            tmp_list = chunckify(arr)
                            final_list.extend(tmp_list)
                    except:
                        pass
            except:
                pass
        X = np.stack([x[0][:,:] for x in final_list],axis=0).astype(np.float32)
        y = np.stack([x[1][:,:] for x in final_list],axis=0).astype(np.float32)
        X = np.expand_dims(X,axis=-1)
        y = np.expand_dims(y,axis=-1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
        
        np.save('X_train.npy', X_train)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    input_seq_len = 124
    target_seq_len = 8
    batch_size = 32
    
    num_layers = 4
    d_model = 4
    dff = 4
    num_heads = 4
    dropout_rate = 0.1
    

    #loss_object = tf.keras.losses.Huber(delta=10.0)
    loss_object = tf.keras.losses.MeanSquaredError()
    
    def loss_function(real, pred):
        loss_ = loss_object(real, pred)
        return tf.reduce_sum(loss_)

    #"acc..."
    def accuracy_function(real, pred):
        accuracies = tf.keras.losses.MSE(real,pred)
        return tf.reduce_sum(accuracies)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    
    eval_loss = tf.keras.metrics.Mean(name='train_loss')
    eval_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    
    #############

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                         epsilon=1e-9)

    temp_learning_rate_schedule = CustomSchedule(d_model)

    ########
    
    transformer = Transformer(num_layers, d_model, num_heads, dff ,rate=dropout_rate,target_seq_len=target_seq_len)

    ########
    
    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    '''
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
    '''

    EPOCHS = 40000

    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    train_step_signature = [
        tf.TensorSpec(shape=(None, input_seq_len, d_model), dtype=tf.float32),
        tf.TensorSpec(shape=(None, target_seq_len, d_model), dtype=tf.float32),
    ]
    
    @tf.function(input_signature=train_step_signature)
    def eval_step(inp, tar_real):
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp[:,:,0], tar[:,:,0])

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar,
                                         training=True, 
                                         enc_padding_mask=enc_padding_mask,
                                         look_ahead_mask=look_ahead_mask, 
                                         dec_padding_mask=dec_padding_mask)
            
            loss = loss_function(tar, predictions)

        eval_loss(loss)
        eval_accuracy(accuracy_function(tar_real, predictions))
        
    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar_real):
        
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp[:,:,0], tar[:,:,0])

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar,
                                         training=True, 
                                         enc_padding_mask=enc_padding_mask,
                                         look_ahead_mask=look_ahead_mask, 
                                         dec_padding_mask=dec_padding_mask)
            
            loss = loss_function(tar, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)    
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))
        
    X_train = tf.random.uniform((batch_size*10, input_seq_len, d_model), dtype=tf.float32, minval=8, maxval=8)
    y_train = tf.random.uniform((batch_size*10, target_seq_len, d_model), dtype=tf.float32, minval=-8, maxval=8)
    X_test = tf.random.uniform((batch_size*2, input_seq_len, d_model), dtype=tf.float32, minval=8, maxval=8)
    y_test = tf.random.uniform((batch_size*2, target_seq_len, d_model), dtype=tf.float32, minval=-8, maxval=8)

    #X_train, X_test, y_train, y_test = train_test_split(temp_input, temp_target, test_size=0.33, random_state=42)
    #print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.cache()
    test_dataset = test_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    
    def to_yaml(history):
        with open('history.yml','w') as f:
            f.write(yaml.dump(history))
            
    history = []        
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

        for (batch, (inp, tar)) in enumerate(test_dataset):
            eval_step(inp, tar)

        if batch % 50 == 0:
            print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
              epoch + 1, batch, train_loss.result(), train_accuracy.result()))
            
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                 ckpt_save_path))

        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                    train_loss.result(), 
                                                    train_accuracy.result()))
        
        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
          epoch + 1, eval_loss.result(), eval_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
        item = dict(
            epoch=epoch,
            train_loss=float(train_loss.result()),
            train_accuracy=float(train_accuracy.result()),
            eval_loss=float(eval_loss.result()),
            eval_accuracy=float(eval_accuracy.result()),
        )
        history.append(item)
        to_yaml(history)
        
if __name__ == '__main__':
    train()