## notes taken while running the transformer.ipynb taken notebooks

dataset used was ted_hrlr_translate_pt_en_converter

words are tokenized to integers

during data prep, input lenghth of sentence is capped at MAX_TOKENS of 128.

### positional encoding

**positional encoding*** - one way to define relative coordinates among words

n, d = 2048, 512
pos_encoding = positional_encoding(n, d)

+ n is sentence length, d is coordinates (crazy relative coordinates)


### masking

"Mask all the pad tokens in the batch of sequence. It ensures that the model does not treat padding as the input. The mask indicates where pad value 0 is present: it outputs a 1 at those locations, and a 0 otherwise."

### look-ahead-mask

"The look-ahead mask is used to mask the future tokens in a sequence. In other words, the mask indicates which entries should not be used.

This means that to predict the third token, only the first and second token will be used. Similarly to predict the fourth token, only the first, second and the third tokens will be used and so on."


don't quiet get how/where masking & look-ahead-mask are used.


## scaled_dot_product_attention

**scaled_dot_product_attention** - query,key,value


`MultiHeadAttention` is the attention block


"Instead of one single attention head, Q, K, and V are split into multiple heads because it allows the model to jointly attend to information from different representation subspaces at different positions. After the split each head has a reduced dimensionality, so the total computation cost is the same as a single head attention with full dimensionality."


## Encoder & Decoder

Given input y 1,60,512 # (batch_size, encoder_sequence, d_model)

+ `encoder_sequence` is length of sentence

+ `d_model` is size post embedding

+ tensor post embedding becomes `normalized word vector` + `positional encoding` (so weird...)

## Transformer model

+ `look-ahead-mask`, `masking`

....


### don't quiet get it but started training with price & volatility data.




```