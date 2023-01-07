import keras

# https://github.com/keras-team/keras-cv/blob/be35ce2e080bd32cb390459b030624604b79f06f/keras_cv/models/stable_diffusion/decoder.py

text_encoder_weights_fpath = keras.utils.get_file(
origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_encoder.h5",
file_hash="4789e63e07c0e54d6a34a29b45ce81ece27060c499a709d556c7755b42bb0dc4",
)
print(text_encoder_weights_fpath)
decoder_weights_fpath = keras.utils.get_file(
    origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_decoder.h5",
    file_hash="ad350a65cc8bc4a80c8103367e039a3329b4231c2469a1093869a345f55b1962",
)
print(decoder_weights_fpath)
diffusion_model_weights_fpath = keras.utils.get_file(
    origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_diffusion_model.h5",
    file_hash="8799ff9763de13d7f30a683d653018e114ed24a6a819667da4f5ee10f9e805fe",
)
print(diffusion_model_weights_fpath)
# image_encoder_weights_fpath = keras.utils.get_file(
#     origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/vae_encoder.h5",
#     file_hash="c60fb220a40d090e0f86a6ab4c312d113e115c87c40ff75d11ffcf380aab7ebb",
# )

