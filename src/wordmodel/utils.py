import tensorflow as tf


def convert_tflite(quantization,prediction_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(prediction_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    if quantization == 'float16':
        converter.target_spec.supported_types = [tf.float16]
    if quantization == 'full_int8':
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
    tf_lite_model = converter.convert()
    open(f'ocr_{quantization}.tflite', 'wb').write(tf_lite_model)

def decoder(y_pred):
    input_shape = tf.keras.backend.shape(y_pred)
    input_length = tf.ones(shape=input_shape[0]) * tf.keras.backend.cast(
        input_shape[1], 'float32')
    unpadded = tf.keras.backend.ctc_decode(y_pred, input_length)[0][0]
    unpadded_shape = tf.keras.backend.shape(unpadded)
    padded = tf.pad(unpadded,
                    paddings=[[0, 0], [0, input_shape[1] - unpadded_shape[1]]],
                    constant_values=-1)
    return padded

def CTCDecoder():
    return tf.keras.layers.Lambda(decoder, name='decode')