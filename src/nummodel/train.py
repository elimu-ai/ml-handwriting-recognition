
from gc import callbacks
from sqlite3 import converters
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import tf2onnx
import onnx

import config
from data import construct_image_from_number,distortion_free_resize,get_lab,clean_labels,clean_train_lab
from model import build_model,decoder




tf.config.run_functions_eagerly(True)
from tensorflow.python.ops.numpy_ops import np_config
tf.data.experimental.enable_debug_mode()
np_config.enable_numpy_behavior()
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

def preprocess_image(image_path, img_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def main():
    train_img_paths,train_labels,validation_img_paths,validation_labels,test_img_paths,test_labels = get_lab()
    max_len, characters, train_labels_cleaned = clean_train_lab(train_labels)
    print("MAX LEN",max_len)
    validation_labels_cleaned = clean_labels(validation_labels)
    test_labels_cleaned = clean_labels(test_labels)
    AUTOTUNE = tf.data.AUTOTUNE
    char_to_num = StringLookup(vocabulary = list(characters), mask_token = None)
    num_to_char = StringLookup(vocabulary = char_to_num.get_vocabulary(), mask_token = None, invert=True)
    model_dir = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
    log_dir = os.path.join(config.LOG_DIR, 'log-dir')

    writer = tf.summary.create_file_writer(log_dir)


    def vectorize_label(label):
        label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        length = tf.shape(label)[0]
        pad_amount = max_len-length
        label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=config.PADDING_TOKEN)
        return label

  
    def process_images_labels(image_path, label):
        image = preprocess_image(image_path)
        label = vectorize_label(label)
        return {"image": image, "label":label}

    def prepare_dataset(image_paths, labels):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
            process_images_labels, num_parallel_calls = AUTOTUNE
        )
        return dataset.batch(config.BATCH_SIZE)
    
    def calculate_edit_distance(labels, predictions):
        sparse_labels = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int64)
        input_len = np.ones(predictions.shape[0])*predictions.shape[1]
        predictions_decoded = keras.backend.ctc_decode(predictions, input_length = input_len, greedy=True)[0][0][:, :max_len]
        sparse_predictions = tf.cast(
            tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
        )
        edit_distances = tf.edit_distance(sparse_predictions, sparse_labels, normalize=False)
        return tf.reduce_mean(edit_distances)
    
    
    train_ds = prepare_dataset(train_img_paths, train_labels_cleaned)
    validation_ds = prepare_dataset(validation_img_paths, validation_labels_cleaned)
    test_ds = prepare_dataset(test_img_paths, test_labels_cleaned)

    validation_images = []
    validation_labels = []
    for batch in validation_ds:
        validation_images.append(batch["image"])
        validation_labels.append(batch["label"])

    class EditDistanceCallback(keras.callbacks.Callback):
        def __init__(self, pred_model):
            super().__init__()
            self.prediction_model = pred_model
        
        def on_epoch_end(self, epoch, logs=None):
            edit_distances = []
            for i in range(len(validation_images)):
                labels =  validation_labels[i]
                predictions = self.prediction_model.predict(validation_images[i])
                edit_distances.append(calculate_edit_distance(labels, predictions).numpy())
            
            print(
                f"Mean edit distance for epoch {epoch+1}: {np.mean(edit_distances):.4f}"
            )
    
    epochs = config.EPOCHS
    model = build_model(char_to_num.get_vocabulary())
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )

     # Create/Load checkpoint
    checkpoint_dir = os.path.join(model_dir, 'training-checkpoints')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                 save_weights_only=True,
                                                 verbose=1)


    edit_distance_callback = EditDistanceCallback(prediction_model)
    print("Model Training Going to Start")
    history = model.fit(
        train_ds,
        validation_data = validation_ds,
        epochs=epochs,
        callbacks=[edit_distance_callback, cp_callback]
    )
    model.save('models/nummodel')
    
    input_signature = [tf.TensorSpec([1,128,32,1], tf.float32, name='x')]
    onnx_model, _ = tf2onnx.convert.from_keras(prediction_model, input_signature, opset=13)
    onnx.save(onnx_model, "models/nummodel/model.onnx")


if __name__ == '__main__':
    main()