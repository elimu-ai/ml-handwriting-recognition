
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

import config

np.random.seed(42)

base_image_path = os.path.join(config.BASE_PATH, "words")

def get_samples():
    words_list=[]
    words = open(f"{config.BASE_PATH}/words.txt","r").readlines()
    for line in words:
        if line[0]=="#":
            continue
        if line.split(" ")[1] != "err":
            words_list.append(line)

    np.random.shuffle(words_list)

    split_idx = int(0.8*len(words_list))
    train_samples = words_list[:split_idx]
    test_samples = words_list[split_idx:]

    val_split_idx = int(0.5*len(test_samples))
    validation_samples = test_samples[:val_split_idx]
    test_samples = test_samples[val_split_idx:]

    assert len(words_list) == len(train_samples) + len(validation_samples) + len(test_samples)

    print(f"Total training Samples:{len(train_samples)}")
    print(f"Toal validation samples: {len(validation_samples)}")
    print(f"Total test samples: {len(test_samples)}")
    return train_samples, validation_samples, test_samples



def get_image_paths_and_labels(samples):
    paths = []
    corrected_samples = []
    for (i, file_line) in enumerate(samples):
        line_split = file_line.strip()
        line_split = line_split.split(" ")

        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path = os.path.join(base_image_path, partI, partI + "-" + partII, image_name + ".png")
        if os.path.getsize(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])
    return paths,corrected_samples

def get_lab():
    train_samples, validation_samples, test_samples = get_samples()
    train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
    validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples)
    test_img_paths, test_labels = get_image_paths_and_labels(test_samples)
    return train_img_paths,train_labels,validation_img_paths,validation_labels,test_img_paths,test_labels

def clean_train_lab(train_labels):
    train_labels_cleaned = []
    characters = set()
    max_len = 0 

    for label in train_labels:
        label = label.split(" ")[-1].strip()
        for char in label:
            characters.add(char)

        max_len = max(max_len, len(label))
        train_labels_cleaned.append(label)

    print("Maximum Length: ", max_len)
    print("Vocab Size: ", len(characters))
    return max_len, characters, train_labels_cleaned

def clean_labels(labels):
    cleaned_labels = []
    for label in labels:
        label = label.split(" ")[-1].strip()
        cleaned_labels.append(label)
    return cleaned_labels


def distortion_free_resize(image, img_size):
    w,h = img_size
    image = tf.image.resize(image, size=(h,w), preserve_aspect_ratio = True)

    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    if pad_height%2!=0:
        height = pad_height//2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2
    
    if pad_width%2!=0:
        width = pad_width//2
        pad_width_left = width+1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width//2
    
    image =tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0,0]
        ],

    )

    image = tf.transpose(image, perm=[1,0,2])
    image = tf.image.flip_left_right(image)
    return image



