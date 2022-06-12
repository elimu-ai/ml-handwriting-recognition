import tensorflow as tf
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from PIL import Image
import config

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
max_length = 7
mnist_digit_dim=28
min_overlap = 0.0
max_overlap = 0.5

def add_left_and_right_paddings(number) :
    """Add paddings to left and right of the number."""
    digits = [int(digit) for digit in list(str(number))]
    remanining_length = max_length - len(digits)
    left_padding = random.randint(0, remanining_length)
    right_padding = remanining_length - left_padding
    digits = [-1] * left_padding + digits + [-1] * right_padding
    return digits

def _get_samples_by_digit() -> Dict[int, List]:
    """Stores a collection of images for each digit."""
    samples_by_digit = defaultdict(list)
    for image, digit in (zip(x_train, y_train)):
        samples_by_digit[digit].append(image.squeeze())
    blank_image = tf.zeros((mnist_digit_dim, mnist_digit_dim))
    samples_by_digit[-1].append(blank_image)
    return samples_by_digit

samples_by_digit = _get_samples_by_digit()

def construct_image_from_number(number: int) -> tf.Tensor:
    """Concatenate images of single digits."""
    overlap = random.uniform(min_overlap, max_overlap)
    overlap_width = int(overlap * mnist_digit_dim)
    width_increment = mnist_digit_dim - overlap_width
    x, y = 0, 2  # Current pointers at x and y coordinates
    digits = add_left_and_right_paddings(number)
    multi_digit_image = tf.zeros((32, mnist_digit_dim * max_length)).numpy()
    for digit in digits:
        digit_image = random.choice(samples_by_digit[digit])
        digit_image = tf.identity(
            digit_image
        ).numpy()  # To avoid overwriting the original image
        digit_image[:, :overlap_width] = tf.maximum(
            multi_digit_image[y : y + mnist_digit_dim, x : x + overlap_width],
            digit_image[:, :overlap_width],
        )
        multi_digit_image[
            y : y + mnist_digit_dim, x : x + mnist_digit_dim
        ] = digit_image
        x += width_increment
    return multi_digit_image

def get_random_number() -> int:
        """Generate a random number.
        The probabiltiy of getting a small number is artifically inflated; otherwise,
        there will be not enough numbers of short lengths and the model will not
        generalize well.
        """
        num_digits_choices = list(range(1, max_length + 1))
        probs = [n / sum(num_digits_choices) for n in num_digits_choices]
        num_digits = random.choices(num_digits_choices, weights=probs)[0]
        rand_num = random.randint(
            int("1" + "0" * (num_digits - 1)), int("1" + "0" * num_digits) - 1
        )
        return rand_num


def gen_dataset(size):
    images = []
    labels = []
    for _ in range(0,size):
        num = get_random_number()
        img = construct_image_from_number(num)
        images.append(img)
        labels.append(num)
    return np.array(images),np.array(labels)

def get_image_paths_and_labels(samples):
    img_paths = []
    for i in samples:
        pa = config.BASE_PATH+"/"+str(i)+"_"+str(random.randint(0,9999))+".png"
        img = construct_image_from_number(i)
        #img = img[np.newaxis, ...]
        img = Image.fromarray(img)
        img = img.convert("L")
        img.save(pa, "PNG")
        img_paths.append(pa)
    return img_paths,samples

def get_lab():
    train_samples, validation_samples, test_samples = get_samples()
    train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
    validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples)
    test_img_paths, test_labels = get_image_paths_and_labels(test_samples)
    return train_img_paths,train_labels,validation_img_paths,validation_labels,test_img_paths,test_labels

def clean_labels(labels):
    cleaned_labels = []
    for label in labels:
        label = str(label)
        cleaned_labels.append(label)
    return cleaned_labels

def clean_train_lab(train_labels):
    train_labels_cleaned = []
    characters = set()
    max_len = 0 

    for label in train_labels:
        label = str(label)
        for char in label:
            characters.add(char)

        max_len = max(max_len, len(label))
        train_labels_cleaned.append(label)

    print("Maximum Length: ", max_len)
    print("Vocab Size: ", len(characters))
    print("Vocab: ",characters)
    return max_len, characters, train_labels_cleaned
    
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

def num_gen():
    a = 0
    b = 20000
    k = 0.432
    return int(math.floor(a + (b - a + 1) * (1.0 - random.random()**(1.0 / k))))

def get_samples():
    labels = [num_gen() for _ in range(50000)]
    print("Data Prepping")
    np.random.shuffle(labels)
    print("Data Shuffled")
    split_idx = int(0.8*len(labels))
    train_samples = labels[:split_idx]
    test_samples = labels[split_idx:]

    val_split_idx = int(0.5*len(test_samples))
    validation_samples = test_samples[:val_split_idx]
    test_samples = test_samples[val_split_idx:]

    assert len(labels) == len(train_samples) + len(validation_samples) + len(test_samples)
    print("Data Generation completed")
    print(f"Total training Samples:{len(train_samples)}")
    print(f"Toal validation samples: {len(validation_samples)}")
    print(f"Total test samples: {len(test_samples)}")
    return train_samples, validation_samples, test_samples