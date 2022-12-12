import tensorflow as tf
import os
import cv2


def parse_image(image, mask):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    mask = tf.io.read_file(mask)
    mask = tf.io.decode_jpeg(mask)

    return image, mask


if __name__ == "__main__":
    image_dir = "/Users/praphul/datasets/full_semantic_train/images/*"
    mask_dir = "/Users/praphul/datasets/full_semantic_train/labels/*"

    list_ds_images = tf.data.Dataset.list_files(image_dir)
    list_ds_masks = tf.data.Dataset.list_files(mask_dir)
    im_file = next(iter(list_ds_images))
    
    mask_file = next(iter(list_ds_masks))

    im, mask = parse_image(im_file, mask_file)
    print(mask.shape)
 
