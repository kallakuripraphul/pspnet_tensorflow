import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras import layers
from keras.applications.resnet import ResNet50


def pool_block(features, stages):

    kernel_stride_dict = {1: 30, 2: 15, 3: 10, 6: 5}
    kernel = (kernel_stride_dict[stages], kernel_stride_dict[stages])
    strides = (kernel_stride_dict[stages], kernel_stride_dict[stages])

    x = AveragePooling2D(pool_size=(kernel), strides=strides)(
        features
    )  # , strides=strides
    x = Conv2D(512, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def resize_feature_map(x, features):
    resize_shape = features.get_shape()
    resized_map = tf.image.resize(
        x, [resize_shape[1], resize_shape[2]], method="bilinear"
    )
    return resized_map


def initial_cnn(inp):  # initial encoder

    enc1 = Conv2D(16, (3, 3), activation="relu")(inp)
    bn1 = BatchNormalization()(enc1)
    pool1 = MaxPooling2D()(bn1)

    enc2 = Conv2D(64, 3, activation="relu", padding="same")(pool1)
    bn2 = BatchNormalization()(enc2)
    pool2 = MaxPooling2D()(bn2)

    dec1 = UpSampling2D(size=(2, 2))(pool2)
    up1 = Conv2D(16, 3, activation="relu", padding="same")(dec1)
    bn3 = BatchNormalization()(up1)

    dec2 = UpSampling2D(size=(2, 2))(up1)
    up2 = Conv2D(3, 3, activation="relu", padding="same")(dec2)
    bn4 = BatchNormalization()(up2)
    return bn4


def pyramid_pooling_block(features):
    p1 = pool_block(features, 1)
    rm1 = resize_feature_map(p1, features)
    p2 = pool_block(features, 2)
    rm2 = resize_feature_map(p2, features)
    p3 = pool_block(features, 3)
    rm3 = resize_feature_map(p3, features)
    p4 = pool_block(features, 6)
    rm4 = resize_feature_map(p4, features)
    concat = Concatenate()([features, rm4, rm3, rm2, rm1])
    return concat


def get_resnet50(input, input_shape):
    resnet_op = ResNet50(
        include_top=False, weights=None, input_tensor=input, input_shape=input_shape
    ).output
    x = UpSampling2D(size=(2, 2))(resnet_op)
    x = UpSampling2D(size=(2, 2))(x)
    x = UpSampling2D(size=(2, 2))(x)  # feature map from resnet with reduced dimensions
    # print(x.shape)
    return x


if __name__ == "__main__":

    input_layer = Input(shape=[512, 512, 3])
    res_inp = initial_cnn(input_layer)
    print(res_inp.get_shape())
    shape_ = res_inp.get_shape()
    res_op = get_resnet50(res_inp, (shape_[1], shape_[2], shape_[3]))
    pyramid_block_op = pyramid_pooling_block(res_op)
    out = Conv2D(128, 3, activation="relu", padding="same")(pyramid_block_op)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)

    out = Conv2D(2, 1)(out)
    final = resize_feature_map(out, input_layer)
    final = Activation("sigmoid")(final)
    psp_model = Model(inputs=input_layer, outputs=final)
    psp_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    print(psp_model.summary())
