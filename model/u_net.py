from keras.models import Model, Sequential
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, Flatten, Dense
from keras.optimizers import RMSprop

from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff

from keras.layers.core import Layer
import tensorflow as tf
import numpy as np

def get_unet_1024(input_shape=(1024, 1024, 3),
                  num_classes=1):
    inputs = Input(shape=input_shape)
    # 1024

    down0b = Conv2D(8, (3, 3), padding='same')(inputs)
    down0b = BatchNormalization()(down0b)
    down0b = Activation('relu')(down0b)
    down0b = Conv2D(8, (3, 3), padding='same')(down0b)
    down0b = BatchNormalization()(down0b)
    down0b = Activation('relu')(down0b)
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    # 512

    down0a = Conv2D(16, (3, 3), padding='same')(down0b_pool)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    # 512

    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    # 1024

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0b)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model

def get_unet_deep(input_shape=(1024, 1024, 3),
                  num_classes=1):
    inputs = Input(shape=input_shape)
    # 1024

    down0 = Conv2D(64, (3, 3), padding='same')(inputs)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(64, (3, 3), padding='same')(inputs)
    down0 = Activation('relu')(down0)
    plus_0 = Conv2D(32, (2, 2), padding='same')(down0)
    down0 = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    down0 = Activation('relu')(down0)

    # 512

    down1 = Conv2D(64, (3, 3), padding='same')(down0)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(32, (3, 3), padding='same')(down1)
    plus_1 = concatenate([down1, down0], axis=3)
    down1= MaxPooling2D((2, 2), strides=(2, 2))(plus_1)
    down1 = Activation('relu')(down1)

    # 256

    down2 = Conv2D(64, (3, 3), padding='same')(down1)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(32, (2, 2), padding='same')(down2)
    plus_2 = concatenate([down2, down1], axis=3)
    down2 = MaxPooling2D((2, 2), strides=(2, 2))(plus_2)
    down2 = Activation('relu')(down2)

    # 128

    down3 = Conv2D(64, (3, 3), padding='same')(down2)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(32, (2, 2), padding='same')(down3)
    plus_3 = concatenate([down3, down2], axis=3)
    down3 = MaxPooling2D((2, 2), strides=(2, 2))(plus_3)
    down3 = Activation('relu')(down3)

    # 64

    down4 = Conv2D(64, (3, 3), padding='same')(down3)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(32, (2, 2), padding='same')(down4)
    plus_4 = concatenate([down4, down3], axis=3)
    down4 = MaxPooling2D((2, 2), strides=(2, 2))(plus_4)
    down4 = Activation('relu')(down4)

    # 32

    down5 = Conv2D(64, (3, 3), padding='same')(down4)
    down5 = Activation('relu')(down5)
    down5 = Conv2D(32, (2, 2), padding='same')(down5)
    plus_5 = concatenate([down5, down4], axis=3)
    down5 = MaxPooling2D((2, 2), strides=(2, 2))(plus_5)
    down5 = Activation('relu')(down5)

    # 16

    down6 = Conv2D(64, (3, 3), padding='same')(down5)
    down6 = Activation('relu')(down6)
    down6 = Conv2D(32, (2, 2), padding='same')(down6)
    plus_6 = concatenate([down6, down5], axis=3)
    down6 = MaxPooling2D((2, 2), strides=(2, 2))(plus_6)
    down6 = Activation('relu')(down6)

    # 8

    up7 = UpSampling2D((2, 2))(down6)

    # 16

    up6 = concatenate([up7, plus_6], axis=3)
    up6 = Conv2D(64, (3, 3), padding='same')(up6)
    up6 = Activation('relu')(up6)
    up6 = Conv2D(32, (3, 3), padding='same')(up6)
    up6 = concatenate([up6, up7], axis=3)
    up6 = Activation('relu')(up6)
    up6 = UpSampling2D((2, 2))(up6)

    # 32

    up5 = concatenate([up6, plus_5], axis=3)
    up5 = Conv2D(64, (3, 3), padding='same')(up5)
    up5 = Activation('relu')(up5)
    up5 = Conv2D(32, (3, 3), padding='same')(up5)
    up5 = concatenate([up5, up6], axis=3)
    up5 = Activation('relu')(up5)
    up5 = UpSampling2D((2, 2))(up5)

    # 64

    up4 = concatenate([up5, plus_4], axis=3)
    up4 = Conv2D(64, (3, 3), padding='same')(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(32, (3, 3), padding='same')(up4)
    up4 = concatenate([up4, up5], axis=3)
    up4 = Activation('relu')(up4)
    up4 = UpSampling2D((2, 2))(up4)

    # 128

    up3 = concatenate([up4, plus_3], axis=3)
    up3 = Conv2D(64, (3, 3), padding='same')(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(32, (3, 3), padding='same')(up3)
    up3 = concatenate([up3, up4], axis=3)
    up3 = Activation('relu')(up3)
    up3 = UpSampling2D((2, 2))(up3)

    # 256

    up2 = concatenate([up3, plus_2], axis=3)
    up2 = Conv2D(64, (3, 3), padding='same')(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(32, (3, 3), padding='same')(up2)
    up2 = concatenate([up2, up3], axis=3)
    up2 = Activation('relu')(up2)
    up2 = UpSampling2D((2, 2))(up2)

    # 512

    up1 = concatenate([up2, plus_1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(32, (3, 3), padding='same')(up1)
    up1 = concatenate([up1, up2], axis=3)
    up1 = Activation('relu')(up1)
    up1 = UpSampling2D((2, 2))(up1)

    # 1024

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=RMSprop(lr=0.0002), loss=bce_dice_loss, metrics=[dice_coeff])

    return model
