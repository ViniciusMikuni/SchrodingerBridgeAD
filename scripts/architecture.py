import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def Resnet(
        input_dim,
        end_dim,
        time_embedding,
        num_embed,
        num_layer = 3,
        mlp_dim=128,
):

    
    act = layers.LeakyReLU(alpha=0.01)
    #act = swish

    def resnet_dense(input_layer,hidden_size):
        layer,time = input_layer
        residual = layers.Dense(hidden_size)(layer)
        embed =  layers.Dense(hidden_size)(time)
        x = act(residual)
        x = layers.Dense(hidden_size)(x)
        x = act(layers.Add()([x, embed]))
        x = layers.Dense(hidden_size)(x)
        x = layers.Add()([x, residual])
        return x

    inputs = keras.Input((input_dim))
    embed = act(layers.Dense(mlp_dim)(time_embedding))
    
    layer = layers.Dense(mlp_dim)(inputs)
    for _ in range(num_layer-1):
        layer =  resnet_dense([layer,embed],mlp_dim)

    outputs = layers.Dense(end_dim,kernel_initializer="zeros")(layer)
    
    return inputs,outputs
