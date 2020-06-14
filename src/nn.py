import json
import tensorflow as tf
from tensorflow import keras


class NN(object):
    def __init__(self, conf = ""):
        self.layers = {}
        self.blocks = {}

        if len(conf) > 0:
            with open(conf, "r") as conf_file:
                conf_data = json.loads(conf_file.read())

                if "layers" in conf_data:
                    self.layers = conf_data["layers"]

                if "blocks" in conf_data:
                    self.blocks = conf_data["blocks"]


    def mkLambdaLayer(self, fun_name, layer_name):
        if fun_name == 'std_image':
            return keras.layers.Lambda(std_image)

        assert False, "Not supported Lambda function: %s" % (fun_name)

    def getPenality(self, regularizer_conf):
        regularizer = None
        l2_penality = None
        l1_penality = None
        if 'l2' in regularizer_conf:
            l2_penality = regularizer_conf['l2']
        if 'l1' in regularizer_conf:
            l1_penality = regularizer_conf['l1']
        if l1_penality != None and l2_penality != None:
            regularizer = keras.regularizers.l1_l2(l1=l1_penality, l2=l2_penality)
        elif l1_penality != None:
            regularizer = keras.regularizers.l1(l=l1_penality)
        elif l2_penality != None:
            regularizer = keras.regularizers.l2(l=l2_penality)
        return regularizer

    def mkLayer(self, layer):
        strides = (1, 1)
        if 'strides' in layer:
            strides = layer['strides']
            if type(strides) is list:
                strides = tuple(strides)

        activation = None
        if 'activation' in layer:
            activation = layer['activation']

        name = None
        if 'name' in layer:
            name = layer['name']

        use_bias = True
        if 'use_bias' in layer:
            use_bias = layer['use_bias']

        layer_type = layer["layer"]
        if  layer_type == 'Dense':
            return keras.layers.Dense(int(layer['units']), activation=activation, use_bias=use_bias, name=name)
        elif layer_type == 'Reshape':
            return keras.layers.Reshape(tuple(layer['shape']), name=name)
        elif layer_type == 'Conv2DTranspose':
            return keras.layers.Conv2DTranspose(int(layer['filters']), layer['kernel'], strides=strides, padding=layer['padding'], activation=activation, use_bias=use_bias, name=name)
        elif layer_type == 'MaxPooling2D':
            strides = None
            if 'strides' in layer:
                strides = layer['strides']
            return keras.layers.MaxPooling2D(pool_size=tuple(layer['size']), strides=strides, name=name)
        elif layer_type == 'MaxPooling1D':
            strides = None
            if 'strides' in layer:
                strides = layer['strides']
            return keras.layers.MaxPooling1D(pool_size=layer['size'], strides=strides, name=name)
        elif layer_type == 'AveragePooling2D':
            strides = None
            if 'strides' in layer:
                strides = layer['strides']
            return keras.layers.AveragePooling2D(pool_size=tuple(layer['size']), strides=strides, name=name)
        elif layer_type == 'AveragePooling1D':
            strides = None
            if 'strides' in layer:
                strides = layer['strides']
            return keras.layers.AveragePooling1D(pool_size=layer['size'], strides=strides, name=name)
        elif layer_type == 'GlobalAveragePooling2D':
            return keras.layers.GlobalAveragePooling1D(name=name)
        elif layer_type == 'GlobalAveragePooling1D':
            return keras.layers.GlobalAveragePooling1D(name=name)
        elif layer_type == 'UpSampling2D':
            return keras.layers.UpSampling2D(size=tuple(layer['size']), name=name)
        elif layer_type == 'UpSampling1D':
            return keras.layers.UpSampling1D(size=layer['size'], name=name)
        elif layer_type == 'Conv1D':
            strides = 1
            if 'strides' in layer:
                strides = layer['strides']
            kernel_regularizer = None
            if 'kernel_regularizer' in layer:
                kernel_regularizer = self.getPenality(layer['kernel_regularizer'])
            activity_regularizer = None
            if 'activity_regularizer' in layer:
                activity_regularizer = slf.getPenality(layer['activity_regularizer'])
            return keras.layers.Conv1D(int(layer['filters']), layer['kernel'], strides=strides, padding=layer['padding'],
                                       kernel_regularizer=kernel_regularizer, activity_regularizer=activity_regularizer,
                                       activation=activation, use_bias=use_bias, name=name)
        elif layer_type == 'Conv2D':
            strides = 1
            if 'strides' in layer:
                strides = layer['strides']
            kernel_regularizer = None
            if 'kernel_regularizer' in layer:
                kernel_regularizer = self.getPenality(layer['kernel_regularizer'])
            activity_regularizer = None
            if 'activity_regularizer' in layer:
                activity_regularizer = slf.getPenality(layer['activity_regularizer'])
            return keras.layers.Conv2D(int(layer['filters']), layer['kernel'], strides=strides, padding=layer['padding'],
                                       kernel_regularizer=kernel_regularizer, activity_regularizer=activity_regularizer,
                                       activation=activation, use_bias=use_bias, name=name)
        elif layer_type == 'BatchNormalization':
            momentum = 0.99
            epsilon = 0.001
            axis = -1
            if 'momentum' in layer:
                momentum = float(layer['momentum'])
            if 'epsilon' in layer:
                epsilon = float(layer['epsilon'])
            if 'axis' in layer:
                axis = layer['axis']
            return keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon, axis=axis, name=name)
        elif layer_type == 'Flatten':
            return keras.layers.Flatten(name=name)
        elif layer_type == 'Dropout':
            return keras.layers.Dropout(float(layer['rate']), name=name)
        elif layer_type == 'Activation':
            return keras.layers.Activation(layer['activation'], name=name)
        elif layer_type == 'LeakyReLU':
            if 'alpha' in layer:
                return keras.layers.LeakyReLU(layer['alpha'])
            return keras.layers.LeakyReLU()
        elif layer_type == 'ZeroPadding2D':
            return keras.layers.ZeroPadding2D(padding=layer['padding'])
        elif layer_type == 'ZeroPadding1D':
            return keras.layers.ZeroPadding1D(padding=int(layer['padding']))
        elif layer_type == 'Add':
            return keras.layers.Add()
        elif layer_type == 'Lambda':
            return self.mkLambdaLayer(layer['function'], name)

        assert False, "Not supported layer type: %s" % (layer_type)


    def makeBlock(self, block, input_layer, parameters):
        block_params = block["parameters"]
        layers = block["layers"]
        block_shortcut = block["shortcut"]
        shortcut = output_layer = input_layer

        params = {}
        for i in range(len(block_params)):
            params[block_params[i]] = parameters[i]

        # block layers
        for layer in layers:
            layer_conf = {}
            # replace parameters
            for k, v in layer.items():
                if type(v) is str and v in params:
                    layer_conf[k] = params[v]
                else:
                    layer_conf[k] = v

            output_layer = self.mkLayer(layer_conf)(output_layer)

        # shortcut layers
        shortcut_layers = block_shortcut["layers"]
        if "shortcut_layer" in params and params["shortcut_layer"]:
            shortcut_layers = block_shortcut["shortcut_layer"]
        for layer in shortcut_layers:
            layer_conf = {}
            # replace parameters
            for k, v in layer.items():
                if type(v) is str and v in params:
                    layer_conf[k] = params[v]
                else:
                    layer_conf[k] = v

            shortcut = self.mkLayer(layer_conf)(shortcut)

        # merge
        output_layer = self.mkLayer(block_shortcut["merge"])([shortcut, output_layer])

        if "activation" in block_shortcut:
            output_layer = self.mkLayer(block_shortcut["activation"])(output_layer)

        return output_layer

    def makeNN(self, layers, input_layer):
        output_layer = input_layer
        for layer in layers:
            if layer["layer"] in self.blocks:
                output_layer = self.makeBlock(self.blocks[layer["layer"]], output_layer, layer["parameters"])
            else:
                output_layer = self.mkLayer(layer)(output_layer)
        return output_layer
    
    def build(self, input_layer):
        return self.makeNN(self.layers, input_layer)
