import numpy as np
import tensorflow as tf

from keras import initializers
from keras import regularizers
from keras import constraints

from utils.batch_normalization import Reds_BatchNormalizationBase
from utils.convolution import Linear_Adaptive, Reds_2DConvolution_Standard
from utils.logs import print_intermediate_activations


def get_reds_ds_cnn_architecture(model_settings,
                                 classes=12, debug=False, use_bias=True,
                                 subnetworks_number=4, model_filters=64, model_size="l"):
    if model_size == "l":
        return Reds_Ds_Cnn_Wake_Model_L(classes=classes, use_bias=use_bias,
                                          subnetworks_number=subnetworks_number,
                                          model_filters=model_filters,
                                          stride_w_first_convolution=1,
                                          strides_first_depth_wise=2,
                                          model_settings=model_settings, debug=debug)
    else:
        if model_size == 'm':
            return Reds_Ds_Cnn_Wake_Model(classes=classes, use_bias=use_bias,
                                          subnetworks_number=subnetworks_number,
                                          model_filters=model_filters,
                                          stride_w_first_convolution=1,
                                          strides_first_depth_wise=2,
                                          model_settings=model_settings, debug=debug)
        elif model_size == 's':
            return Reds_Ds_Cnn_Wake_Model(classes=classes, use_bias=use_bias,
                                          model_filters=model_filters,
                                          subnetworks_number=subnetworks_number,
                                          model_settings=model_settings, debug=debug)


class Reds_DepthwiseConv2D(tf.keras.layers.Layer):

    def __init__(
            self,
            in_channels,
            kernel_size,
            batch_dimensions=4,
            strides=(1, 1),
            padding="valid",
            depth_multiplier=1,
            data_format=None,
            dilation_rate=(1, 1),
            activation=None,
            use_bias=True,
            debug=False,
            depthwise_initializer="glorot_uniform",
            bias_initializer="zeros",
            depthwise_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            depthwise_constraint=None,
            bias_constraint=None,
            **kwargs
    ):
        super(Reds_DepthwiseConv2D, self).__init__()
        self.use_bias = use_bias
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.strides = strides
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        depthwise_kernel_shape = self.kernel_size + (
            in_channels,
            self.depth_multiplier,
        )

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name="depthwise_kernel",
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(in_channels * self.depth_multiplier,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        self.debug = debug
        self.batch_dimensions = batch_dimensions
        self.filters = in_channels * self.depth_multiplier
        self.filters_splittings = tf.Variable(
            initial_value=tf.constant([self.filters], dtype=tf.int32),
            trainable=False)
        self.built = True
        self.registered_subnetworks_kernels_number = False

    def add_splitting_filters_indexes(self, filters_indexes):
        self.filters_splittings = tf.concat([self.filters_splittings, tf.constant([filters_indexes], dtype=tf.int32)],
                                            axis=0)

    def get_trainable_parameters_number(self, subnetwork_index):
        """
        Get the number of trainable filters and bias parameters of the subnetwork
        @param subnetwork_index: index of the subnetowork to obtain the number of trainable parameters
        """

        return np.prod(self.depthwise_kernel[:, :, 0:int(self.filters_splittings[subnetwork_index]),
                       :].shape) + np.prod(
            self.bias[:int(self.filters_splittings[subnetwork_index])].shape)

    def compute_layer_lookup_table(self, inputs):
        """
        Compute the number of MACs for each unit considered inside the layer and the memory bytes required to store it
        @param inputs: the input tensor to the layer
        @return: the lookup table for the layer and the transformed input
        """

        layer_units = self.depthwise_kernel.shape[2]

        activation_map = tf.nn.depthwise_conv2d(
            inputs,
            self.depthwise_kernel[:, :, :, :],
            strides=(1,) + self.strides + (1,),
            padding=self.padding.upper()
        )

        if self.use_bias:
            activation_map = tf.nn.bias_add(activation_map, self.bias)

        kernel_height, kernel_width, channels, _ = self.depthwise_kernel.shape
        output_channels, output_height, output_width, _ = activation_map.shape
        layer_macs = output_channels * channels * kernel_height * kernel_width * output_height * output_width

        filters_parameters_number = tf.size(self.depthwise_kernel).numpy() / layer_units
        element_size = self.depthwise_kernel.dtype.size
        filters_byte_memory = np.full(layer_units, filters_parameters_number * element_size, dtype=float)

        units_macs = np.full(layer_units, layer_macs / layer_units, dtype=float)

        return activation_map, units_macs, filters_byte_memory

    def call(self, inputs, **kwargs):

        for subnet_number in range(len(inputs)):

            inputs[subnet_number] = tf.nn.depthwise_conv2d(
                inputs[subnet_number],
                self.depthwise_kernel[:, :, 0:int(self.filters_splittings[subnet_number]), :],
                strides=(1,) + self.strides + (1,),
                padding=self.padding.upper()
            )

            if self.use_bias:
                inputs[subnet_number] = tf.nn.bias_add(
                    inputs[subnet_number], self.bias[:int(self.filters_splittings[subnet_number])]
                )

        return inputs


def set_trainable_pointwise_batch_norm(model, trainable_pointwise_batch_norm=True):
    first_layer = True

    layer_index = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization) and isinstance(model.layers[layer_index - 1],
                                                                                tf.keras.layers.Conv2D):
            if first_layer:
                first_layer = False
                continue
            layer.trainable = trainable_pointwise_batch_norm


def get_pointwise_convolutions_layers_weights(model):
    pointwise_layers_weights = []
    pointwise_layers_biases = []
    first_layer = True

    for layer in model.layers:

        if isinstance(layer, tf.keras.layers.Conv2D):
            if first_layer:
                first_layer = False
                continue
            pointwise_layers_weights.append(layer.weights[0])
            pointwise_layers_biases.append(layer.weights[1])

    return pointwise_layers_weights, pointwise_layers_biases


class Reds_Ds_Cnn_Wake_Model_L(tf.keras.Model):
    def __init__(self, classes, subnetworks_number, model_settings,
                 batch_dimensions=4, pool_size=(13, 5), strides_h_first_convolution=2, stride_w_first_convolution=1,
                 strides_first_depth_wise=2,
                 model_filters=64,
                 use_bias=True,
                 debug=False):

        super(Reds_Ds_Cnn_Wake_Model_L, self).__init__()
        self.model_settings = model_settings

        self.reshape = tf.keras.layers.Reshape(
            (self.model_settings['spectrogram_length'], self.model_settings['dct_coefficient_count'], 1),
            input_shape=(self.model_settings['fingerprint_size'], 1))

        self.standard_convolution = Reds_2DConvolution_Standard(in_features=1, out_features=model_filters,
                                                                kernel_size=(10, 4),
                                                                batch_dimensions=batch_dimensions,
                                                                use_bias=use_bias,
                                                                strides=(strides_h_first_convolution,
                                                                         stride_w_first_convolution),
                                                                debug=debug, padding='same')
        self.batch_norm1 = Reds_BatchNormalizationBase(fused=False)
        self.relu1 = tf.keras.layers.ReLU()
        self.depthwise1 = Reds_DepthwiseConv2D(in_channels=model_filters, kernel_size=(3, 3),
                                               use_bias=use_bias,
                                               strides=(strides_first_depth_wise, strides_first_depth_wise),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug, padding='same')
        self.batch_norm2 = Reds_BatchNormalizationBase(fused=False)
        self.relu2 = tf.keras.layers.ReLU()
        self.pointwise_conv1 = Reds_2DConvolution_Standard(in_features=model_filters, out_features=model_filters,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='valid')
        self.batch_norm3 = Reds_BatchNormalizationBase(fused=False)
        self.relu3 = tf.keras.layers.ReLU()
        self.depthwise2 = Reds_DepthwiseConv2D(in_channels=model_filters, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug, padding='same')
        self.batch_norm4 = Reds_BatchNormalizationBase(fused=False)
        self.relu4 = tf.keras.layers.ReLU()
        self.pointwise_conv2 = Reds_2DConvolution_Standard(in_features=model_filters, out_features=model_filters,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='valid')
        self.batch_norm5 = Reds_BatchNormalizationBase(fused=False)
        self.relu5 = tf.keras.layers.ReLU()
        self.depthwise3 = Reds_DepthwiseConv2D(in_channels=model_filters, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug, padding='same')
        self.batch_norm6 = Reds_BatchNormalizationBase(fused=False)
        self.relu6 = tf.keras.layers.ReLU()
        self.pointwise_conv3 = Reds_2DConvolution_Standard(in_features=model_filters, out_features=model_filters,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='valid')
        self.batch_norm7 = Reds_BatchNormalizationBase(fused=False)
        self.relu7 = tf.keras.layers.ReLU()
        self.depthwise4 = Reds_DepthwiseConv2D(in_channels=model_filters, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug, padding='same')
        self.batch_norm8 = Reds_BatchNormalizationBase(fused=False)
        self.relu8 = tf.keras.layers.ReLU()
        self.pointwise_conv4 = Reds_2DConvolution_Standard(in_features=model_filters, out_features=model_filters,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='valid')
        self.batch_norm9 = Reds_BatchNormalizationBase(fused=False)
        self.relu9 = tf.keras.layers.ReLU()

        self.depthwise5 = Reds_DepthwiseConv2D(in_channels=model_filters, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug, padding='same')
        self.batch_norm10 = Reds_BatchNormalizationBase(fused=False)
        self.relu10 = tf.keras.layers.ReLU()
        self.pointwise_conv5 = Reds_2DConvolution_Standard(in_features=model_filters, out_features=model_filters,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='valid')
        self.batch_norm11 = Reds_BatchNormalizationBase(fused=False)
        self.relu11 = tf.keras.layers.ReLU()

        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=(pool_size[0], pool_size[1]),
                                                  strides=(pool_size[0], pool_size[1]), padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = Linear_Adaptive(in_features=model_filters, out_features=classes,
                                   debug=debug, use_bias=use_bias)

        self.subnetworks_number = subnetworks_number
        self.print_hidden_feature = False
        self.debug = debug
        self.classes = classes
        self.use_bias = use_bias
        self.dense_model_trainable_weights = 0

    def trainable_layers(self, trainable_parameters=True, trainable_batch_normalization=False):
        for layer in self.layers:

            if isinstance(layer, Reds_2DConvolution_Standard) or isinstance(layer, Reds_DepthwiseConv2D) or isinstance(
                    layer, Linear_Adaptive):

                layer.trainable = trainable_parameters
            elif isinstance(layer, Reds_BatchNormalizationBase):
                layer.trainable = trainable_batch_normalization
            else:
                pass

    def get_subnetwork_parameters_percentage(self, subnetwork_index):
        """
        If the input feature are not registered perform a forward pass on the adaptive layers to compute them, return the
        number of trainable parameters used by the subnetwork
        @param subnetwork_index:
        @return:
        """

        if self.dense_model_trainable_weights == 0:
            self.dense_model_trainable_weights = sum(
                [np.prod(tensor.shape) if not isinstance(tensor, Reds_BatchNormalizationBase) else 0 for tensor in
                 self.trainable_weights])

        subnetwork_trainable_parameters = 0
        for layer in self.layers:

            if isinstance(layer, Reds_BatchNormalizationBase) or isinstance(layer,
                                                                            tf.keras.layers.Reshape) or isinstance(
                layer, tf.keras.layers.ReLU) or isinstance(layer, tf.keras.layers.Flatten) or isinstance(layer,
                                                                                                         tf.keras.layers.AveragePooling2D) or isinstance(
                layer, Linear_Adaptive):
                continue
            else:
                subnetwork_trainable_parameters += layer.get_trainable_parameters_number(
                    subnetwork_index=subnetwork_index)

        # iterate over model layers and count the weights of each layer specific for the subnetwork
        return subnetwork_trainable_parameters / self.dense_model_trainable_weights

    def set_subnetworks_number(self, subnetworks_number):
        self.subnetworks_number = subnetworks_number

    def finetune_batch_normalization(self):
        """
        Used to finetune the Batch Normalization layers while freezing all
        the other layers
        """
        for layer in self.layers:
            if isinstance(layer, Reds_BatchNormalizationBase):
                layer.trainable = True
            else:
                layer.trainable = False

    def set_subnetwork_indexes(self, subnetworks_filters_first_convolution, subnetworks_filters_depthwise,
                               subnetworks_filters_pointwise):

        for subnetwork_index in range(self.subnetworks_number - 1):

            # add filters in standard convolution layer
            self.standard_convolution.add_splitting_filters_indexes(
                subnetworks_filters_first_convolution[subnetwork_index][0] + 1)

            # add filters in depthwise and pointwise convolution layers
            pointwise_filters_subnetwork = subnetworks_filters_pointwise[subnetwork_index]# [0]
            depthwise_filters_subnetwork = subnetworks_filters_depthwise[subnetwork_index]# [0]

            block_index_depthwise = 0
            for layer in self.layers:
                if isinstance(layer, Reds_DepthwiseConv2D):
                    layer.add_splitting_filters_indexes(depthwise_filters_subnetwork[block_index_depthwise] + 1)
                    block_index_depthwise += 1

            first_layer = True

            block_index_pointwise = 0
            for layer in self.layers:
                if isinstance(layer, Reds_2DConvolution_Standard):

                    if first_layer:
                        first_layer = False
                        continue

                    layer.add_splitting_filters_indexes(pointwise_filters_subnetwork[block_index_pointwise] + 1)
                    block_index_pointwise += 1

    def set_print_hidden_feature(self, print=True):
        self.print_hidden_feature = print
        self.debug = print

    def set_debug(self, debug=True):
        self.debug = debug

    def get_model_name(self):
        return "Wake_Word_DS_CNN_L"

    def compute_lookup_table(self, train_data, average_run=100):
        """
        Compute the lookup table for the model given the training data set. A batch of the data is passed as input
        inside the model and the linear operation associated to the layer is performed for average_run times. Then the
        MACs for each layers's filters are computed and the memory size of each filter is computed too.
        @return: List[List] containing the filters forward times for each layer
        List[List] containing the filters macs for each layer
        """
        inputs = None
        for images, _ in train_data.take(1):
            inputs = images
            break

        layers_filters_macs = []
        layers_filters_byte = []

        inputs = self.reshape(inputs)

        for layer in self.layers:

            if isinstance(layer, Reds_DepthwiseConv2D) or isinstance(layer, Reds_2DConvolution_Standard):
                inputs, macs, filters_byte_memory = layer.compute_layer_lookup_table(
                    inputs=inputs)
                layers_filters_macs.append(macs)
                layers_filters_byte.append(filters_byte_memory)

        return layers_filters_macs, layers_filters_byte

    def build(self, input_shape):
        """
        Initialize model's layers weights and biases
        """

        init_input = tf.ones(
            input_shape,
            dtype=tf.dtypes.float32,
        )

        init_input = self.reshape(init_input)
        init_input = [init_input for _ in range(1)]

        for layer in self.layers[1:]:

            if isinstance(layer, Reds_BatchNormalizationBase):
                _ = layer.build(init_input[0].shape)
            elif isinstance(layer, Reds_2DConvolution_Standard):
                init_input, times = layer(init_input)
            elif isinstance(layer, Reds_DepthwiseConv2D):
                init_input = layer(init_input)
            elif isinstance(layer, Linear_Adaptive):
                init_input[0] = self.avg_pool(init_input[0])
                init_input[0] = self.flatten(init_input[0])
                _ = layer(init_input)

        self.built = True

    def call(self, inputs, training=None, mask=None):
        inputs = self.reshape(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Initial Input shape after Reshape") if self.debug else None

        # copy the input once for each subnetwork
        inputs = [inputs for _ in range(self.subnetworks_number)]

        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Initial Inputs shape") if self.debug else None

        inputs, times = self.standard_convolution(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Initial Standard Convolution") if self.debug else None

        inputs = self.batch_norm1(inputs)
        inputs = [self.relu1(input) for input in inputs]

        inputs = self.depthwise1(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Depthwise Convolution Layer 1") if self.debug else None
        inputs = self.batch_norm2(inputs)
        inputs = [self.relu2(input) for input in inputs]

        inputs, times = self.pointwise_conv1(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Pointwise Convolution Layer 1") if self.debug else None
        inputs = self.batch_norm3(inputs)
        inputs = [self.relu3(input) for input in inputs]

        inputs = self.depthwise2(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Depthwise Convolution Layer 2") if self.debug else None
        inputs = self.batch_norm4(inputs)
        inputs = [self.relu4(input) for input in inputs]

        inputs, times = self.pointwise_conv2(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Pointwise Convolution Layer 2") if self.debug else None
        inputs = self.batch_norm5(inputs)
        inputs = [self.relu5(input) for input in inputs]

        inputs = self.depthwise3(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Depthwise Convolution Layer 3") if self.debug else None
        inputs = self.batch_norm6(inputs)
        inputs = [self.relu6(input) for input in inputs]

        inputs, times = self.pointwise_conv3(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Pointwise Convolution Layer 3") if self.debug else None
        inputs = self.batch_norm7(inputs)
        inputs = [self.relu7(input) for input in inputs]

        inputs = self.depthwise4(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Depthwise Convolution Layer 4") if self.debug else None
        inputs = self.batch_norm8(inputs)
        inputs = [self.relu8(input) for input in inputs]

        inputs, times = self.pointwise_conv4(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Pointwise Convolution Layer 4") if self.debug else None
        inputs = self.batch_norm9(inputs)
        inputs = [self.relu9(input) for input in inputs]

        inputs = self.depthwise5(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Depthwise Convolution Layer 5") if self.debug else None
        inputs = self.batch_norm10(inputs)
        inputs = [self.relu10(input) for input in inputs]

        inputs, times = self.pointwise_conv5(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Pointwise Convolution Layer 5") if self.debug else None
        inputs = self.batch_norm11(inputs)
        inputs = [self.relu11(input) for input in inputs]

        inputs = [
            tf.nn.avg_pool(input, ksize=[1, input.shape[1], input.shape[2], 1], strides=[1, 1, 1, 1], padding="VALID")
            for input in inputs]

        inputs = [tf.keras.layers.Flatten()(input) for input in inputs]
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="AFTER Flatten") if self.debug else None
        inputs, times = self.fc1(inputs)

        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="LAST output vector") if self.debug else None

        return inputs


class Reds_Ds_Cnn_Wake_Model(tf.keras.Model):

    def __init__(self, classes, subnetworks_number, model_settings,
                 batch_dimensions=4, pool_size=(25, 5), strides_h_first_convolution=2, stride_w_first_convolution=2,
                 strides_first_depth_wise=1,
                 model_filters=64,
                 use_bias=True,
                 debug=False):

        super(Reds_Ds_Cnn_Wake_Model, self).__init__()
        self.model_settings = model_settings

        self.reshape = tf.keras.layers.Reshape(
            (self.model_settings['spectrogram_length'], self.model_settings['dct_coefficient_count'], 1),
            input_shape=(self.model_settings['fingerprint_size'], 1))
        self.standard_convolution = Reds_2DConvolution_Standard(in_features=1, out_features=model_filters,
                                                                kernel_size=(10, 4),
                                                                batch_dimensions=batch_dimensions,
                                                                use_bias=use_bias,
                                                                strides=(strides_h_first_convolution,
                                                                         stride_w_first_convolution),
                                                                debug=debug, padding='same')
        self.batch_norm1 = Reds_BatchNormalizationBase(fused=False)
        self.relu1 = tf.keras.layers.ReLU()
        self.depthwise1 = Reds_DepthwiseConv2D(in_channels=model_filters, kernel_size=(3, 3),
                                               use_bias=use_bias,
                                               strides=(strides_first_depth_wise, strides_first_depth_wise),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug, padding='same')
        self.batch_norm2 = Reds_BatchNormalizationBase(fused=False)
        self.relu2 = tf.keras.layers.ReLU()
        self.pointwise_conv1 = Reds_2DConvolution_Standard(in_features=model_filters, out_features=model_filters,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='valid')
        self.batch_norm3 = Reds_BatchNormalizationBase(fused=False)
        self.relu3 = tf.keras.layers.ReLU()
        self.depthwise2 = Reds_DepthwiseConv2D(in_channels=model_filters, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug, padding='same')
        self.batch_norm4 = Reds_BatchNormalizationBase(fused=False)
        self.relu4 = tf.keras.layers.ReLU()
        self.pointwise_conv2 = Reds_2DConvolution_Standard(in_features=model_filters, out_features=model_filters,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='valid')
        self.batch_norm5 = Reds_BatchNormalizationBase(fused=False)
        self.relu5 = tf.keras.layers.ReLU()
        self.depthwise3 = Reds_DepthwiseConv2D(in_channels=model_filters, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug, padding='same')
        self.batch_norm6 = Reds_BatchNormalizationBase(fused=False)
        self.relu6 = tf.keras.layers.ReLU()
        self.pointwise_conv3 = Reds_2DConvolution_Standard(in_features=model_filters, out_features=model_filters,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='valid')
        self.batch_norm7 = Reds_BatchNormalizationBase(fused=False)
        self.relu7 = tf.keras.layers.ReLU()
        self.depthwise4 = Reds_DepthwiseConv2D(in_channels=model_filters, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug, padding='same')
        self.batch_norm8 = Reds_BatchNormalizationBase(fused=False)
        self.relu8 = tf.keras.layers.ReLU()
        self.pointwise_conv4 = Reds_2DConvolution_Standard(in_features=model_filters, out_features=model_filters,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='valid')
        self.batch_norm9 = Reds_BatchNormalizationBase(fused=False)
        self.relu9 = tf.keras.layers.ReLU()
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=(pool_size[0], pool_size[1]),
                                                  strides=(pool_size[0], pool_size[1]), padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = Linear_Adaptive(in_features=model_filters, out_features=classes,
                                   debug=debug, use_bias=use_bias)

        self.subnetworks_number = subnetworks_number
        self.print_hidden_feature = False
        self.debug = debug
        self.classes = classes
        self.use_bias = use_bias
        self.dense_model_trainable_weights = 0

    def trainable_layers(self, trainable_parameters=True, trainable_batch_normalization=False):
        for layer in self.layers:

            if isinstance(layer, Reds_2DConvolution_Standard) or isinstance(layer, Reds_DepthwiseConv2D) or isinstance(
                    layer, Linear_Adaptive):

                layer.trainable = trainable_parameters
            elif isinstance(layer, Reds_BatchNormalizationBase):
                layer.trainable = trainable_batch_normalization
            else:
                pass

    def get_subnetwork_parameters_percentage(self, subnetwork_index):
        """
        If the input feature are not registered perform a forward pass on the adaptive layers to compute them, return the
        number of trainable parameters used by the subnetwork
        @param subnetwork_index:
        @return:
        """

        if self.dense_model_trainable_weights == 0:
            self.dense_model_trainable_weights = sum(
                [np.prod(tensor.shape) if not isinstance(tensor, Reds_BatchNormalizationBase) else 0 for tensor in
                 self.trainable_weights])

        subnetwork_trainable_parameters = 0
        for layer in self.layers:

            if isinstance(layer, Reds_BatchNormalizationBase) or isinstance(layer,
                                                                            tf.keras.layers.Reshape) or isinstance(
                layer, tf.keras.layers.ReLU) or isinstance(layer, tf.keras.layers.Flatten) or isinstance(layer,
                                                                                                         tf.keras.layers.AveragePooling2D) or isinstance(
                layer, Linear_Adaptive):
                continue
            else:
                subnetwork_trainable_parameters += layer.get_trainable_parameters_number(
                    subnetwork_index=subnetwork_index)

        # iterate over model layers and count the weights of each layer specific for the subnetwork
        return subnetwork_trainable_parameters / self.dense_model_trainable_weights

    def set_subnetworks_number(self, subnetworks_number):
        self.subnetworks_number = subnetworks_number

    def finetune_batch_normalization(self):
        """
        Used to finetune the Batch Normalization layers while freezing all
        the other layers
        """
        for layer in self.layers:
            if isinstance(layer, Reds_BatchNormalizationBase):
                layer.trainable = True
            else:
                layer.trainable = False

    def set_subnetwork_indexes(self, subnetworks_filters_first_convolution, subnetworks_filters_depthwise,
                               subnetworks_filters_pointwise):

        for subnetwork_index in range(self.subnetworks_number - 1):

            # add filters in standard convolution layer
            self.standard_convolution.add_splitting_filters_indexes(
                subnetworks_filters_first_convolution[subnetwork_index][0] + 1)

            # add filters in depthwise and pointwise convolution layers
            pointwise_filters_subnetwork = subnetworks_filters_pointwise[subnetwork_index] #[0]
            depthwise_filters_subnetwork = subnetworks_filters_depthwise[subnetwork_index] # [0]

            block_index_depthwise = 0
            for layer in self.layers:
                if isinstance(layer, Reds_DepthwiseConv2D):
                    layer.add_splitting_filters_indexes(depthwise_filters_subnetwork[block_index_depthwise] + 1)
                    block_index_depthwise += 1

            first_layer = True

            block_index_pointwise = 0
            for layer in self.layers:
                if isinstance(layer, Reds_2DConvolution_Standard):

                    if first_layer:
                        first_layer = False
                        continue

                    layer.add_splitting_filters_indexes(pointwise_filters_subnetwork[block_index_pointwise] + 1)
                    block_index_pointwise += 1

    def set_print_hidden_feature(self, print=True):
        self.print_hidden_feature = print
        self.debug = print

    def set_debug(self, debug=True):
        self.debug = debug

    def get_model_name(self):
        return "Wake_Word_DS_CNN"

    def compute_lookup_table(self, train_data, average_run=100):
        """
        Compute the lookup table for the model given the training data set. A batch of the data is passed as input
        inside the model and the linear operation associated to the layer is performed for average_run times. Then the
        MACs for each layers's filters are computed and the memory size of each filter is computed too.
        @return: List[List] containing the filters forward times for each layer
        List[List] containing the filters macs for each layer
        """
        inputs = None
        for images, _ in train_data.take(1):
            inputs = images
            break

        layers_filters_macs = []
        layers_filters_byte = []

        inputs = self.reshape(inputs)

        for layer in self.layers:

            if isinstance(layer, Reds_DepthwiseConv2D) or isinstance(layer, Reds_2DConvolution_Standard):
                inputs, macs, filters_byte_memory = layer.compute_layer_lookup_table(
                    inputs=inputs)
                layers_filters_macs.append(macs)
                layers_filters_byte.append(filters_byte_memory)

        return layers_filters_macs, layers_filters_byte

    def build(self, input_shape):
        """
        Initialize model's layers weights and biases
        """

        init_input = tf.ones(
            input_shape,
            dtype=tf.dtypes.float32,
        )

        init_input = self.reshape(init_input)
        init_input = [init_input for _ in range(1)]

        for layer in self.layers[1:]:

            if isinstance(layer, Reds_BatchNormalizationBase):
                _ = layer.build(init_input[0].shape)
            elif isinstance(layer, Reds_2DConvolution_Standard):
                init_input, times = layer(init_input)
            elif isinstance(layer, Reds_DepthwiseConv2D):
                init_input = layer(init_input)
            elif isinstance(layer, Linear_Adaptive):
                init_input[0] = self.avg_pool(init_input[0])
                init_input[0] = self.flatten(init_input[0])
                _ = layer(init_input)

        self.built = True

    def call(self, inputs, training=None, mask=None):
        inputs = self.reshape(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Initial Input shape after Reshape") if self.debug else None

        # copy the input once for each subnetwork
        inputs = [inputs for _ in range(self.subnetworks_number)]

        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Initial Inputs shape") if self.debug else None

        inputs, times = self.standard_convolution(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Initial Standard Convolution") if self.debug else None

        inputs = self.batch_norm1(inputs)
        inputs = [self.relu1(input) for input in inputs]

        inputs = self.depthwise1(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Depthwise Convolution Layer 1") if self.debug else None
        inputs = self.batch_norm2(inputs)
        inputs = [self.relu2(input) for input in inputs]

        inputs, times = self.pointwise_conv1(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Pointwise Convolution Layer 1") if self.debug else None
        inputs = self.batch_norm3(inputs)
        inputs = [self.relu3(input) for input in inputs]

        inputs = self.depthwise2(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Depthwise Convolution Layer 2") if self.debug else None
        inputs = self.batch_norm4(inputs)
        inputs = [self.relu4(input) for input in inputs]

        inputs, times = self.pointwise_conv2(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Pointwise Convolution Layer 2") if self.debug else None
        inputs = self.batch_norm5(inputs)
        inputs = [self.relu5(input) for input in inputs]

        inputs = self.depthwise3(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Depthwise Convolution Layer 3") if self.debug else None
        inputs = self.batch_norm6(inputs)
        inputs = [self.relu6(input) for input in inputs]

        inputs, times = self.pointwise_conv3(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Pointwise Convolution Layer 3") if self.debug else None
        inputs = self.batch_norm7(inputs)
        inputs = [self.relu7(input) for input in inputs]

        inputs = self.depthwise4(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Depthwise Convolution Layer 4") if self.debug else None
        inputs = self.batch_norm8(inputs)
        inputs = [self.relu8(input) for input in inputs]

        inputs, times = self.pointwise_conv4(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Pointwise Convolution Layer 4") if self.debug else None
        inputs = self.batch_norm9(inputs)
        inputs = [self.relu9(input) for input in inputs]

        inputs = [
            tf.nn.avg_pool(input, ksize=[1, input.shape[1], input.shape[2], 1], strides=[1, 1, 1, 1], padding="VALID")
            for input in inputs]

        inputs = [tf.keras.layers.Flatten()(input) for input in inputs]
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="AFTER Flatten") if self.debug else None
        inputs, times = self.fc1(inputs)

        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="LAST output vector") if self.debug else None

        return inputs

    def get_model(self):
        model = tf.keras.Sequential()

        return model