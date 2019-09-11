import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, \
    ReLU, UpSampling2D, Concatenate, Lambda, Input
from tensorflow.python.keras.applications import MobileNetV2
from tensorflow.python.keras.models import Model


def MobilenetSeparableConv2D(input, filters,
                             kernel_size,
                             strides=(1, 1),
                             padding='valid',
                             use_bias=True):
    x = DepthwiseConv2D(kernel_size, padding=padding, use_bias=use_bias, strides=strides)(input)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    x = Conv2D(filters, 1, padding='same', use_bias=use_bias, strides=1)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    return x


def make_last_layers_mobilenet(input, id, num_filters, out_filters):
    x = Conv2D(num_filters, kernel_size=1, padding='same', use_bias=False,
               name='block_' + str(id) + '_conv')(input)
    x = BatchNormalization(momentum=0.9, name='block_' + str(id) + '_BN')(x)
    x = ReLU(6., name='block_' + str(id) + '_relu6')(x)
    x = MobilenetSeparableConv2D(x, 2 * num_filters,
                                 kernel_size=(3, 3),
                                 use_bias=False,
                                 padding='same')
    x = Conv2D(num_filters, kernel_size=1, padding='same',
               use_bias=False, name='block_' + str(id + 1) + '_conv')(x)
    x = BatchNormalization(momentum=0.9, name='block_' + str(id + 1) + '_BN')(x)
    x = ReLU(6., name='block_' + str(id + 1) + '_relu6')(x)
    x = MobilenetSeparableConv2D(x, 2 * num_filters,
                                 kernel_size=(3, 3),
                                 use_bias=False,
                                 padding='same')
    x = Conv2D(num_filters, kernel_size=1, padding='same', use_bias=False, name='block_' + str(id + 2) + '_conv')(x)
    x = BatchNormalization(momentum=0.9, name='block_' + str(id + 2) + '_BN')(x)
    x = ReLU(6., name='block_' + str(id + 2) + '_relu6')(x)

    t = MobilenetSeparableConv2D(x, 2 * num_filters, kernel_size=(3, 3), use_bias=False, padding='same')
    y = Conv2D(out_filters, kernel_size=1, padding='same', use_bias=False)(t)
    return x, y


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MobilenetConv2D(input, kernel, alpha, filters):
    last_block_filters = _make_divisible(filters * alpha, 8)

    x = Conv2D(last_block_filters, kernel, padding='same', use_bias=False)(input)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    return x


def mobilenetv2_yolo_body(inputs, num_anchors, num_classes, alpha=1.0):
    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x 1024
    # conv_pw_11_relu :26 x 26 x 512
    # conv_pw_5_relu : 52 x 52 x 256
    mobilenetv2 = MobileNetV2(input_tensor=inputs, include_top=False, weights='imagenet')
    x, y1 = make_last_layers_mobilenet(mobilenetv2.output, 17, 512, num_anchors * (num_classes + 5))
    x = Conv2D(256, kernel_size=1, padding='same', use_bias=False, name='block_20_conv')(x)
    x = BatchNormalization(momentum=0.9, name='block_20_BN')(x)
    x = ReLU(6., name='block_20_relu6')(x)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, MobilenetConv2D(mobilenetv2.get_layer('block_12_project_BN').output, (1, 1), alpha, 384)])

    x, y2 = make_last_layers_mobilenet(x, 21, 256, num_anchors * (num_classes + 5))
    x = Conv2D(128, kernel_size=1, padding='same', use_bias=False, name='block_24_conv')(x)
    x = BatchNormalization(momentum=0.9, name='block_24_BN')(x)
    x = ReLU(6., name='block_24_relu6')(x)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, MobilenetConv2D(mobilenetv2.get_layer('block_5_project_BN').output, (1, 1), alpha, 128)])
    x, y3 = make_last_layers_mobilenet(x, 25, 128, num_anchors * (num_classes + 5))

    # y1 = Lambda(lambda y: tf.reshape(y, [-1, tf.shape(y)[1],tf.shape(y)[2], num_anchors, num_classes + 5]),name='y1')(y1)
    # y2 = Lambda(lambda y: tf.reshape(y, [-1, tf.shape(y)[1],tf.shape(y)[2], num_anchors, num_classes + 5]),name='y2')(y2)
    # y3 = Lambda(lambda y: tf.reshape(y, [-1, tf.shape(y)[1],tf.shape(y)[2], num_anchors, num_classes + 5]),name='y3')(y3)
    return Model(inputs, [y1, y2, y3])
    # f1 = mobilenetv2.output
    # # f1 :13 x 13 x 1024
    # x, y1 = last_layers(f1, 512, num_anchors * (num_classes + 5))
    #
    # x = DarnetConv_BN_Leaky(x,256, 1,blocks='a')
    # x = UpSampling2D(2)(x)
    #
    # f2 = mobilenetv2.get_layer('block_12_project_BN').output
    # # f2: 26 x 26 x 512
    # x = Concatenate()([x, f2])
    #
    # x, y2 = last_layers(x, 256, num_anchors * (num_classes + 5))
    #
    # x = DarnetConv_BN_Leaky(x,128, 1,blocks='b')
    # x = UpSampling2D(2)(x)
    #
    # f3 = mobilenetv2.get_layer('block_5_project_BN').output
    # # f3 : 52 x 52 x 256
    # x = Concatenate()([x, f3])
    # x, y3 = last_layers(x, 128, num_anchors * (num_classes + 5))
    #
    # return tf.keras.models.Model(inputs=inputs, outputs=[y1, y2, y3])


class _LayersOverride:

    def __init__(self,
                 default_batchnorm_momentum=0.999,
                 conv_hyperparams=None,
                 use_explicit_padding=False,
                 alpha=1.0,
                 min_depth=None):
        """Alternative tf.keras.layers interface, for use by the Keras MobileNetV2.
        It is used by the Keras applications kwargs injection API to
        modify the Mobilenet v2 Keras application with changes required by
        the Object Detection API.
        These injected interfaces make the following changes to the network:
        - Applies the Object Detection hyperparameter configuration
        - Supports FreezableBatchNorms
        - Adds support for a min number of filters for each layer
        - Makes the `alpha` parameter affect the final convolution block even if it
            is less than 1.0
        - Adds support for explicit padding of convolutions
        Args:
          batchnorm_training: Bool. Assigned to Batch norm layer `training` param
            when constructing `freezable_batch_norm.FreezableBatchNorm` layers.
          default_batchnorm_momentum: Float. When 'conv_hyperparams' is None,
            batch norm layers will be constructed using this value as the momentum.
          conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
            containing hyperparameters for convolution ops. Optionally set to `None`
            to use default mobilenet_v2 layer builders.
          use_explicit_padding: If True, use 'valid' padding for convolutions,
            but explicitly pre-pads inputs so that the output dimensions are the
            same as if 'same' padding were used. Off by default.
          alpha: The width multiplier referenced in the MobileNetV2 paper. It
            modifies the number of filters in each convolutional layer.
          min_depth: Minimum number of filters in the convolutional layers.
        """
        self._default_batchnorm_momentum = default_batchnorm_momentum
        self._conv_hyperparams = conv_hyperparams
        self._use_explicit_padding = use_explicit_padding
        self._alpha = alpha
        self._min_depth = min_depth
        self._regularizer = tf.keras.regularizers.l2(0.00004)
        self._initializer = tf.random_normal_initializer(stddev=0.03)

    def Conv2D(self, filters, **kwargs):
        """Builds a Conv2D layer according to the current Object Detection config.

        Overrides the Keras MobileNetV2 application's convolutions with ones that
        follow the spec specified by the Object Detection hyperparameters.

        Args:
          filters: The number of filters to use for the convolution.
          **kwargs: Keyword args specified by the Keras application for
            constructing the convolution.

        Returns:
          A one-arg callable that will either directly apply a Keras Conv2D layer to
          the input argument, or that will first pad the input then apply a Conv2D
          layer.
        """
        if kwargs.get('name') == 'Conv_1' and self._alpha < 1.0:
            filters = _make_divisible(1280 * self._alpha, 8)

        if self._min_depth and (filters < self._min_depth) and not kwargs.get('name').endswith('expand'):
            filters = self._min_depth

        # if self._conv_hyperparams:
        #     kwargs=self._conv_hyperparams.params(**kwargs)
        # else:
        #     kwargs['kernel_regularizer']=self._regularizer
        #     kwargs['kernel_initializer']=self._initializer

        kwargs['padding'] = 'same'
        kernel_size = kwargs.get('kernel_size')
        if self._use_explicit_padding and kernel_size > 1:
            kwargs['padding'] = 'valid'

            def padded_conv(features):
                padded_features = self._FixedPaddingLayer(kernel_size)(features)
                return tf.keras.layers.Conv2D(filters, **kwargs)(padded_features)

            return padded_conv
        else:
            return tf.keras.layers.Conv2D(filters, **kwargs)

    def DepthwiseConv2D(self, **kwargs):
        """Builds a DepthwiseConv2D according to the Object Detection config.

        Overrides the Keras MobileNetV2 application's convolutions with ones that
        follow the spec specified by the Object Detection hyperparameters.

        Args:
          **kwargs: Keyword args specified by the Keras application for
            constructing the convolution.

        Returns:
          A one-arg callable that will either directly apply a Keras DepthwiseConv2D
          layer to the input argument, or that will first pad the input then apply
          the depthwise convolution.
        """
        if self._conv_hyperparams:
            kwargs = self._conv_hyperparams.params(**kwargs)
        else:
            kwargs['depthwise_initializer'] = self._initializer

        kwargs['padding'] = 'same'
        kernel_size = kwargs.get('kernel_size')
        if self._use_explicit_padding and kernel_size > 1:
            kwargs['padding'] = 'valid'

            def padded_depthwise_conv(features):
                padded_features = self._FixedPaddingLayer(kernel_size)(features)
                return tf.keras.layers.DepthwiseConv2D(**kwargs)(padded_features)

            return padded_depthwise_conv
        else:
            return tf.keras.layers.DepthwiseConv2D(**kwargs)

    def BatchNormalization(self, **kwargs):
        """Builds a normalization layer.
        Overrides the Keras application batch norm with the norm specified by the
        Object Detection configuration.
        Args:
          **kwargs: Only the name is used, all other params ignored.
            Required for matching `layers.BatchNormalization` calls in the Keras
            application.
        Returns:
          A normalization layer specified by the Object Detection hyperparameter
          configurations.
        """
        name = kwargs.get('name')
        if self._conv_hyperparams:
            return self._conv_hyperparams.build_batch_norm(name=name)
        else:
            return tf.keras.layers.BatchNormalization(
                momentum=self._default_batchnorm_momentum, name=name)

    def Input(self, shape):
        """Builds an Input layer.

        Overrides the Keras application Input layer with one that uses a
        tf.placeholder_with_default instead of a tf.placeholder. This is necessary
        to ensure the application works when run on a TPU.

        Args:
          shape: The shape for the input layer to use. (Does not include a dimension
            for the batch size).
        Returns:
          An input layer for the specified shape that internally uses a
          placeholder_with_default.
        """
        default_size = 224
        default_batch_size = 1
        shape = list(shape)
        default_shape = [default_size if dim is None else dim for dim in shape]

        input_tensor = tf.constant(0.0, shape=[default_batch_size] + default_shape)

        placeholder_with_default = tf.placeholder_with_default(
            input=input_tensor, shape=[None] + shape)
        return tf.keras.layers.Input(tensor=placeholder_with_default)

    def ReLU(self, *args, **kwargs):
        """Builds an activation layer.

        Overrides the Keras application ReLU with the activation specified by the
        Object Detection configuration.

        Args:
          *args: Ignored, required to match the `tf.keras.ReLU` interface
          **kwargs: Only the name is used,
            required to match `tf.keras.ReLU` interface

        Returns:
          An activation layer specified by the Object Detection hyperparameter
          configurations.
        """
        name = kwargs.get('name')
        if self._conv_hyperparams:
            return self._conv_hyperparams.build_activation_layer(name=name)
        else:
            return tf.keras.layers.Lambda(tf.nn.relu6, name=name)

    def ZeroPadding2D(self, **kwargs):
        """Replaces explicit padding in the Keras application with a no-op.

        Args:
          **kwargs: Ignored, required to match the Keras applications usage.

        Returns:
          A no-op identity lambda.
        """
        return lambda x: x


def mobilenet_v2(default_batchnorm_momentum=0.999,
                 conv_hyperparams=None,
                 use_explicit_padding=False,
                 alpha=1.0,
                 min_depth=None,
                 **kwargs):
    """Instantiates the MobileNetV2 architecture, modified for object detection.
      This wraps the MobileNetV2 tensorflow Keras application, but uses the
      Keras application's kwargs-based monkey-patching API to override the Keras
      architecture with the following changes:
      - Changes the default batchnorm momentum to 0.9997
      - Applies the Object Detection hyperparameter configuration
      - Supports FreezableBatchNorms
      - Adds support for a min number of filters for each layer
      - Makes the `alpha` parameter affect the final convolution block even if it
          is less than 1.0
      - Adds support for explicit padding of convolutions
      - Makes the Input layer use a tf.placeholder_with_default instead of a
          tf.placeholder, to work on TPUs.
      Args:
          batchnorm_training: Bool. Assigned to Batch norm layer `training` param
            when constructing `freezable_batch_norm.FreezableBatchNorm` layers.
          default_batchnorm_momentum: Float. When 'conv_hyperparams' is None,
            batch norm layers will be constructed using this value as the momentum.
          conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
            containing hyperparameters for convolution ops. Optionally set to `None`
            to use default mobilenet_v2 layer builders.
          use_explicit_padding: If True, use 'valid' padding for convolutions,
            but explicitly pre-pads inputs so that the output dimensions are the
            same as if 'same' padding were used. Off by default.
          alpha: The width multiplier referenced in the MobileNetV2 paper. It
            modifies the number of filters in each convolutional layer.
          min_depth: Minimum number of filters in the convolutional layers.
          **kwargs: Keyword arguments forwarded directly to the
            `tf.keras.applications.MobilenetV2` method that constructs the Keras
            model.
      Returns:
          A Keras model instance.
      """
    layers_override = _LayersOverride(
        default_batchnorm_momentum=default_batchnorm_momentum,
        conv_hyperparams=conv_hyperparams,
        use_explicit_padding=use_explicit_padding,
        min_depth=min_depth,
        alpha=alpha)
    return tf.keras.applications.MobileNetV2(alpha=alpha, layers=layers_override, **kwargs)
