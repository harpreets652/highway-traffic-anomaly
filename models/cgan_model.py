import tensorflow as tf
import collections

"""
Paper: https://arxiv.org/abs/1611.07004
Code from: https://github.com/affinelayer/pix2pix-tensorflow
"""

EPS = 1e-12
CROP_SIZE = 256

Model = collections.namedtuple("Model",
                               "outputs, predict_real, predict_fake, discrim_loss, "
                               "discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")


def create_model(arguments, inputs, targets):
    def create_discriminator(discriminator_inputs, discriminator_targets):
        """
        Discriminator architecture described in docs/discriminator_architecture.txt
        """

        n_layers = 3
        layers = []

        d_input = tf.concat([discriminator_inputs, discriminator_targets], axis=3)

        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(d_input, arguments.ndf, stride=2)
            rectified = leaky_relu(convolved, 0.2)
            layers.append(rectified)

        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                output_channels = arguments.ndf * min(2 ** (i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv(layers[-1], output_channels, stride=stride)
                normalized = batch_norm(convolved)
                rectified = leaky_relu(normalized, 0.2)
                layers.append(rectified)

        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        gen_outputs = create_generator(arguments, inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables: reuse=True under same variable scope
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            predict_fake = create_discriminator(inputs, gen_outputs)

    # losses
    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1; shape of this method is a parabola -(log(x) + log(1 - x))
        # disc output: 1 == real, 0 == fake .... log(1) == 0
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    # G attempts minimizes the second component (plus L1 loss) while D attempts to maximize the loss
    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_gan = tf.reduce_mean(-tf.log(predict_fake + EPS))

        gen_loss_l1 = tf.reduce_mean(tf.abs(targets - gen_outputs))

        # note~ while it's called l1 loss, it's actually average epe error
        # gen_loss_l1 = average_endpoint_error(targets, gen_outputs)

        gen_loss = gen_loss_gan * arguments.gan_weight + gen_loss_l1 * arguments.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(arguments.lr, arguments.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(arguments.lr, arguments.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_gan, gen_loss_l1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_gan),
        gen_loss_L1=ema.average(gen_loss_l1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=gen_outputs,
        train=tf.group(update_losses, incr_global_step, gen_train)
    )


def create_generator(arguments, generator_inputs, generator_outputs_channels):
    layers = []

    # encoder
    with tf.variable_scope("encoder_1"):
        output = gen_conv(arguments, generator_inputs, arguments.ngf, dilate=(4, 4), stride=(1, 1))
        output = tf.layers.max_pooling2d(output, pool_size=(2, 2), strides=(2, 2))
        layers.append(output)

    layer_specs = [
        (arguments.ngf * 2, (4, 4), (1, 1)),  # encoder_2: ngf => ngf * 2
        (arguments.ngf * 4, (2, 2), (1, 1)),  # encoder_3: ngf * 2 => ngf * 4
        (arguments.ngf * 8, (2, 2), (1, 1)),  # encoder_4: ngf * 4 => ngf * 8
        (arguments.ngf * 8, (1, 1), (2, 2)),  # encoder_5: ngf * 8 => ngf * 8
        (arguments.ngf * 8, (1, 1), (2, 2)),  # encoder_6: ngf * 8 => ngf * 8
        (arguments.ngf * 8, (1, 1), (2, 2)),  # encoder_7: ngf * 8 => ngf * 8
        # (arguments.ngf * 8, (1, 1), (2, 2)),  # encoder_8: ngf * 8 => ngf * 8
    ]

    for (out_channels, dilation, stride) in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = leaky_relu(layers[-1], 0.2)

            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(arguments, rectified, out_channels, stride=stride, dilate=dilation)

            if dilation[0] > 1:
                convolved = tf.layers.max_pooling2d(convolved, pool_size=(2, 2), strides=(2, 2))

            output = batch_norm(convolved)
            layers.append(output)

    # decoder
    layer_specs = [
        (arguments.ngf * 8, 0.5),  # decoder_8: ngf * 8 => ngf * 8 * 2
        (arguments.ngf * 8, 0.5),  # decoder_7: ngf * 8 * 2 => ngf * 8 * 2
        (arguments.ngf * 8, 0.5),  # decoder_6: ngf * 8 * 2 => ngf * 8 * 2
        (arguments.ngf * 8, 0.0),  # decoder_5: ngf * 8 * 2 => ngf * 8 * 2
        (arguments.ngf * 4, 0.0),  # decoder_4: ngf * 8 * 2 => ngf * 4 * 2
        (arguments.ngf * 2, 0.0),  # decoder_3: ngf * 4 * 2 => ngf * 2 * 2
        # (arguments.ngf, 0.0)  # decoder_2: ngf * 2 * 2] => ngf * 2
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                dec_input = layers[-1]
            else:
                dec_input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(dec_input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(arguments, rectified, out_channels)
            output = batch_norm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: ngf * 2 => generator_outputs_channels
    with tf.variable_scope("decoder_1"):
        dec_input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(dec_input)
        output = gen_deconv(arguments, rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def average_endpoint_error(labels, predictions):
    """
    Given labels and predictions of size (N, H, W, 2), calculates average endpoint error:
        sqrt[sum_across_channels{(X - Y)^2}]

    Average endpoint error is the average euclidean distance between the two optical flow vectors
    """
    num_samples = predictions.shape.as_list()[0]
    with tf.name_scope(None, "average_endpoint_error", (predictions, labels)) as scope:
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        squared_difference = tf.square(tf.subtract(predictions, labels))
        # sum across channels: sum[(X - Y)^2] -> N, H, W, 1
        loss = tf.reduce_sum(squared_difference, 3, keep_dims=True)
        loss = tf.sqrt(loss)

    return tf.reduce_sum(loss) / num_samples


def batch_norm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def leaky_relu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: arguments*x/2 - arguments*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def gen_deconv(arguments, batch_input, out_channels, stride=(2, 2)):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if arguments.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same",
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4,
                                          strides=stride, padding="same", kernel_initializer=initializer)


def gen_conv(arguments, batch_input, out_channels, stride=(2, 2), dilate=(1, 1)):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if arguments.separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4,
                                          strides=stride, padding="same", dilation_rate=dilate,
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=stride,
                                dilation_rate=dilate, padding="same", kernel_initializer=initializer)


def discrim_conv(batch_input, out_channels, stride):
    # rank in Tensorflow represents dimensions [d, 0] => before, [d, 1] => after where d=pad_row=input rank
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels,
                            kernel_size=4, strides=(stride, stride), padding="valid",
                            kernel_initializer=tf.random_normal_initializer(0, 0.02))