import collections
import tensorflow as tf

"""
Discover GAN
Paper: https://arxiv.org/pdf/1709.08810.pdf 
"""

EPS = 1e-12

Model = collections.namedtuple("Model",
                               "outputs,predict_ab_real,predict_ab_fake,predict_ba_real,predict_ba_fake,"
                               "discrim_ab_loss,discrim_ba_loss,discrim_ab_grads_and_vars,discrim_ba_grads_and_vars,"
                               "gen_ab_loss_GAN,gen_ab_loss_L1,gen_ba_loss_GAN,gen_ba_loss_L1,"
                               "gen_ab_grads_and_vars,gen_ba_grads_and_vars,train")


def create_model(arguments, source_input_a, source_input_b):
    # create two generators
    with tf.name_scope("generator_ab_forward"):
        with tf.variable_scope("GeneratorAB"):
            out_channels = int(source_input_b.get_shape()[-1])
            gen_a_b = create_generator(arguments, source_input_a, out_channels)

    with tf.name_scope("generator_ba_forward"):
        with tf.variable_scope("GeneratorBA"):
            out_channels = int(source_input_a.get_shape()[-1])
            gen_b_a = create_generator(arguments, source_input_b, out_channels)

    with tf.name_scope("generator_ab_inverse"):
        with tf.variable_scope("GeneratorAB", reuse=True):
            out_channels = int(source_input_b.get_shape()[-1])
            gen_b_a_b = create_generator(arguments, gen_b_a, out_channels)

    with tf.name_scope("generator_ba_inverse"):
        with tf.variable_scope("GeneratorBA", reuse=True):
            out_channels = int(source_input_a.get_shape()[-1])
            gen_a_b_a = create_generator(arguments, gen_a_b, out_channels)

    # create two discriminators
    with tf.name_scope("real_discriminator_ab"):
        with tf.variable_scope("DiscriminatorAB"):
            predict_ab_real = create_patch_discriminator(arguments, source_input_a, source_input_b)

    with tf.name_scope("fake_discriminator_ab"):
        with tf.variable_scope("DiscriminatorAB", reuse=True):
            predict_ab_fake = create_patch_discriminator(arguments, source_input_a, gen_a_b)

    with tf.name_scope("real_discriminator_ba"):
        with tf.variable_scope("DiscriminatorBA"):
            predict_ba_real = create_patch_discriminator(arguments, source_input_b, source_input_a)

    with tf.name_scope("fake_discriminator_ba"):
        with tf.variable_scope("DiscriminatorBA", reuse=True):
            predict_ba_fake = create_patch_discriminator(arguments, source_input_b, gen_b_a)

    # losses

    # minimizing -tf.log will try to get inputs to 1; shape of this method is a parabola -(log(x) + log(1 - x))
    # disc output: 1 == real, 0 == fake .... log(1) == 0
    # predict_real => 1
    # predict_fake => 0
    with tf.name_scope("discriminator_ab_loss"):
        discrim_ab_loss = tf.reduce_mean(-(tf.log(predict_ab_real + EPS) + tf.log(1 - predict_ab_fake + EPS)))

    with tf.name_scope("discriminator_ba_loss"):
        discrim_ba_loss = tf.reduce_mean(-(tf.log(predict_ba_real + EPS) + tf.log(1 - predict_ba_fake + EPS)))

    # predict_fake => 1
    # abs(targets - outputs) => 0
    with tf.name_scope("generator_ab_loss"):
        gen_ab_loss_gan = tf.reduce_mean(-tf.log(predict_ab_fake + EPS))
        gen_ab_loss_l1 = tf.reduce_mean(tf.abs(source_input_a - gen_a_b_a))
        gen_ab_loss = gen_ab_loss_gan * arguments.gan_weight + gen_ab_loss_l1 * arguments.l1_weight

    with tf.name_scope("generator_ba_loss"):
        gen_ba_loss_gan = tf.reduce_mean(-tf.log(predict_ba_fake + EPS))
        gen_ba_loss_l1 = tf.reduce_mean(tf.abs(source_input_b - gen_b_a_b))
        gen_ba_loss = gen_ba_loss_gan * arguments.gan_weight + gen_ba_loss_l1 * arguments.l1_weight

    # training
    with tf.name_scope("discriminator_ab_train"):
        discrim_ab_tvars = [var for var in tf.trainable_variables() if var.name.startswith("DiscriminatorAB")]
        discrim_ab_optim = tf.train.AdamOptimizer(arguments.lr, arguments.beta1)
        discrim_ab_grads_and_vars = discrim_ab_optim.compute_gradients(discrim_ab_loss, var_list=discrim_ab_tvars)
        discrim_ab_train = discrim_ab_optim.apply_gradients(discrim_ab_grads_and_vars)

    with tf.name_scope("discriminator_ba_train"):
        discrim_ba_tvars = [var for var in tf.trainable_variables() if var.name.startswith("DiscriminatorBA")]
        discrim_ba_optim = tf.train.AdamOptimizer(arguments.lr, arguments.beta1)
        discrim_ba_grads_and_vars = discrim_ba_optim.compute_gradients(discrim_ba_loss, var_list=discrim_ba_tvars)
        discrim_ba_train = discrim_ba_optim.apply_gradients(discrim_ba_grads_and_vars)

    with tf.name_scope("generator_ab_train"):
        with tf.control_dependencies([discrim_ab_train]):
            gen_ab_tvars = [var for var in tf.trainable_variables() if var.name.startswith("GeneratorAB")]
            gen_ab_optim = tf.train.AdamOptimizer(arguments.lr, arguments.beta1)
            gen_ab_grads_and_vars = gen_ab_optim.compute_gradients(gen_ab_loss, var_list=gen_ab_tvars)
            gen_ab_train = gen_ab_optim.apply_gradients(gen_ab_grads_and_vars)

    with tf.name_scope("generator_ba_train"):
        with tf.control_dependencies([discrim_ba_train]):
            gen_ba_tvars = [var for var in tf.trainable_variables() if var.name.startswith("GeneratorBA")]
            gen_ba_optim = tf.train.AdamOptimizer(arguments.lr, arguments.beta1)
            gen_ba_grads_and_vars = gen_ba_optim.compute_gradients(gen_ba_loss, var_list=gen_ba_tvars)
            gen_ba_train = gen_ba_optim.apply_gradients(gen_ba_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_ab_loss, gen_ab_loss_gan, gen_ab_loss_l1,
                               discrim_ba_loss, gen_ba_loss_gan, gen_ba_loss_l1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(predict_ab_real=predict_ab_real,
                 predict_ab_fake=predict_ab_fake,
                 predict_ba_real=predict_ba_real,
                 predict_ba_fake=predict_ba_fake,
                 discrim_ab_loss=ema.average(discrim_ab_loss),
                 discrim_ba_loss=ema.average(discrim_ba_loss),
                 discrim_ab_grads_and_vars=discrim_ab_grads_and_vars,
                 discrim_ba_grads_and_vars=discrim_ba_grads_and_vars,
                 gen_ab_loss_GAN=ema.average(gen_ab_loss_gan),
                 gen_ba_loss_GAN=ema.average(gen_ba_loss_gan),
                 gen_ab_loss_L1=ema.average(gen_ab_loss_l1),
                 gen_ba_loss_L1=ema.average(gen_ba_loss_l1),
                 gen_ab_grads_and_vars=gen_ab_grads_and_vars,
                 gen_ba_grads_and_vars=gen_ba_grads_and_vars,
                 outputs=[gen_a_b, gen_b_a, gen_a_b_a, gen_b_a_b],
                 train=tf.group(update_losses, incr_global_step, gen_ab_train, gen_ba_train))


def create_generator(arguments, generator_inputs, generator_outputs_channels):
    layers = []

    # encoder
    with tf.variable_scope("encoder_1"):
        output = convolution_layer(arguments, generator_inputs, arguments.ngf)
        layers.append(output)

    layer_specs = [
        arguments.ngf * 2,  # encoder_2: ngf => ngf * 2
        arguments.ngf * 4,  # encoder_3: ngf * 2 => ngf * 4
        arguments.ngf * 8,  # encoder_4: ngf * 4 => ngf * 8
        arguments.ngf * 8,  # encoder_5: ngf * 8 => ngf * 8
        arguments.ngf * 8,  # encoder_6: ngf * 8 => ngf * 8
        arguments.ngf * 8,  # encoder_7: ngf * 8 => ngf * 8
        arguments.ngf * 8,  # encoder_8: ngf * 8 => ngf * 8
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = leaky_relu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = convolution_layer(arguments, rectified, out_channels)
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
        (arguments.ngf, 0.0),  # decoder_2: ngf * 2 * 2] => ngf * 2
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
            output = deconvolution_layer(arguments, rectified, out_channels)
            output = batch_norm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: ngf * 2 => generator_outputs_channels
    with tf.variable_scope("decoder_1"):
        dec_input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(dec_input)
        output = deconvolution_layer(arguments, rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_patch_discriminator(arguments, discriminator_inputs, discriminator_targets):
    """
    Discriminator architecture described in docs/discriminator_architecture.txt
    """

    n_layers = 3
    layers = []

    d_input = tf.concat([discriminator_inputs, discriminator_targets], axis=3)

    with tf.variable_scope("layer_1"):
        convolved = discriminator_convolution_layer(d_input, arguments.ndf, stride=2)
        rectified = leaky_relu(convolved, 0.2)
        layers.append(rectified)

    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            output_channels = arguments.ndf * min(2 ** (i + 1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = discriminator_convolution_layer(layers[-1], output_channels, stride=stride)
            normalized = batch_norm(convolved)
            rectified = leaky_relu(normalized, 0.2)
            layers.append(rectified)

    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = discriminator_convolution_layer(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]


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


def deconvolution_layer(arguments, batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if arguments.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same",
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                          kernel_initializer=initializer)


def convolution_layer(arguments, batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if arguments.separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                kernel_initializer=initializer)


def discriminator_convolution_layer(batch_input, out_channels, stride):
    # rank in Tensorflow represents dimensions [d, 0] => before, [d, 1] => after where d=pad_row=input rank
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels,
                            kernel_size=4, strides=(stride, stride), padding="valid",
                            kernel_initializer=tf.random_normal_initializer(0, 0.02))