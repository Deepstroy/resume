def residual_block(input_data, filters, strides, is_train, block_name):
    with tf.variable_scope(block_name):
        # Layer1
        with tf.variable_scope('layer1'):
            layer = tf.layers.Conv2D(filters=filters, kernel_size=3, use_bias=False,
                                     strides=strides, padding='SAME', activation=None) \
                (input_data)
            layer = batch_norm_(layer, filters, is_train)
            layer = tf.nn.relu(layer)
        # Layer2
        with tf.variable_scope('layer2'):
            layer = tf.layers.Conv2D(filters=filters, kernel_size=3, use_bias=False,
                                     strides=strides, padding='SAME', activation=None) \
                (layer)
            layer = batch_norm_(layer, filters, is_train)
            layer = tf.nn.relu(layer)
        # Projection layer
        if input_data.shape[-1] != filters:
            input_data = tf.layers.Conv2D(filters=filters, kernel_size=1,
                                          strides=1,
                                          padding='SAME',
                                          activation=tf.nn.relu
                                          )(input_data)
        return layer + input_data