# Copyright 2023, Erik Myklebust, Andreas Koehler, MIT license
""" 
Various variations of PhaseNet
Author: Erik Myklebust
"""

import tensorflow as tf 
import tensorflow.keras.layers as tfl 
import tensorflow.keras.backend as K
import numpy as np

def crop_and_concat(x, y):
    to_crop = x.shape[1] - y.shape[1]
    if to_crop < 0:
        to_crop = abs(to_crop)
        of_start, of_end = to_crop // 2, to_crop // 2
        of_end += to_crop % 2
        y = tfl.Cropping1D((of_start, of_end))(y)
    elif to_crop > 0:
        of_start, of_end = to_crop // 2, to_crop // 2
        of_end += to_crop % 2
        y = tfl.ZeroPadding1D((of_start, of_end))(y)
    return tfl.concatenate([x,y])

def crop_and_add(x, y):
    to_crop = x.shape[1] - y.shape[1]
    if to_crop < 0:
        to_crop = abs(to_crop)
        of_start, of_end = to_crop // 2, to_crop // 2
        of_end += to_crop % 2
        y = tfl.Cropping1D((of_start, of_end))(y)
    elif to_crop > 0:
        of_start, of_end = to_crop // 2, to_crop // 2
        of_end += to_crop % 2
        y = tfl.ZeroPadding1D((of_start, of_end))(y)
    return x + y



class TransformerBlock(tfl.Layer):
    def __init__(self, key_dim, num_heads, ff_dim, value_dim=None, rate=0.1):
        super().__init__()
        self.att = tfl.MultiHeadAttention(num_heads=num_heads,
                                          key_dim=key_dim,
                                          value_dim=value_dim)
        self.ffn = tf.keras.Sequential(
            [tfl.Dense(ff_dim, activation="relu"), tfl.Dense(key_dim)]
        )
        self.layernorm1 = tfl.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tfl.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tfl.Dropout(rate)
        self.dropout2 = tfl.Dropout(rate)

    def call(self, inputs, training):
        if isinstance(inputs, (list, tuple)):
            query, value = inputs
        else:
            query, value = inputs, inputs

        attn_output = self.att(query, value)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(query + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class ResnetBlock1D(tfl.Layer):
    def __init__(self,
                 filters,
                 kernelsize,
                 activation='linear',
                 dropout=0.1, **kwargs):
        """1D resnet block

        Args:
            filters (int): number of filters .
            kernel_size (int): size of filters .
            activation (str): layer activation.
            dropout (float): dropout fraction .
        """
        super(ResnetBlock1D, self).__init__()
        self.filters = filters
        self.projection = tfl.Conv1D(filters, 1, padding='same', **kwargs)
        self.conv1 = tfl.Conv1D(filters, kernelsize, activation=None, padding='same', **kwargs)
        self.conv2 = tfl.Conv1D(filters, kernelsize, activation=None, padding='same', **kwargs)
        self.dropout1 = tfl.Dropout(dropout)
        self.bn1 = tfl.BatchNormalization()
        self.bn2 = tfl.BatchNormalization()
        self.bn3 = tfl.BatchNormalization()
        self.add = tfl.Add()
        self.relu = tfl.Activation(activation)

    def call(self, inputs, training=None):
        x = self.projection(inputs)
        fx = self.bn1(inputs)
        fx = self.conv1(fx)
        fx = self.bn2(fx)
        fx = self.relu(fx)
        fx = self.dropout1(fx)
        fx = self.conv2(fx)
        x = self.add([x, fx])
        x = self.bn3(x)
        x = self.relu(x)
        return x

class ResidualConv1D(tfl.Layer):

    def __init__(self,
                 filters=32,
                 kernel_size=3,
                 stacked_layer=1,
                 activation='relu',
                 causal=False):
        """1D residual convolution 
        
        Args:
            filters (int): number of filters.
            kernel_size (int): size of filters .
            stacked_layers (int): number of stacked layers .
        """

        super(ResidualConv1D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stacked_layer = stacked_layer
        self.causal = causal
        self.activation = activation

    def build(self, input_shape):
        self.sigmoid_layers = []
        self.tanh_layers = []
        self.conv_layers = []

        self.shape_matching_layer = tfl.Conv1D(self.filters, 1, padding = 'same')
        self.add = tfl.Add()
        self.final_activation = tf.keras.activations.get(self.activation)

        for dilation_rate in [2 ** i for i in range(self.stacked_layer)]:
            self.sigmoid_layers.append(
                tfl.Conv1D(self.filters, self.kernel_size, dilation_rate=dilation_rate,
                           padding='causal' if self.causal else 'same',
                                    activation='sigmoid'))
            self.tanh_layers.append(
                tfl.Conv1D(self.filters, self.kernel_size, dilation_rate=dilation_rate,
                           padding='causal' if self.causal else 'same',
                                    activation='tanh'))
            self.conv_layers.append(tfl.Conv1D(self.filters, 1, padding='same'))

    def get_config(self):
        return dict(name=self.name,
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    stacked_layer=self.stacked_layer)

    def call(self, inputs):
        out = self.shape_matching_layer(inputs)
        residual_output = out
        x = inputs
        for sl, tl, cl in zip(self.sigmoid_layers, self.tanh_layers, self.conv_layers):
            sigmoid_x = sl(x)
            tanh_x = tl(x)

            x = tfl.multiply([sigmoid_x, tanh_x])
            x = cl(x)
            residual_output = tfl.add([residual_output, x])

        return self.final_activation(self.add([out, x]))



class PhaseNet(tf.keras.Model):
    def __init__(self,
                 num_classes=2,
                 filters=None,
                 kernelsizes=None,
                 output_activation='linear',
                 kernel_regularizer=None,
                 dropout_rate=0.2,
                 pool_type='max',
                 activation='relu',
                 initializer='glorot_normal',
                 conv_type='default',
                 name='PhaseNet'):
        """Adapted to 1D from https://keras.io/examples/vision/oxford_pets_image_segmentation/

        Args:
            num_classes (int, optional): number of outputs. Defaults to 2.
            filters (list, optional): list of number of filters. Defaults to None.
            kernelsizes (list, optional): list of kernel sizes. Defaults to None.
            output_activation (str, optional): output activation, eg., 'softmax' for multiclass problems. Defaults to 'linear'.
            kernel_regularizer (tf.keras.regualizers.Regualizer, optional): kernel regualizer. Defaults to None.
            dropout_rate (float, optional): dropout. Defaults to 0.2.
            initializer (tf.keras.initializers.Initializer, optional): weight initializer. Defaults to 'glorot_normal'.
            name (str, optional): model name. Defaults to 'PhaseNet'.
        """
        super(PhaseNet, self).__init__(name=name)
        self.num_classes = num_classes
        self.initializer = initializer
        self.kernel_regularizer = kernel_regularizer
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.activation = activation

        if filters is None:
            self.filters = [4, 8, 16, 32]
        else:
            self.filters = filters

        if kernelsizes is None:
            self.kernelsizes = [7, 7, 7, 7]
        else:
            self.kernelsizes = kernelsizes
            
        if pool_type == 'max':
            self.pool_layer = tfl.MaxPooling1D
        else:
            self.pool_layer = tfl.AveragePooling1D
            
        if conv_type == 'seperable':
            self.conv_layer = tfl.SeparableConv1D
        else:
            self.conv_layer = tfl.Conv1D

    def _down_block(self, f, ks, x):
        x = self.conv_layer(f, 
                        ks, 
                        padding="same",
                        kernel_regularizer=self.kernel_regularizer,
                        kernel_initializer=self.initializer)(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)
        x = self.pool_layer(4, 2, padding='same')(x)
        return x
    
    def _up_block(self, f, ks, x):
        x = self.conv_layer(f, 
                            ks, 
                            padding="same",
                            kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=self.initializer)(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)
        x = tfl.UpSampling1D(2)(x)
        return x
        

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape[1:])

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = self.conv_layer(self.filters[0], 
                            self.kernelsizes[0],
                       kernel_regularizer=self.kernel_regularizer,
                       padding="same",
                       name='entry')(inputs)

        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)

        skips = [x]
        
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for i, _ in enumerate(self.filters):
            x = self._down_block(self.filters[i], self.kernelsizes[i], x)
            skips.append(x)
            
        skips = skips[:-1]

        self.encoder = tf.keras.Model(inputs, x)
        
        for i in list(range(len(self.filters)))[::-1]:
            x = self._up_block(self.filters[i], self.kernelsizes[i], x)
            x = crop_and_concat(x, skips[i])

        to_crop = x.shape[1] - input_shape[1]
        if to_crop != 0:
            of_start, of_end = to_crop // 2, to_crop // 2
            of_end += to_crop % 2
            x = tfl.Cropping1D((of_start, of_end))(x)
        
        #Exit block
        x = self.conv_layer(self.filters[0], 
                            self.kernelsizes[0],
                            kernel_regularizer=self.kernel_regularizer,
                            padding="same",
                            name='exit')(x)

        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)

        # Add a per-pixel classification layer
        if self.num_classes is not None:
            x = tfl.Conv1D(self.num_classes,
                           1,
                           padding="same")(x)
            outputs = tfl.Activation(self.output_activation, dtype='float32')(x)
        else:
            outputs = x

        # Define the model
        self.model = tf.keras.Model(inputs, outputs)

    @property
    def num_parameters(self):
        return sum([np.prod(K.get_value(w).shape) for w in self.model.trainable_weights])

    def summary(self):
        return self.model.summary()

    def call(self, inputs):
        return self.model(inputs)
    
class EPick(PhaseNet):
    def __init__(self,
                 num_classes=2,
                 output_layer=None,
                 filters=None,
                 kernelsizes=None,
                 output_activation='linear',
                 kernel_regularizer=None,
                 dropout_rate=0.2,
                 att_type='additive',
                 activation='relu',
                 pool_type='max',
                 initializer='glorot_normal',
                 residual_attention=None,
                 name='EPick'):
        """
        https://arxiv.org/abs/2109.02567
        
        Args:
            num_classes (int, optional): number of outputs. Defaults to 2.
            filters (list, optional): list of number of filters. Defaults to None.
            kernelsizes (list, optional): list of kernel sizes. Defaults to None.
            residual_attention (list: optional): list of residual attention sizes, one longer that filters. 
            att_type (str): dot or concat
            output_activation (str, optional): output activation, eg., 'softmax' for multiclass problems. Defaults to 'linear'.
            kernel_regularizer (tf.keras.regualizers.Regualizer, optional): kernel regualizer. Defaults to None.
            dropout_rate (float, optional): dropout. Defaults to 0.2.
            initializer (tf.keras.initializers.Initializer, optional): weight initializer. Defaults to 'glorot_normal'.
            name (str, optional): model name. Defaults to 'PhaseNet'.
        """
        super(EPick, self).__init__(num_classes=num_classes, 
                                              filters=filters, 
                                              kernelsizes=kernelsizes, 
                                              output_activation=output_activation, 
                                              kernel_regularizer=kernel_regularizer, 
                                              dropout_rate=dropout_rate,
                                              pool_type=pool_type, 
                                              activation=activation, 
                                              initializer=initializer, 
                                              name=name)

        if residual_attention is None:
            self.residual_attention = [16, 16, 16, 16, 16]
        else:
            self.residual_attention = residual_attention

    def _down_block(self, f, ks, x):
        x = tfl.Conv1D(f, ks, padding="same",
                        kernel_regularizer=self.kernel_regularizer,
                        kernel_initializer=self.initializer,
                        )(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)
        x = self.pool_layer(4, strides=2, padding='same')(x)
        return x

    def _up_block(self, f, ks, x, upsample=True):
        x = tfl.Conv1DTranspose(f, ks, padding="same",
                                kernel_regularizer=self.kernel_regularizer,
                                kernel_initializer=self.initializer,
                                )(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)
        if upsample:
            x = tfl.UpSampling1D(2)(x)
        return x
            

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape[1:])

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = tfl.Conv1D(self.filters[0], self.kernelsizes[0],
                       strides=1,
                       kernel_regularizer=self.kernel_regularizer,
                       padding="same",
                       name='entry')(inputs)

        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)
        
        skips = [x]
        
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for ks, f in zip(self.kernelsizes[1:], self.filters[1:]):
            x = self._down_block(f, ks, x) 
            skips.append(x)
        
        attentions = []
        for i, skip in enumerate(skips):
            if self.residual_attention[i] <= 0:
                att = skip
            elif i == 0:
                att = tfl.MultiHeadAttention(num_heads=8, 
                                             key_dim=self.residual_attention[i],)(skip, skip, return_attention_scores=False)
            else:
                tmp = []
                z = skips[i]
                for j, skip2 in enumerate(skips[:i]):
                    if self.residual_attention[j] <= 0:
                        att = tfl.Conv1D(self.filters[j], 3, activation='relu', padding='same')(z)
                    else:
                        att = tfl.MultiHeadAttention(num_heads=8, 
                                                     key_dim=self.residual_attention[j])(z, skip2, return_attention_scores=False)
                    tmp.append(att)
                att = tfl.Concatenate()(tmp)
            attentions.append(att)
            
        x = crop_and_concat(x, attentions[-1])
        self.encoder = tf.keras.Model(inputs, x)
            
        i = len(self.filters) - 1
        for f, ks in zip(self.filters[::-1][:-1], self.kernelsizes[::-1][:-1]):
            x = self._up_block(f, ks, x, upsample = i != 0)
            x = crop_and_concat(x, attentions[i-1])
            i -= 1
        
        to_crop = x.shape[1] - input_shape[1]
        if to_crop != 0:
            of_start, of_end = to_crop // 2, to_crop // 2
            of_end += to_crop % 2
            x = tfl.Cropping1D((of_start, of_end))(x)

        # Add a per-pixel classification layer
        if self.num_classes is not None:
            x = tfl.Conv1D(self.num_classes,
                           1,
                           padding="same")(x)
            outputs = tfl.Activation(self.output_activation, dtype='float32')(x)
        elif self.output_layer is not None:
            outputs = self.output_layer(x)
        else:
            outputs = x

        # Define the model
        self.model = tf.keras.Model(inputs, outputs)

    @property
    def num_parameters(self):
        return sum([np.prod(K.get_value(w).shape) for w in self.model.trainable_weights])

    def summary(self):
        return self.model.summary()

    def call(self, inputs):
        return self.model(inputs)

class EarthQuakeTransformer(tf.keras.Model):
    def __init__(self,
                 input_dim,
                 filters=None,
                 kernelsizes=None,
                 resfilters=None,
                 reskernelsizes=None,
                 lstmfilters=None,
                 attention_width=3,
                 dropout=0.0,
                 transformer_sizes=None,
                 kernel_regularizer=None,
                 classify=True,
                 pool_type='max',
                 att_type='additive',
                 activation='relu',
                 name='EarthQuakeTransformer'):
        """
        https://www.nature.com/articles/s41467-020-17591-w

        Example usage:
        import numpy as np
        test = np.random.random(size=(16,1024,3))
        detection = np.random.randint(2, size=(16,1024,1))
        p_arrivals = np.random.randint(2, size=(16,1024,1))
        s_arrivals = np.random.randint(2, size=(16,1024,1))

        model = EarthQuakeTransformer(input_dim=test.shape[1:])
        model.compile(optimizer='adam', loss=['binary_crossentropy',
                                            'binary_crossentropy',
                                            'binary_crossentropy'])

        model.fit(test, (detection,p_arrivals,s_arrivals))

        Args:
            input_dim (tuple): input size of the model.
            filters (list, optional): list of number of filters. Defaults to None.
            kernelsizes (list, optional): list of kernel sizes. Defaults to None.
            resfilters (list, optional): list of number of residual filters. Defaults to None.
            reskernelsizes (list, optional): list of residual filter sizes. Defaults to None.
            lstmfilters (list, optional): list of number of lstm filters. Defaults to None.
            attention_width (int, optional): width of attention mechanism. Defaults to 3. Use None for full. 
            dropout (float, optional): dropout. Defaults to 0.0.
            transformer_sizes (list, optional): list of sizes of attention layers. Defaults to [64, 64].
            kernel_regularizer (tf.keras.regualizers.Regualizer, optional): kernel regualizer. Defaults to None.
            classify (bool, optional): whether to classify phases or provide raw output. Defaults to True.
            att_type (str, optional): attention type. Defaults to 'additive'. 'multiplicative' is also supported. 
            name (str, optional): model name. Defaults to 'EarthQuakeTransformer'.

        """
        super(EarthQuakeTransformer, self).__init__(name=name)

        if filters is None:
            filters = [8, 16, 16, 32, 32, 64, 64]
        if kernelsizes is None:
            kernelsizes = [11, 9, 7, 7, 5, 5, 3]
        invfilters = filters[::-1]
        invkernelsizes = kernelsizes[::-1]
        if resfilters is None:
            resfilters = [64, 64, 64, 64, 64]
        if reskernelsizes is None:
            reskernelsizes = [3, 3, 3, 2, 2]
        if lstmfilters is None:
            lstmfilters = [16, 16]
        if transformer_sizes is None:
            transformer_sizes = [64,64]

        pool_layer = tfl.MaxPooling1D if pool_type == 'max' else tfl.AveragePooling1D

        def conv_block(f,kz):
            return tf.keras.Sequential([tfl.Conv1D(f, kz, padding='same', kernel_regularizer=kernel_regularizer),
                                        tfl.BatchNormalization(),
                                        tfl.Activation(activation),
                                        tfl.Dropout(dropout),
                                        pool_layer(4, strides=2, padding="same")])

        def block_BiLSTM(f, x):
            'Returns LSTM residual block'
            x = tfl.Bidirectional(tfl.LSTM(f, return_sequences=True))(x)
            x = tfl.Conv1D(f, 1, padding='same', kernel_regularizer=kernel_regularizer)(x)
            x = tfl.BatchNormalization()(x)
            return x        

        def _encoder():
            inp = tfl.Input(input_dim)
            def encode(x):
                for f, kz in zip(filters, kernelsizes):
                    x = conv_block(f, kz)(x)
                for f, kz in zip(resfilters, reskernelsizes):
                    x = ResnetBlock1D(f, 
                                      kz, 
                                      activation=activation,
                                      dropout=dropout, 
                                      kernel_regularizer=kernel_regularizer)(x)
                for f in lstmfilters:
                    x = block_BiLSTM(f, x)
                x = tfl.LSTM(f, return_sequences=True, kernel_regularizer=kernel_regularizer)(x)
                for ts in transformer_sizes:
                    x = TransformerBlock(num_heads=8, key_dim=ts, ff_dim=ts*4, rate=dropout)(x)
                return x
            return tf.keras.Model(inp, encode(inp))

        def inv_conv_block(f,kz):
            return tf.keras.Sequential([tfl.UpSampling1D(2),
                                        tfl.Conv1D(f, kz, padding='same', kernel_regularizer=kernel_regularizer),
                                        tfl.BatchNormalization(),
                                        tfl.Activation(activation),
                                        tfl.Dropout(dropout)])

        def _decoder(input_shape, attention=False, activation='sigmoid', output_name=None):
            inp = tfl.Input(input_shape)
            x = inp
            if attention:
                x = tfl.LSTM(filters[-1], 
                             return_sequences=True, 
                             kernel_regularizer=kernel_regularizer)(x)
                x = TransformerBlock(num_heads=8, ff_dim=filters[-1]*4, key_dim=filters[-1])(x)

            x = tf.keras.Sequential([inv_conv_block(f, kz) for f, kz in zip(invfilters, invkernelsizes)])(x)
            to_crop = x.shape[1] - input_dim[0]
            of_start, of_end = to_crop//2, to_crop//2
            of_end += to_crop % 2
            x = tfl.Cropping1D((of_start, of_end))(x)
            if activation is not None:
                x = tfl.Conv1D(1, 1, 
                               padding='same')(x)
                x = tfl.Activation(activation, 
                               name=output_name, 
                               dtype=tf.float32)(x)
            return tf.keras.Model(inp, x)

        self.feature_extractor = _encoder()
        encoded_dim = self.feature_extractor.layers[-1].output.shape[1:]
        
        self.detector = _decoder(encoded_dim, attention=False, activation='sigmoid' if classify else None, output_name='detection')
        self.p_picker = _decoder(encoded_dim, attention=True, activation='sigmoid' if classify else None, output_name='p_phase')
        self.s_picker = _decoder(encoded_dim, attention=True, activation='sigmoid' if classify else None, output_name='s_phase')
        
    @property
    def num_parameters(self):
        s = 0
        for m in [self.feature_extractor, self.detector, self.s_picker, self.p_picker]:
            s += sum([np.prod(K.get_value(w).shape) for w in m.trainable_weights])
        return s

    def call(self, inputs):
        encoded = self.feature_extractor(inputs)
        d = self.detector(encoded)
        p = self.p_picker(encoded)
        s = self.s_picker(encoded)
        return d, p, s


class TransPhaseNet(PhaseNet):
    def __init__(self,
                 num_classes=2,
                 filters=None,
                 kernelsizes=None,
                 output_activation='linear',
                 kernel_regularizer=None,
                 dropout_rate=0.2,
                 initializer='glorot_normal',
                 residual_attention=None,
                 pool_type='max',
                 att_type='across',
                 num_transformers=1,
                 rnn_type='lstm',
                 additive_att=True,
                 stacked_layer=4,
                 activation='relu',
                 name='TransPhaseNet'):
        """Adapted to 1D from https://keras.io/examples/vision/oxford_pets_image_segmentation/

        Args:
            num_classes (int, optional): number of outputs. Defaults to 2.
            filters (list, optional): list of number of filters. Defaults to None.
            kernelsizes (list, optional): list of kernel sizes. Defaults to None.
            residual_attention (list: optional): list of residual attention sizes, one longer that filters. 
            output_activation (str, optional): output activation, eg., 'softmax' for multiclass problems. Defaults to 'linear'.
            kernel_regularizer (tf.keras.regualizers.Regualizer, optional): kernel regualizer. Defaults to None.
            dropout_rate (float, optional): dropout. Defaults to 0.2.
            initializer (tf.keras.initializers.Initializer, optional): weight initializer. Defaults to 'glorot_normal'.
            name (str, optional): model name. Defaults to 'PhaseNet'.
            att_type (str, optional): if the attention should work during downstep or across (self attention). 
            rnn_type (str, optional): use "lstm" rnns or "causal" dilated conv.  
        """
        super(TransPhaseNet, self).__init__(num_classes=num_classes, 
                                              filters=filters, 
                                              kernelsizes=kernelsizes, 
                                              output_activation=output_activation, 
                                              kernel_regularizer=kernel_regularizer, 
                                              dropout_rate=dropout_rate,
                                              pool_type=pool_type, 
                                              activation=activation, 
                                              initializer=initializer, 
                                              name=name)
        self.att_type = att_type
        self.rnn_type = rnn_type
        self.stacked_layer = stacked_layer
        self.additive_att = additive_att
        self.num_transformers = num_transformers
            
        if residual_attention is None:
            self.residual_attention = [16, 16, 16, 16]
        else:
            self.residual_attention = residual_attention
    
    def _down_block(self, f, ks, x):
        x = ResnetBlock1D(f, 
                        ks, 
                        activation=self.activation, 
                        dropout=self.dropout_rate)(x)    
        x = self.pool_layer(4, strides=2, padding="same")(x)
        return x
    
    def _up_block(self, f, ks, x):
        x = ResnetBlock1D(f, 
                        ks, 
                        activation=self.activation, 
                        dropout=self.dropout_rate)(x)
        x = tfl.UpSampling1D(2)(x)
        return x

    def _att_block(self, x, y, ra):
        if self.rnn_type == 'lstm':
            x = tfl.Bidirectional(tfl.LSTM(ra, return_sequences=True))(x)
        elif self.rnn_type == 'causal':
            x1 = ResidualConv1D(ra, 3, stacked_layer=self.stacked_layer, causal=True)(x)
            x2 = ResidualConv1D(ra, 3, stacked_layer=self.stacked_layer, causal=True)(tf.reverse(x, axis=[1]))
            x = tf.concat([x1, tf.reverse(x2, axis=[1])], axis=-1)
        else:
            raise NotImplementedError('rnn type:' + self.rnn_type + ' is not supported')
        x = tfl.Conv1D(ra, 1, padding='same')(x)
        
        att = TransformerBlock(num_heads=8,
                               key_dim=ra,
                               ff_dim=ra*4,
                               rate=self.dropout_rate)([x,y])
        if self.num_transformers > 1:
            for _ in range(1, self.num_transformers):
                att = TransformerBlock(num_heads=8,
                            key_dim=ra,
                            ff_dim=ra*4,
                            rate=self.dropout_rate)(att)
        
        return att

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape[1:])
        
        # Entry block
        
        x = ResnetBlock1D(self.filters[0], 
                          self.kernelsizes[0], 
                          activation=self.activation, 
                          dropout=self.dropout_rate)(inputs)

        skips = [x]
        
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for i in range(1, len(self.filters)):
            x = self._down_block(self.filters[i], self.kernelsizes[i], x)
            if self.residual_attention[i] > 0 and self.att_type == 'downstep':
                att = self._att_block(x, skips[-1], self.residual_attention[i])
                if self.additive_att:
                    x += att
                else:
                    x = crop_and_add(x, att)
                    x = tfl.Conv1D(self.filters[i], 1, padding='same')(x)
            skips.append(x)

        if self.residual_attention[-1] > 0:
            att = self._att_block(x, x, self.residual_attention[-1])
            if self.additive_att:
                x = crop_and_add(x, att)
            else:
                x = crop_and_concat(x, att)
                x = tfl.Conv1D(self.filters[-1], 1, padding='same')(x)

        self.encoder = tf.keras.Model(inputs, x)
        ### [Second half of the network: upsampling inputs] ###
        
        for i in range(1, len(self.filters)):
            x = self._up_block(self.filters[::-1][i], self.kernelsizes[::-1][i], x)
            
            if self.residual_attention[::-1][i] > 0 and self.att_type == 'across':
                att = self._att_block(skips[::-1][i], skips[::-1][i], self.residual_attention[::-1][i])
                if self.additive_att:
                    x = crop_and_add(x, att)
                else:
                    x = crop_and_concat(x, att)
                    x = tfl.Conv1D(self.filters[::-1][i], 1, padding='same')(x)

        to_crop = x.shape[1] - input_shape[1]
        if to_crop != 0:
            of_start, of_end = to_crop // 2, to_crop // 2
            of_end += to_crop % 2
            x = tfl.Cropping1D((of_start, of_end))(x)
        
        #Exit block
        x = tfl.Conv1D(self.filters[0], 
                       self.kernelsizes[0],
                       strides=1,
                       kernel_regularizer=self.kernel_regularizer,
                       padding="same",
                       name='exit')(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)

        # Add a per-pixel classification layer
        if self.num_classes is not None:
            x = tfl.Conv1D(self.num_classes,
                           1,
                           padding="same")(x)
            outputs = tfl.Activation(self.output_activation, dtype='float32')(x)
        else:
            outputs = x

        # Define the model
        self.model = tf.keras.Model(inputs, outputs)

    @property
    def num_parameters(self):
        return sum([np.prod(K.get_value(w).shape) for w in self.model.trainable_weights])

    def summary(self):
        return self.model.summary()

    def call(self, inputs):
        return self.model(inputs)

