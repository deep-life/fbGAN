import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Input, Dense, Reshape, ReLU, Permute
from tensorflow.keras.layers import Conv1D, Input, LSTM, Embedding, Dense, TimeDistributed, Bidirectional, \
    LayerNormalization
from tensorflow.keras.models import Model
from globals import *


def softmax(logits):
    shape = tf.shape(logits)
    res = tf.nn.softmax(tf.reshape(logits, [-1, N_CHAR]))
    return tf.reshape(res, shape)


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.relu = ReLU()
        self.conv1d_1 = Conv1D(filters=DIM, kernel_size=KERNEL_SIZE, padding='same', strides=1, activation='relu')
        self.conv1d_2 = Conv1D(filters=DIM, kernel_size=KERNEL_SIZE, padding='same', strides=1)

    def __call__(self, X, alpha=0.3):
        x = self.relu(X)
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        return x + alpha * x


class Generator(tf.keras.Model):

    def __init__(self):
        """
        Implementation of the Generator.
        :param input_size: size of the sequence (input noise)
        """
        super(Generator, self).__init__(name='generator')

        self.model = tf.keras.models.Sequential()
        self.model.add(Input(shape=(NOISE_SHAPE,), batch_size=BATCH_SIZE))
        self.model.add(Dense(units=DIM * SEQ_LENGTH))
        self.model.add(Reshape((SEQ_LENGTH, DIM)))

        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())

        self.model.add(Conv1D(filters=N_CHAR, kernel_size=1))

    def call(self, inputs):
        x = self.model(inputs)
        x = softmax(x)
        return x


class Discriminator(tf.keras.Model):

    def __init__(self, clip=1):
        """
        Implementation of Discriminator
        :param clip: value to which you clip the gradients (or False)
        """
        super(Discriminator, self).__init__(name='discriminator')

        self.model = tf.keras.models.Sequential()
        self.model.add(Input(shape=(SEQ_LENGTH, N_CHAR), batch_size=BATCH_SIZE))
        self.model.add(Conv1D(filters=DIM, kernel_size=1))

        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())

        self.model.add(Reshape((-1, DIM * SEQ_LENGTH)))
        self.model.add(Dense(units=DIM * SEQ_LENGTH))
        self.model.add(Dense(units=1))

    def call(self, inputs, training=False):
        """
        model's forward pass
        :param X: input of the size [batch_size, seq_length];
        :param training: specifies the behavior of the call;
        :return: Y: probability of each sequences being real of shape [batch_size, 1]
        """
        x = self.model(inputs)
        return x


class Feedback():
    def __init__(self):
        input = Input(shape=(MAX_LEN_PROTEIN,))
        x = Embedding(input_dim=n_words, output_dim=128, input_length=MAX_LEN_PROTEIN)(input)
        x = LayerNormalization()(x)
        x = Bidirectional(LSTM(units=128, return_sequences=True, use_bias=True))(x)
        x = Bidirectional(LSTM(units=128, return_sequences=True, use_bias=True))(x)
        x = Bidirectional(LSTM(units=128, use_bias=True))(x)
        y = Dense(n_tags, activation="sigmoid")(x)
        self.model = Model(input, y)

    def train(self, OPTIM="rmsprop", LOSS='binary_crossentropy', BATCH_SIZE=128, EPOCHS=5):
        self.model.compile(optimizer=OPTIM, loss=LOSS, metrics=[tf.keras.metrics.Precision(),
                                                                tf.keras.metrics.Recall(),
                                                                tf.keras.metrics.Hinge()])
        history = self.model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                                 validation_data=(X_test, y_test), verbose=1)
        self.model.save(save_feedback)
        return history
