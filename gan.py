import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from models import Generator, Discriminator
from utils.protein_utilities import *
from utils.data_utilities import *
from globals import *

# Adapted from https://github.com/igul222/improved_wgan_training/blob/master/gan_language.py 

class GAN:

    def __init__(self, batch_size=BATCH_SIZE, discriminator_steps=0, lr=0.0002,
                 gradient_penalty_weight=5, generator_weights_path=None, discriminator_weights_path=None):

        self.batch_size = batch_size
        self.G = Generator()
        self.D = Discriminator()

        self.d_steps = discriminator_steps

        self.history = {"G_loss": [], "D_loss": [], "gradient_penalty": [], "sequences": []}

        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
        self.D_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)

        self.gp_weight = gradient_penalty_weight
        self.step_log = None

        if generator_weights_path:
            self.G.load_weights(generator_weights_path)

        if discriminator_weights_path:
            self.D.load_weights(discriminator_weights_path)

    def generate_samples(self, number=None, decoded=False):
        if number is None:
            number = self.batch_size
        z = tf.random.normal([number, NOISE_SHAPE])
        generated = self.G(z)

        if decoded:
            OneHot = OneHot_Seq(letter_type=TASK_TYPE)
            generated = OneHot.onehot_to_seq(generated)

        return generated

    def generator_loss(self, fake_score):
        return -tf.math.reduce_mean(fake_score)

    def discriminator_loss(self, real_score, fake_score):
        return tf.math.reduce_mean(fake_score) - tf.math.reduce_mean(real_score)

    # @tf.function
    def gradient_penalty(self, real_samples, fake_samples):
        alpha = tf.random.normal([self.batch_size, 1, 1], 0.0, 1.0)
        real_samples = tf.cast(real_samples, tf.float32)
        diff = fake_samples - real_samples
        interpolated = real_samples + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.D(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    # @tf.function
    def G_train_step(self):
        with tf.GradientTape() as tape:
            fake_samples = self.generate_samples()
            fake_score = self.D(fake_samples, training=True)
            G_loss = self.generator_loss(fake_score)

        G_gradients = tape.gradient(G_loss, self.G.trainable_variables)
        self.G_optimizer.apply_gradients((zip(G_gradients, self.G.trainable_variables)))

        return G_loss

    # @tf.function
    def D_train_step(self, real_samples):
        with tf.GradientTape() as tape:
            fake_samples = self.generate_samples()
            real_score = self.D(real_samples, training=True)
            fake_score = self.D(fake_samples, training=True)

            D_loss = self.discriminator_loss(real_score, fake_score)
            GP = self.gradient_penalty(real_samples, fake_samples) * self.gp_weight
            D_loss = D_loss + GP

        D_gradients = tape.gradient(D_loss, self.D.trainable_variables)
        self.D_optimizer.apply_gradients((zip(D_gradients, self.D.trainable_variables)))

        return D_loss, GP

    def create_dataset(self, inputs):
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.shuffle(inputs.shape[0], seed=0).batch(self.batch_size, drop_remainder=True)
        return dataset

    def train(self, inputs, epochs, step_log=50):
        n_steps = len(self.create_dataset(inputs)) * epochs
        step = 0
        self.step_log = step_log

        # Pre-train discriminator
        print('Pretraining discriminator...')
        for step in range(self.d_steps):
            dataset = self.create_dataset(inputs)

            for sample_batch in dataset:
                self.D_train_step(sample_batch)

        # Train discriminator and generator
        for epoch in range(epochs):
            dataset = self.create_dataset(inputs)

            print(f"Epoch {epoch}/{epochs}:")

            for sample_batch in dataset:
                G_loss = self.G_train_step()
                D_loss, GP = self.D_train_step(sample_batch)

                if step % self.step_log == 0:
                    example_sequence = self.get_highest_scoring()
                    self.history["G_loss"].append(G_loss.numpy())
                    self.history["D_loss"].append(D_loss.numpy())
                    self.history['gradient_penalty'].append(GP.numpy())
                    self.history['sequences'].append(example_sequence)
                    print(
                        f'\t Step {step}/{n_steps} \t Generator: {G_loss.numpy()} \t Discriminator: {D_loss.numpy()} \t Sequence: {example_sequence}')
                step += 1

    def get_highest_scoring(self, num_to_generate=BATCH_SIZE, decoded=True):
        fake_samples = self.generate_samples(num_to_generate)
        fake_scores = self.D(fake_samples)
        best_indx = np.argmax(fake_scores)
        best_seq = fake_samples[best_indx].numpy()

        if decoded:
            OneHot = OneHot_Seq(letter_type=TASK_TYPE)
            best_seq = OneHot.onehot_to_seq(best_seq)

        return best_seq

    def plot_history(self):
        D_losses = np.array(self.history['D_loss'])
        G_losses = np.array(self.history['G_loss'])

        plt.plot(np.arange(D_losses.shape[0]), D_losses, label='Discriminator loss')
        plt.plot(np.arange(G_losses.shape[0]), G_losses, label='Generator loss')
        plt.ylabel('Loss')
        plt.xlabel(f'Steps (x{self.step_log})')
        plt.legend()

        plt.show()

    def show_sequences_history(self):
        sequences_history = self.history['sequences']
        print('History of top scoring generated sequences... \n')
        for i in range(len(sequences_history)):
            print(f'Step {i * self.step_log}: \t {sequences_history[i][0]}')
