import numpy as np
import tensorflow as tf
import keras
import csv
from gan import GAN
from globals import *
from utils.protein_utilities import DNA_to_protein
from utils.data_utilities import triplets, OneHot_Seq
from models import Feedback


class FB_GAN():

    def __init__(self, features=DESIRED_FEATURES, generator_path=None, discriminator_path=None,
                 fbnet_path=None, multip_factor=20, log_id=None):
        """
        Parameters
        ----------
        generator_path: str (optional)
            Path to the weights of a pretrained generator.
        discriminator_path: str (optional)
             Path to the weights of a pretrained generator.
        fbnet_path: str (optional)
            Path to the saved model.
        features: list (optional)
            Features for which to optimize sequences
        multip_factor: int (optional)
            Factor indicating how many times best sequences are added to the discriminator at each step.
        log_id: str (optional)
            If log_id is provided, history logs during training will be written into "./Experiments/Experiment_{log_id}"
            else, the history is logged into self.history
        """

        self.GAN = GAN(generator_weights_path=generator_path, discriminator_weights_path=discriminator_path)
        self.FBNet = Feedback(fbnet_path)
        self.tokenizer = self.FBNet.tokenizer
        self.label_order = np.array(['B', 'C', 'E', 'G', 'H', 'I', 'S',
                                     'T'])  # order of labels as output by Multilabel binarizer - don't change!
        self.desired_features = features
        self.multip_factor = multip_factor
        self.data = None
        self.OneHot = OneHot_Seq(letter_type=TASK_TYPE)
        self.id = log_id
        if self.id:
            # If experiment is getting logged, initialize files to log it
            self.log_initialize()
        else:
            self.history = {'D_loss': [], 'G_loss': [], 'average_score': [], 'best_score': [], 'percent_fake': []}

    def get_scores(self, inputs):
        # convert the DNA sequences to protein sequences
        protein_sequence = DNA_to_protein(inputs)
        input_grams = triplets(protein_sequence)
        transformed = self.tokenizer.texts_to_sequences(input_grams)
        transformed = keras.preprocessing.sequence.pad_sequences(transformed, maxlen=MAX_LEN, padding='post')

        # use FBNet to grade the sequences
        scores = self.FBNet.model.predict(transformed)
        return scores

    def get_score_per_feature(self, scores):
        scores = np.array(scores)
        avg_scores = np.rint(100 * np.mean(scores, axis=0))
        score_per_feature = []
        for feature in self.desired_features:
            i = int(np.where(feature == self.label_order)[0])
            try:
                score_i = int(avg_scores[i])
            except:
                score_i = 0
            fscore = [feature, score_i]
            score_per_feature.append(fscore)

        return score_per_feature

    def add_samples(self, generated, scores, score_threshold=0.75, replace=False):
        best_index = scores > score_threshold
        best_samples = []
        best_scores = []
        for i in range(len(best_index)):
            passed_threshold = set(self.label_order[best_index[i]])
            if set(self.desired_features).issubset(passed_threshold):
                best_samples.append(generated[i])
                best_scores.append(scores[i])

        if replace:
            pass
        else:
            if len(best_samples) != 0:
                best_samples = np.repeat(best_samples, self.multip_factor, axis=0)
                self.data = np.concatenate((self.data, np.array(best_samples)), axis=0)

        return best_samples, best_scores

    def train(self, inputs, epochs, step_log=50, batch_size=264, steps_per_epoch=None,):
        """

        Parameters
        ----------
        inputs: ndarray
            Real one-hot encoded data of shape = (N, N_CHAR, SEQ_LEN)
        epochs: int (optional)
            number of epochs for which to train
        step_log: int (optional)
            specifies how often steps should be logged into the history or to the external file
        steps_per_epoch: int (optional)
            specifies how many steps per each epoch to take
            if None steps_per_epoch whole dataset will be used for each epoch (step_per_log = len(inputs)//batch_size)
        batch_size: int (optional)
            batch size
        -------

        """
        self.data = inputs
        self.batch_size = batch_size

        for epoch in range(epochs):

            print(f'Epoch {epoch} / {epochs}')
            dataset = self.create_dataset(self.data)
            step = 0

            for sample_batch in dataset:

                if step == steps_per_epoch:
                    break

                G_loss = self.GAN.G_train_step()
                D_loss, GP = self.GAN.D_train_step(sample_batch)

                generated = self.GAN.generate_samples(number=BATCH_SIZE, decoded=False)
                decoded_generated = self.OneHot.onehot_to_seq(generated)
                scores = self.get_scores(decoded_generated)
                generated = tf.cast(generated, tf.float32)
                best_samples, best_scores = self.add_samples(generated, scores)

                if step % step_log == 0:
                    print(
                        f'\tStep {step}:   Generator: {G_loss.numpy()}   Discriminator: {D_loss.numpy()}   Samples: {len(self.data)}')

                    print('\tBest scores per feature: ', end=' ')
                    best_per_feature = self.get_score_per_feature(best_scores)
                    pprint = [f'{sc[0]}: {sc[1]}%' for sc in best_per_feature]
                    print(*pprint, sep=' ')

                    print('\tAverage scores per feature: ', end=' ')
                    average_per_feature = self.get_score_per_feature(scores)
                    pprint = [f'{sc[0]}: {sc[1]}%' for sc in average_per_feature]
                    print(*pprint, sep=' ')

                    percent_fake = int(((len(self.data) - len(inputs)) / len(self.data)) * 100)

                    if self.id:
                        self.log_train(best_per_feature, average_per_feature, G_loss, D_loss, percent_fake)

                    else:
                        self.history['D_loss'].append(D_loss.numpy())
                        self.history['G_loss'].append(G_loss.numpy())
                        self.history['best_score'].append(best_per_feature)
                        self.history['average_score'].append(average_per_feature)

                step += 1

            if not self.id:
                self.history['percent_fake'].append(percent_fake)

            print(f'\tPercent of the fake samples in the discriminator: {percent_fake}%.')

        if self.id:
            # If you are in a log mode - generate resulting sequences. Store them in Experiments folder
            exp_folder = os.path.join(ROOT_PATH, "Experiments/Experiment {}".format(self.id))
            with open(exp_folder + "/seq_after.txt".format(self.id), 'w+') as f:
                for line in self.GAN.generate_samples(number=100, decoded=True):
                    f.write(line[0])
                    f.write("\n")

    def create_dataset(self, inputs):
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.shuffle(inputs.shape[0], seed=0).batch(self.batch_size, drop_remainder=True)
        return dataset

    def log_train(self, best, average, g_loss, d_loss, fake):
        """

        Parameters
        ----------
        best:
            best scores per feature computed during training
        average
            average scores per feature computed during training
        g_loss
            loss of generator
        d_loss
            loss of discriminator
        fake
            percent of fake sequences

        Returns
        -------
        None
            write to "Experiment/Experiment_{id} " folder

        """

        exp_folder = os.path.join(ROOT_PATH, "Experiments/Experiment_{}".format(self.id))
        with open(exp_folder + "/GAN_loss.csv", 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(
                [g_loss.numpy(), d_loss.numpy(), fake])

        with open(exp_folder + "/Average_Scores.csv".format(self.id), 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([sc[1] for sc in average])

        with open(exp_folder + "/Best_Scores.csv".format(self.id), 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([sc[1] for sc in best])

    def log_initialize(self):
        exp_folder = os.path.join(ROOT_PATH, "Experiments/Experiment_{}".format(self.id))
        os.makedirs(exp_folder)
        with open(exp_folder + "/seq_before.txt".format(self.id), 'w+') as f:
            for line in self.GAN.generate_samples(number=100, decoded=True):
                f.write(line[0])
                f.write("\n")
        with open(exp_folder + "/GAN_loss.csv".format(self.id), 'w+') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['g_loss', 'd_loss', 'percent_fake'])
        with open(exp_folder + "/Best_Scores.csv".format(self.id), 'w+') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([x for x in self.desired_features])
        with open(exp_folder + "/Average_Scores.csv".format(self.id), 'w+') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([x for x in self.desired_features])
