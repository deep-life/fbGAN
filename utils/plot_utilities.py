import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import numpy as np
import json


#Plot style
plt.style.use('default')
sns.set_context("paper")
sns.set_palette("husl", 5)


# Set the globals for font
rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':9})
rc('mathtext',**{'default':'regular'})


"""
uncomment when creating plots in python notebook
"""
#%config InlineBackend.figure_format = 'retina'


def plot_history(history,name):
    fig, axs = plt.subplots(2,figsize=(4,6))
    plt.subplots_adjust(hspace=0.3)
    fig.suptitle(name)

    axs[0].plot(history.history['categorical_accuracy'])
    axs[0].plot(history.history['val_categorical_accuracy'])
    axs[0].set_title('model accuracy')
    axs[0].set_ylabel('accuracy')
    axs[0].set_xlabel('epoch')
    #axs[0].hlines(0.9,0,len(history.history['categorical_accuracy']), linestyle='--', linewidth=1, label="0.9")
    axs[0].legend(['train', 'test'], loc='upper left')

    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('model loss')
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('epoch')
    #axs[1].hlines(0.1, 0,len(history.history['categorical_accuracy']), linestyle='--', linewidth=1, label="0.1")
    axs[1].legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_fbGAN_scores(features, scores, best_scores=None, save_path = None, epochs=None):

    colors = ['navy', 'darkmagenta', 'green', 'gold', 'salmon', 'silver', 'indianred', 'darkolive']
    fig, ax = plt.subplots()
    lines = []

    if epochs:
        interv = int(len(scores) / epochs)

    for i, feature in enumerate(features):
        scr = scores[:, i].astype(float)
        if epochs:
            scr = [np.mean(scr[e * interv:(e + 1) * interv]) for e in range(epochs)]
        l1 = ax.plot(np.arange(len(scr)), scr, label=feature, color=colors[i])

        if best_scores is not None:
            best = best_scores[:, i].astype(float)
            if epochs:
                best = [np.mean(best[e * interv:(e + 1) * interv]) for e in range(epochs)]
            l2 = ax.plot(np.arange(len(best)), best, color=colors[i], linestyle=':', alpha=0.4)
            if i == 0:
                lines.append([l1, l2])

    if best_scores is not None:
        legend_dash = ax.legend(np.array(lines).ravel(), ['average', 'best'], bbox_to_anchor=(1, 1))
        plt.gca().add_artist(legend_dash)

    plt.xlabel('Steps')
    if epochs:
        plt.xlabel('Epochs')

    plt.legend(bbox_to_anchor=(1, 0.75))
    plt.ylabel('Score (%)')
    plt.title('Score history')

    if save_path:
        plt.savefig(os.path.join(save_path,'scores.png'))
    plt.show()


def plot_gan_loss(g_loss, d_loss, save_path=None, epochs=None):
    if epochs:
        interv = int(len(g_loss) / epochs)
        d_loss = [np.mean(d_loss[e * interv: (e + 1) * interv]) for e in range(epochs)]
        g_loss = [np.mean(g_loss[e * interv: (e + 1) * interv]) for e in range(epochs)]

    plt.plot(np.arange(len(d_loss)), d_loss, label='Discriminator loss', color='navy')
    plt.plot(np.arange(len(g_loss)), g_loss, label='Generator loss', color='darkmagenta')
    plt.ylabel('Loss')
    plt.xlabel(f'Steps')
    plt.legend()
    if save_path:
        save_path = os.path.join(save_path, 'gan_loss.png')
        plt.savefig(save_path)

    if epochs:
        plt.xlabel(f'Epochs')
    plt.show()



def analyze_experiment(path):
    params_path = os.path.join(path, 'Parameters.txt')
    with open(params_path) as json_file:
        parameters = json.load(json_file)
    print(parameters)

    epochs = parameters['epochs']
    features = parameters['desired features']

    gan_loss = pd.read_csv(os.path.join(path, 'GAN_loss.csv'))

    g_loss = gan_loss['g_loss']
    d_loss = gan_loss['d_loss']
    p_fake = gan_loss['percent_fake']

    plot_gan_loss(g_loss, d_loss, epochs=epochs, save_path=EX_PATH)
    plot_gan_loss(g_loss, d_loss, epochs=None, save_path=EX_PATH)

    avg_score = pd.read_csv(os.path.join(path, 'Average_Scores.csv')).to_numpy()
    best_score = pd.read_csv(os.path.join(path, 'Best_Scores.csv')).to_numpy()

    plot_fbGAN_scores(features, save_path=EX_PATH, scores=avg_score, best_scores=best_score, epochs=epochs)


