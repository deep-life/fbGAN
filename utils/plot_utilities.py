import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import numpy as np


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


def plot_fbGAN_scores(scores, best_scores=None, epochs=None):
    """
    Plots lines for each featuer in scores array. If epochs is provided, steps in each epoch are averaged out.
    For example, if there are 300 steps total and there was 10 epochs, moving interval of 10 will be averaged
    and the final sequence is of length 30.

    If best scores are provided, plotst them in dashed lines on the same graph.

    :param scores: (ndarray, list) Output of the fbGAN log with feature and score per step.  For example:
                                    np.array( [[['C', 70],['H', 30],['E', 10]],
                                               [['C', 97],['H', 71],['E', 50]])
    :param best_scores: (ndarray, list) Output of the fbGAN log with feature and score per step.
    :param epochs: Number of epochs for which to average the steps.
    :return:
        None
    """
    colors = np.array(['navy', 'darkmagenta', 'green', 'gold', 'salmon', 'silver', 'indianred', 'darkolive'])
    features = scores[0, :, 0]
    fig, ax = plt.subplots()
    lines = []

    if epochs:
        interv = int(len(scores) / epochs)

    for i, feature in enumerate(features):
        scr = scores[:, i, 1].astype(float)
        if epochs:
            scr = [np.mean(scr[e * interv:(e + 1) * interv]) for e in range(epochs)]
        l1 = ax.plot(np.arange(len(scr)), scr, label=feature, color=colors[i])

        if best_scores is not None:
            best = best_scores[:, i, 1].astype(float)
            if epochs:
                best = [np.mean(best[e * interv:(e + 1) * interv]) for e in range(epochs)]
            l2 = ax.plot(np.arange(len(best)), best, color=colors[i], linestyle=':', alpha=0.4)
            if i == 0:
                lines.append([l1, l2])

    if best_scores is not None:
        legend_dash = ax.legend(np.array(lines).ravel(), ['average', 'best'], bbox_to_anchor=(1, 1))
        plt.gca().add_artist(legend_dash)

    if epochs:
        plt.xlabel('Epochs')
    else:
        plt.xlabel('Steps')

    plt.legend(bbox_to_anchor=(1, 0.75))
    plt.ylabel('Score (%)')
    plt.title('Score history')
    plt.show()