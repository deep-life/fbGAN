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



def plot_fbGAN_history(history, scores):
    raise ValueError('Not finished implementing, please try again soon.')
    colors = ['navy', 'darkmagenta', 'green', 'gold', 'indianred','silver','olivedrab', 'sandybrown' ]

    scores = np.array(scores)
    features = scores[0,:,0]

    fig, ax = plt.subplots()

    for i,feature in enumerate(features):
        scr = scores[:, i,1].astype(float)

        ax.plot(np.arange(len(scores)), scr, label = f'{feature}: average score' , color = colors[i,0])

        ax.set_xlabel('Steps')
        ax.set_ylabel('Score (%)')
        ax.legend()
        ax.set_title('Score history')
    plt.show()