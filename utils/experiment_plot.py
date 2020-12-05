import plotly.graph_objs as go
from collections import OrderedDict
from itertools import chain
import pandas as pd
from plotly.offline import iplot
from plotly import tools
from collections import Counter
import re
from globals import *


def get_score_counts(id='trial'):
    exp_folder = os.path.join(ROOT_PATH, "Experiments/Experiment_{}/".format(id))
    with open(exp_folder + "seq_before.txt") as f:
        seq_before = f.read().splitlines()
    before = sum([Counter(x) for x in seq_before], Counter())
    with open(exp_folder + "seq_after.txt") as f:
        seq_after = f.read().splitlines()
    after = sum([Counter(x) for x in seq_after], Counter())
    try:
        del before['P']
        del after['P']
    except:
        pass
    return before, after


def get_score_distribution(id='trial'):
    exp_folder = os.path.join(ROOT_PATH, "Experiments/Experiment_{}/".format(id))
    with open(exp_folder + "seq_before.txt") as f:
        seq_before = f.read().splitlines()
    with open(exp_folder + "seq_after.txt") as f:
        seq_after = f.read().splitlines()
    properties = list(OrderedDict.fromkeys(chain.from_iterable(seq_before)))
    before = pd.DataFrame(columns=properties)
    after = pd.DataFrame(columns=properties)
    for line in seq_before:
        before = before.append({x: line.count(x) for x in properties}, ignore_index=True)
    for line in seq_after:
        after = after.append({x: line.count(x) for x in properties}, ignore_index=True)
    try:
        del before['P']
        del after['P']
    except:
        pass
    return before, after


def get_parameters(id):
    exp_folder = os.path.join(ROOT_PATH, "Experiments/Experiment_{}/".format(id))
    with open(exp_folder + "Parameters.txt") as f:
        parameters = f.read().splitlines()[0]
    parameters = ','.join([str(x) for x in re.findall(r"\{(.*?)\}", parameters, re.MULTILINE | re.DOTALL)])
    parameters = parameters.replace(',', '<br>')
    parameters = parameters.split(']')[-1]
    return parameters


def plot_experiment(website=False, id="trial", FONT="Didot"):
    """
    website - specify whether plots are for paper or website (font is different)
    id - the same id you used to log the experiment
    """
    exp_folder = os.path.join(ROOT_PATH, "Experiments/Experiment_{}/".format(id))
    losses = pd.read_csv(exp_folder + "GAN_loss.csv")
    best = pd.read_csv(exp_folder + "Best_Scores.csv")
    av = pd.read_csv(exp_folder + "Average_Scores.csv")
    parameters = get_parameters(id)
    steps = [_ for _ in range(len(losses.g_loss))]

    color_map = {'G': '#AC90CE', 'C': '#CB90CE', 'H': '#CE90B2', 'E': '#CE9093',
                 'I': '#CEAC90', 'B': '#CECB90', 'T': '#90C4CE', 'S': '#90A5CE'}

    # Initialize subplots
    fig = tools.make_subplots(rows=2, cols=2,
                              subplot_titles=("GAN loss", "Scores",
                                              "Property distribution", "Total count of properties"),
                              horizontal_spacing=0.1)
    # Generator Loss
    trace1 = go.Scatter(x=steps, y=losses.g_loss, name="Generator",
                        marker_color='#74b4d4', legendgroup='group1')
    fig.add_trace(trace1, row=1, col=1)
    # Discriminator Loss
    trace2 = go.Scatter(x=steps, y=losses.d_loss, name="Discriminator",
                        marker_color='#136394', legendgroup='group1')
    fig.add_trace(trace2, row=1, col=1)

    # Best scores
    for feature in best.columns:
        f_trace = go.Scatter(x=steps, y=list(best[feature].to_numpy()),
                             name=feature + ' best', line=dict(dash='dash'),
                             marker_color=color_map[feature], legendgroup='group2')
        fig.add_trace(f_trace, row=1, col=2)
    # Average scores
    for feature in av.columns:
        f_trace = go.Scatter(x=steps, y=list(av[feature].to_numpy()), name=feature + ' average',
                             marker_color=color_map[feature], opacity=0.5, legendgroup='group2')
        fig.add_trace(f_trace, row=1, col=2)
    # Annotate percent of fake sequences
    for j, perc in enumerate(losses.percent_fake):
        fig.add_annotation(x=j, y=losses.d_loss[j],
                           text=str(perc) + '%',
                           showarrow=False,
                           yshift=10, font=dict(color="#000000"))
    before, after = get_score_counts(id)
    # Count of features before and after training
    trace3 = go.Bar(x=list(before.keys()), y=list(before.values()),
                    name='Before training', legendgroup='group3', marker_color='#84C2DA')
    trace4 = go.Bar(x=list(after.keys()), y=list(after.values()),
                    name='After training', legendgroup='group3', marker_color='#90AFCE')
    fig.add_trace(trace3, row=2, col=2)
    fig.add_trace(trace4, row=2, col=2)

    # Distribution of features before and after training

    d_before, d_after = get_score_distribution(id)
    for feature in d_before.columns:
        trace_before = go.Box(
            y=d_before[feature], name=feature, showlegend=False,
            legendgroup='group3', marker_color='#84C2DA')
        trace_after = go.Box(
            y=d_after[feature], name=feature, showlegend=False,
            legendgroup='group3', marker_color='#90AFCE')
        fig.add_trace(trace_before, row=2, col=1)
        fig.add_trace(trace_after, row=2, col=1)

    fig.add_annotation(text='Parameters:<br>' + parameters,
                       align='left',
                       showarrow=False, xref='paper',
                       yref='paper',
                       x=1.2,
                       y=0)

    # set x axis
    fig.update_xaxes(title_text="Steps", row=1, col=1)
    fig.update_xaxes(title_text="Steps", row=1, col=2)
    fig.update_xaxes(title_text="Property", row=2, col=1)
    fig.update_xaxes(title_text="Property", row=2, col=2)
    # set y axis
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="%", row=1, col=2)
    fig.update_yaxes(title_text="Count per sequence", row=2, col=1)
    fig.update_layout(height=600, width=900,
                      title={'text': 'Experiment {}'.format(id), 'y': 0.9, 'x': 0.45},
                      font_family=FONT, title_font_family=FONT, paper_bgcolor='rgba(0,0,0,0)', boxmode='group')

    if website:
        fig.update_layout(font=dict(
            color="white"))

    iplot(fig)