import matplotlib.pyplot as plt
import numpy as np
from pylab import MaxNLocator
import os


def gen_batches(n, batch_size):
    start = 0
    list_ranges = list()
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        list_ranges.append(list(range(start, end)))
        start = end
    if start < n:
        list_ranges.append(list(range(start, n)))
    return list_ranges


def plot_pp_train_per(x, inspectors, models):
    markers = ['x', '.', '*']
    for insp, marker in zip(inspectors, markers):
        plt.plot(x * 100, insp, lw=1, marker=marker)
        plt.xlabel("% of observed documents")
        plt.ylabel("log-likelihood per word")
    plt.legend(models)
    plt.show()


def plot_pp_topics(topics, inspectors, models):
    markers = ['x', '.', '*']
    for insp, marker in zip(inspectors, markers):
        plt.plot(topics, insp, lw=1, marker=marker)
        plt.xlabel("Number of topics")
        plt.ylabel("log-likelihood per word")
    plt.legend(models)
    plt.show()


def plot_precision_recall(precision, recall, average_precision):
    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AUC={0:0.2f}'
              .format(average_precision["micro"]))
    plt.show()


def plot_accuracy_topics(topics, inspectors, sds, models):
    plt.figure()
    lss = ['--', ':']
    for insp, sd, ls in zip(inspectors, sds, lss):
        (_, caps, _) = plt.errorbar(topics, insp, yerr=sd, marker='.', ls=ls, capsize=5, lw=1)
        for cap in caps:
            cap.set_markeredgewidth(1)
        plt.xlabel("Number of Topics")
        plt.ylabel("Classification Accuracy")
    plt.ylim(0.2, 0.7)
    plt.legend(models)
    plt.show()


def plot_convergence_time(times, inspectors, models, lim):
    markers = ['+', '*']
    plt.figure()
    for insp, time, marker in zip(inspectors, times, markers):
        plt.plot(np.cumsum(time), insp, lw=1, marker=marker)
    plt.xlabel("time in seconds")
    plt.ylabel("heldout log-likelihood per word")
    plt.xlim(0, lim)
    plt.legend(models)
    plt.show()


def plot_convergence_epochs(nb_iterations, inspectors, models):
    markers = ['o', 's']
    iterations = np.arange(nb_iterations) + 1
    fig = plt.figure()
    for insp, marker in zip(inspectors, markers):
        ax = fig.add_subplot(111)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(iterations, insp, lw=1, marker=marker, markerfacecolor='None')
        plt.xlabel("number of passes")
        plt.ylabel("heldout log-likelihood per word")
    plt.legend(models)
    plt.show()


def plot_convergence_iterations(stoch, batch, models, nb_documents, batch_size):
    nb_mini_batches = len(gen_batches(nb_documents, batch_size))
    mini_batch_ll = [iter for epoch in stoch for iter in epoch[:-1]]
    epoch_ll = [epoch[-1:][0] for epoch in stoch]
    mini_batch_iter = [(idx + 1) + np.arange(5, nb_mini_batches, 5) / nb_mini_batches
                       for idx, epoch in enumerate(stoch)]
    flat_mini_batch_iter = [iter for epoch in mini_batch_iter for iter in epoch]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(flat_mini_batch_iter, mini_batch_ll, lw=1, marker='+')
    plt.plot(np.arange(1, len(batch) + 1) + 1, epoch_ll, marker='o', linestyle='None', markersize=4)
    plt.plot(np.arange(1, len(batch) + 1) + 1, batch, lw=1, marker='*', markerfacecolor='None')
    plt.legend(models)
    plt.xlabel("number of passes")
    plt.ylabel("heldout log-likelihood per word")
    plt.show()


def results_cvi(de_news=True, plt_epochs=False, plt_iterations=True, plt_conv_time=False):
    if de_news:
        print("De-News dataset")
        results_directory = "../../results/de-news"
        stoch_file = os.path.join(results_directory, "conv_stoch_cvi_B=140_OF=64_D=6_S=7_every5.txt")
        batch_file = os.path.join(results_directory, "conv_cvi_S=7.txt")
        red = 40
        xlim = 5700
        nb_documents = 6863
        batch_size = 140
    else:
        print("20 Newsgroups dataset")
        results_directory = "../../results/20_news_groups"
        stoch_file = os.path.join(results_directory, "conv_stoch_cvi_B=150_OF=10_D=7_S=7_every5.txt")
        batch_file = os.path.join(results_directory, "conv_cvi_S=7.txt")
        red = 20
        xlim = 5500
        nb_documents = 7684
        batch_size = 150

    if plt_epochs:
        models = ["CTM_STOCH_CVI", "CTM_CVI"]
        ll_s = [-9.3905869510306399, -8.9783825280745404, -8.7654311894691403, -8.6639217244632292, -8.5948068441554888,
                -8.5390325825889644, -8.4910349970142036, -8.4482733737782301, -8.4093731877038032, -8.3736070065478447,
                -8.3402398914212927, -8.3090742341837469, -8.2800454417708771, -8.2532505311449054, -8.2284474155872758,
                -8.2047647572635931, -8.2406521986845309, -8.1846585082613785, -8.1378356702777506, -8.1179902170491491]
        ll = [-9.9769332657824865, -9.672278968789092, -9.3551317281003108, -9.214891679539484, -9.116656254438178,
              -9.0350000546440654, -8.9624206163590596, -8.8955941878679532, -8.8327227442298675, -8.772879001935646,
              -8.7152962053338516, -8.6594732029314443, -8.605581979720899, -8.5524350193367358, -8.499499508453237,
              -8.4469018627518349, -8.3942362392794383, -8.3425064282761046, -8.3278199501732697, -8.2493634562980613]
        plot_convergence_epochs(20, [ll_s, ll], models)

    stoch = np.loadtxt(stoch_file).tolist()
    batch, time_batch = np.loadtxt(batch_file)

    if plt_iterations:
        stoch_red = stoch[1:red]
        batch_red = batch[1:red].tolist()
        models = ["CTM_STOCH_CVI every 5 mini-batches", "CTM_STOCH_CVI after a full pass", "CTM_CVI"]
        plot_convergence_iterations(stoch_red, batch_red, models, nb_documents, batch_size)

    if plt_conv_time:
        vb_file = os.path.join(results_directory, "conv_vb.txt")
        vb, time_vb = np.loadtxt(vb_file)
        models = ["CTM_CVI", "CTM"]
        plot_convergence_time([time_batch, time_vb], [batch, vb], models, xlim)


if __name__ == "__main__":
    # Comparison between CTM, CVI and stochastic CVI
    results_cvi(de_news=True, plt_epochs=False, plt_iterations=True, plt_conv_time=False)
