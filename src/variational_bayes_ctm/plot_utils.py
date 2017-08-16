import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_pp_train_per(x, inspectors, models):
    sns.set()
    for insp in inspectors:
        plt.plot(x * 100, insp, lw=2)
        plt.xlabel("% of observed documents")
        plt.ylabel("predictive perplexity")
    plt.legend(models)
    plt.show()
