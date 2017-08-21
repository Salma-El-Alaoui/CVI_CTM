import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_pp_train_per(x, inspectors, models):
    sns.set()
    for insp in inspectors:
        plt.plot(x * 100, insp, lw=2)
        plt.xlabel("% of observed documents")
        plt.ylabel("log-predictive perplexity")
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

if __name__ == "__main__":
    test_ctm = [-10.344016397834826, -9.2053581737863919, -8.8249573482824655, -8.4925241672434559, -8.2968172657881869,
                -8.1419220706265509, -8.01474055832246, -8.0486872454118004, -8.0143044685818641]

    test_lda = [-9.6904686306093062, -9.6407050609641107, -9.5485135718130945, -9.4531017577657579, -9.3893403512526241,
                -9.358140637869635, -9.3558844865670956, -9.2661561741099732, -9.2425589114371469]

    plot_pp_train_per(np.linspace(0.2, 0.9, num=8), [test_ctm[1:], test_lda[1:]], ["CTM", "LDA"])