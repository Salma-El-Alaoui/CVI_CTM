import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pylab import MaxNLocator
import os


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


def plot_convergence_iterations(stoch, batch, models):
    flat_epochs = [iter for epoch in stoch for iter in epoch]
    epoch_ll = [epoch[-1:][0] for epoch in stoch]
    iterations = [(idx + 1)/len(epoch) * np.arange(1, len(epoch) + 1) for idx, epoch in enumerate(stoch)]
    flat_iterations = [iter for epoch in iterations for iter in epoch]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(flat_iterations, flat_epochs, lw=1, marker='o', markerfacecolor='None')
    plt.plot(np.arange(len(batch)) + 1, epoch_ll, marker='o', linestyle='None')
    plt.plot(np.arange(len(batch)) + 1, batch, lw=1, marker='s', markerfacecolor='None')
    plt.legend(models)
    plt.xlabel("number of passes")
    plt.ylabel("heldout log-likelihood per word")
    plt.show()


def results_lda_ctm():
    models = ["CTM", "LDA"]

    test_ctm = [-10.344016397834826, -9.2053581737863919, -8.8249573482824655, -8.4925241672434559, -8.2968172657881869,
                -8.1419220706265509, -8.01474055832246, -8.0486872454118004, -8.0143044685818641]

    test_lda = [-9.6904686306093062, -9.6407050609641107, -9.5485135718130945, -9.4531017577657579, -9.3893403512526241,
                -9.358140637869635, -9.3558844865670956, -9.2661561741099732, -9.2425589114371469]

    # ======================= Classification Accuracy single split ======================#
    # ===================================================================================#

    accuracy_ctm = [0.22, 0.44, 0.49, 0.49, 0.51, 0.5, 0.49]

    accuracy_lda = [0.22, 0.34, 0.35, 0.37, 0.38, 0.37, 0.33]

    # ======================= Classification Accuracy with CV ======================#
    # ==============================================================================#

    topics = [20, 25, 30, 35, 40, 45, 50, 60]
    mean_lda = [0.31, 0.36, 0.37, 0.36, 0.38, 0.35, 0.36, 0.36]
    sd_lda = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02]
    mean_ctm = [0.46, 0.48, 0.47, 0.47, 0.49, 0.49, 0.50, 0.49]
    sd_ctm = [0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.001, 0.01]
    # plot_accuracy_topics(topics, [mean_ctm, mean_lda], [sd_ctm, sd_lda], models)

    # ======================= PP = f(number of topics) ======================#
    # ========================================================================#

    models = ["CTM_CVI", "CTM", "LDA"]

    # de-news
    topics = [5, 8, 10, 15, 20, 25, 30, 35, 40]
    ctm = [-7.580532598474275296e+00, - 7.560194590903261513e+00, - 7.602279183259960327e+00,
           - 7.659581682628681776e+00,
           - 6.253641959965173136e+00, - 6.184206010213144644e+00, - 6.114064321088665821e+00,
           - 6.122649375861390908e+00,
           - 6.058106228010641914e+00]
    lda = [-8.8657684869019189, -8.7491414829973504, -8.710538809781859, -8.6274111055055052, -8.4553626107178843,
           -8.3483223863977596,
           -8.2882130177387552, -8.1883907740478463, -8.1036552427066066]
    cvi = [-7.6892, -7.5974, -7.5417, -7.1740874522508324, -6.2740794017267962, -6.1054090344680256,
           -6.3753498789361069, -6.0971858211621948, -6.0607233944469527]

    plot_pp_topics(topics, [cvi, ctm, lda], models)

    # ======================= PP = f(%train) =======================#
    # ===============================================================#

    # 20-news
    train_sizes = np.asarray([0.1, 0.15, 0.2, 0.25, 0.3, 0.35])  # ,  0.4, 0.45, 0.5])
    ctm = [-9.031582651321196309e+00, -8.953745962642818412e+00, -8.679029202735506487e+00, -8.676770626309597390e+00,
           -8.463241156409551991e+00,
           -8.486064399501110245e+00]  # , -9.142707224616151152e+00, -9.128579067827756077e+00, -9.124728648318916058e+00]
    lda = [-10.749882566493323, -10.520613243129059, -10.32215594830998, -10.220704600022529, -10.124963194939893,
           -10.152993505602398]  # , -10.092290800352727, -10.088510595830661, -10.162138369372952]
    cvi = [-8.4172610442327436, -8.3530458734511708, -8.2275314264144388, -8.209207969427224, -8.245440111487369,
           -8.2011908800972328]

    # plot_pp_train_per(train_sizes, [cvi, ctm, lda], models)

    # AP
    train_sizes = np.asarray([0.1, 0.15, 0.2, 0.25, 0.3, 0.35])  # 0.4,   0.45,  0.5])
    ctm = [-7.355208321568674990e+00, -7.163444725373114608e+00, -7.047840999666488138e+00, -6.983631773165615897e+00,
           -6.937004033542123693e+00, -6.905019013962870069e+00]
    # -7.706969304332586646e+00, -7.707866040766431226e+00, -7.746350410289387867e+00]

    lda = [-8.7584331340407999, -8.5154937999609537, -8.3409690529532696, -8.2689974257455017, -8.206393353288794,
           -8.1696335750536662]  # -8.1516505909073462, -8.1200573259939333, -8.1307340717049907]

    cvi = [-6.7205426707492899, -6.6474289718167352, -6.6257758124872215, -6.6452026874455035,
           -6.6387637283509617, -6.6344638234716716]

    # plot_pp_train_per(train_sizes, [cvi, ctm, lda], models)

    # de-News
    train_sizes = np.asarray([0.1, 0.15, 0.2, 0.25, 0.3, 0.35])  # 0.4, 0.45, 0.5])
    ctm = [-7.003100397218386064e+00, - 6.801027884843848348e+00, - 6.745441049493919650e+00,
           - 6.711361331697173860e+00, - 6.666628685362040230e+00, - 6.638640527042508843e+00]
    # - 7.607168330262510381e+00, - 7.600665596859945694e+00, - 7.609458069902792943e+00]
    lda = [-8.5389442537837681, -8.3443683270632167, -8.203670290639911, -8.1095499115967389,
           -8.0948734577283794, -8.0804914558436565]
    # -8.0928187958842503, -8.0511905841909908, -8.0741773324908692]
    cvi = [-6.6330647105098199, -6.5400816920048941, -6.5472609428130628, -6.5824695252972445,
           -6.5595797793645891, -6.5117585633776809]
    # plot_pp_train_per(train_sizes, [cvi, ctm, lda], models)


if __name__ == "__main__":
    results_directory = "../../results/20_news_groups"
    models = ["stochastic CVI", "CVI"]
    ll_s = [-9.3905869510306399, -8.9783825280745404, -8.7654311894691403, -8.6639217244632292, -8.5948068441554888, -8.5390325825889644, -8.4910349970142036, -8.4482733737782301, -8.4093731877038032, -8.3736070065478447, -8.3402398914212927, -8.3090742341837469, -8.2800454417708771, -8.2532505311449054, -8.2284474155872758, -8.2047647572635931, -8.2406521986845309, -8.1846585082613785, -8.1378356702777506, -8.1179902170491491]
    ll = [-9.9769332657824865, -9.672278968789092, -9.3551317281003108, -9.214891679539484, -9.116656254438178, -9.0350000546440654, -8.9624206163590596, -8.8955941878679532, -8.8327227442298675, -8.772879001935646, -8.7152962053338516, -8.6594732029314443, -8.605581979720899, -8.5524350193367358, -8.499499508453237, -8.4469018627518349, -8.3942362392794383, -8.3425064282761046, -8.3278199501732697, -8.2493634562980613]
    # plot_convergence_epochs(20, [ll_s, ll], models)
    stoch_file = os.path.join(results_directory, "conv_stoch_cvi_B=150_OF=10_D=7_S=7.txt")
    stoch = np.loadtxt(stoch_file)[0:20, :].tolist()
    batch_file = os.path.join(results_directory, "conv_cvi_S=7.txt")
    batch, _ = np.loadtxt(batch_file)
    batch = batch[0:20].tolist()
    models = ["stochastic CVI", "stochastic CVI after full pass", "CVI"]
    plot_convergence_iterations(stoch, batch, models)




