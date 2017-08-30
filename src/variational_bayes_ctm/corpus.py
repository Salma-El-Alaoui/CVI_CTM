import numpy as np
import string
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import re
import os
from sklearn.model_selection import train_test_split

class ToyDataset:

    def __init__(self, nb_topics=3, nb_documents=100, vocab_size=5, document_size=50, concentration=1/3, ctm=True):
        # Constants
        self.K = nb_topics
        self.D = nb_documents
        self.V = vocab_size
        self.N = document_size
        self.ctm = ctm
        self.concentration = concentration

        # hyper-parameters:
        self.mu = np.zeros(self.K)
        self.sigma = 0.1 * np.identity(self.K)
        self.gamma = concentration * np.ones(self.V)

        # Draw topic
        self.beta = np.random.dirichlet(alpha=self.gamma, size=self.K)

        # Draw topic proportion for each document
        if ctm:
            self.eta = np.random.multivariate_normal(mean=self.mu, cov=self.sigma, size=self.D)
            self.theta = np.exp(self.eta) / np.sum(np.exp(self.eta), axis=1)[:, None]
        else:
            self.theta_prior = concentration * np.ones(self.K)
            self.theta = np.random.dirichlet(alpha=self.theta_prior, size=self.D)

        # Draw topic assignment for each document and each word
        self.Z = np.asarray([np.random.multinomial(n=1, pvals=self.theta[d, :], size=self.N) for d in range(self.D)])

        # Draw Words:
        self.W = np.asarray(
            [[np.random.multinomial(n=1, pvals=self.beta[np.where(self.Z[d, n, :] == 1)[0].squeeze()])
                for n in range(self.N)]
                for d in range(self.D)])

        list_docs = [[string.ascii_lowercase[(np.where(self.W[d, n, :] == 1)[0].squeeze())]
                        for n in range(self.N)]
                        for d in range(self.D)]

        self.vocabulary = [string.ascii_lowercase[i] for i in range(vocab_size)]
        self.doc_set = [" ".join(d) for d in list_docs]


class ApDataset:
    def __init__(self, shuffle=True, random_state=0, train_size=0.9):
        input_directory = "../../data/ap"
        vocabulary_path = os.path.join(input_directory, 'vocabulary.txt')
        input_voc_stream = open(vocabulary_path, 'r')
        vocab = []
        for line in input_voc_stream:
            vocab.append(line.strip().lower().split()[0])
        self.vocabulary = list(set(vocab))
        self.shuffle = shuffle
        self.random_state = random_state
        self.train_size = train_size

        docs = []
        input_file = open(os.path.join(input_directory, 'doc.dat'), 'r')
        for line in input_file:
            docs.append(line.strip())
        print("successfully load all training documents...")

        self.doc_set_train, self.doc_set_test = train_test_split(docs, train_size=self.train_size,
                                                           random_state=self.random_state, shuffle=self.shuffle)


class NipsDataset:
    def __init__(self, shuffle=True, random_state=0, train_size=0.9):
        input_directory = "../../data/nips-abstract"
        vocabulary_path = os.path.join(input_directory, 'vocabulary.txt')
        input_voc_stream = open(vocabulary_path, 'r')
        vocab = []
        for line in input_voc_stream:
            vocab.append(line.strip().lower().split()[0])
        self.vocabulary = list(set(vocab))
        self.shuffle = shuffle
        self.random_state = random_state
        self.train_size = train_size

        self.doc_set_train = []
        input_file_train = open(os.path.join(input_directory, 'train.dat'), 'r')
        for line in input_file_train:
            self.doc_set_train.append(line.strip())

        self.doc_set_test = []
        input_file_test = open(os.path.join(input_directory, 'test.dat'), 'r')
        for line in input_file_test:
            self.doc_set_test.append(line.strip())

        print("successfully load all training documents...")

        #self.doc_set_train, self.doc_set_test = train_test_split(docs, train_size=self.train_size,
        #                                                   random_state=self.random_state, shuffle=self.shuffle)


class DeNewsDataset:

    def __init__(self, shuffle=True, random_state=0, train_size=0.9):
        input_directory = "../../data/de-news"
        vocabulary_path = os.path.join(input_directory, 'vocabulary.txt')
        input_voc_stream = open(vocabulary_path, 'r')
        vocab = []
        for line in input_voc_stream:
            vocab.append(line.strip().lower().split()[0])
        self.vocabulary = list(set(vocab))
        self.shuffle = True
        self.random_state = random_state
        self.train_size = train_size

        docs = []
        input_file = open(os.path.join(input_directory, 'doc.dat'), 'r')
        for line in input_file:
            docs.append(line.strip())
        print("successfully load all training documents...")

        self.doc_set_train, self.doc_set_test = train_test_split(docs, train_size=self.train_size,
                                                           random_state=self.random_state, shuffle=self.shuffle)


class NewsDataset:

    def __init__(self, n_samples=None, shuffle=True, random_state=0, train_size=0.8):
        input_directory = "../../data/20_news_groups"
        vocabulary_path = os.path.join(input_directory, 'vocabulary.txt')
        input_voc_stream = open(vocabulary_path, 'r')
        vocab = []
        for line in input_voc_stream:
            vocab.append(line.strip().lower().split()[0])
        self.vocabulary = list(set(vocab))
        self.shuffle = True
        self.random_state = random_state
        self.train_size = train_size

        dataset = fetch_20newsgroups(shuffle=shuffle, random_state=self.random_state, remove=('headers', 'footers', 'quotes'))
        if n_samples is None:
            self.n_samples = len(dataset.target)
        else:
            self.n_samples = n_samples

        self.categories = dataset.target[:n_samples]
        self.categories_names = dataset.target_names[:n_samples]
        self.raw_samples = dataset.data[:n_samples]
        self.targets = dataset.target[:n_samples]
        self.target_names = dataset.target_names[:n_samples]

        vectorizer = CountVectorizer(max_df=0.95, min_df=2, vocabulary=vocab, preprocessor=self.preprocessor)
        self.X = vectorizer.fit_transform(self.raw_samples)
        self.X_train, self.X_test, self.y_train, self.y_test =\
            train_test_split(self.X, self.targets, train_size=self.train_size, random_state=self.random_state)
        self.doc_set_train = [" ".join(d) for d in vectorizer.inverse_transform(self.X_train)]
        self.doc_set_test = [" ".join(d) for d in vectorizer.inverse_transform(self.X_test)]
        self.doc_train_list = [d.tolist() for d in vectorizer.inverse_transform(self.X_train)]
        self.doc_test_list = [d.tolist() for d in vectorizer.inverse_transform(self.X_test)]

    @staticmethod
    def preprocessor(doc):
        doc = doc.lower()
        doc = re.sub(r'-', ' ', doc)
        doc = re.sub(r'[^a-z ]', '', doc)
        doc = re.sub(r' +', ' ', doc)
        return doc


def remove_stop_words(input_directory="../../data/nips-abstract", old_vocab_file='voc.dat', new_vocab_file='vocabulary.txt'):
    ENGLISH_STOP_WORDS = frozenset([
        "a", "about", "above", "across", "after", "afterwards", "again", "against",
        "all", "almost", "alone", "along", "already", "also", "although", "always",
        "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
        "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
        "around", "as", "at", "back", "be", "became", "because", "become",
        "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
        "below", "beside", "besides", "between", "beyond", "bill", "both",
        "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
        "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
        "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
        "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
        "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
        "find", "fire", "first", "five", "for", "former", "formerly", "forty",
        "found", "four", "from", "front", "full", "further", "get", "give", "go",
        "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
        "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
        "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
        "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
        "latterly", "least", "less", "ltd", "made", "many", "may", "me",
        "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
        "move", "much", "must", "my", "myself", "name", "namely", "neither",
        "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
        "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
        "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
        "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
        "please", "put", "rather", "re", "same", "see", "seem", "seemed",
        "seeming", "seems", "serious", "several", "she", "should", "show", "side",
        "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
        "something", "sometime", "sometimes", "somewhere", "still", "such",
        "system", "take", "ten", "than", "that", "the", "their", "them",
        "themselves", "then", "thence", "there", "thereafter", "thereby",
        "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
        "third", "this", "those", "though", "three", "through", "throughout",
        "thru", "thus", "to", "together", "too", "top", "toward", "towards",
        "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
        "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
        "whence", "whenever", "where", "whereafter", "whereas", "whereby",
        "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
        "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
        "within", "without", "would", "yet", "you", "your", "yours", "yourself",
        "yourselves"])

    vocabulary_path = os.path.join(input_directory, old_vocab_file)
    input_voc_stream = open(vocabulary_path, 'r')
    vocab = []
    for line in input_voc_stream:
        vocab.append(line.strip().lower().split()[0])
    new_vocab = set(vocab) - ENGLISH_STOP_WORDS
    output_voc_stream = open(os.path.join(input_directory, new_vocab_file), 'w')
    for item in new_vocab:
        output_voc_stream.write("%s\n" % item)


if __name__ == "__main__":
    pass

