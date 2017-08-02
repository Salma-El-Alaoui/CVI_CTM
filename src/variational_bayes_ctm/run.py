from corpus import ToyDataset
from ctm import CTM

if __name__ == "__main__":
    data = ToyDataset(ctm=True)
    ctm = CTM(corpus=data.doc_set, vocab=data.vocab, number_of_topics=data.K)
    for i in range(1000):
        ctm.em_step()