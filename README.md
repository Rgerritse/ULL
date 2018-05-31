# UNSUPERVISED LANGUAGE LEARNING
Matthew van Rijn<br>
Ruben Gerritse

### Practical 1: Evaluating Word Representations
To run: navigate to Project 1 and open main.ipynb.

### Practical 2: Learning Word Representations
The three main models can be trained from different files:<br />
SG.py for Skip-gram<br />
BSG.py for Bayesian Skip-gram<br />
EA.py for Embed-Align

The scripts are designed to be run interactively, but do not have to be.

The remaining files contain the following:<br />
evaluate.py: code for performing the lexical substitution task<br />
utils.py: code primarily for reading the datasets<br />
stopwords_language: stopwords to map to unknown

The expected data locations are:
data/hansards/ for the hansards dataset
data/lst for the lst dataset and evaluation script

### Practical 3: Evaluating Sentence Representations
The skip-gram model is the same as for practical 2, with some no longer needed code removed.<br>
The gensim model is trained as follows:<br>

~~~~
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
model = Word2Vec(LineSentence('data/europarl/training.en'), window=5, sg=1, min_count=5, hs=0, negative=15)
model.wv.save_word2vec_format('embeddings.txt')
~~~~
