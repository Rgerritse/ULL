# %%
import gensim
from gensim.models import KeyedVectors
from operator import itemgetter
from tqdm import tqdm

# %%
deps = KeyedVectors.load_word2vec_format('deps.words')
bow2 = KeyedVectors.load_word2vec_format('bow2.words')
bow5 = KeyedVectors.load_word2vec_format('bow5.words')

# %% Load SimLex/men
with open('SimLex-999/SimLex-999.txt', 'r') as f:
    data = f.read().strip().split('\n')
    data_sim = [itemgetter(*[0, 1, 3])(line.split()) for line in data][1:]

with open('SimLex-999/SimLex-converted.txt', 'w+') as f:
    for line in data_sim:
        f.write(' '.join(line) + '\n')

# %%
deps_similarities = {(w1, w2): deps.similarity(w1, w2) for }

# %%
datasets = ['MEN/MEN_dataset_natural_form_full', 'SimLex-999/SimLex-converted.txt']
models = [deps, bow2, bow5]

for model in models:
    for dataset in datasets:
        out = model.evaluate_word_pairs(dataset, delimiter=' ')
        print(out)

# %% LOAD GOOGLES
with open('questions-words.txt') as f:
    data = f.read().strip().split('\n')
    data = [line.split() for line in data]
    data = [tuple([word.lower() for word in line]) for line in data if len(line) == 4]

# %%
model = bow2

top1 = 0
for w1, w2, w3, w4 in data:
    ranking = model.most_similar(positive=[w2, w3], negative=[w1], topn=10)
    ranking = [word for word, _ in ranking]

    if ranking[0] == w4:
        top1 += 1

accuracy = top1/500
accuracy
