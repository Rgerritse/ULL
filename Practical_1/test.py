# %%
import gensim
from gensim.models import KeyedVectors
from operator import itemgetter
from tqdm import tqdm

# %%
embeddings = ['deps', 'bow2', 'bow5']
models = {x: KeyedVectors.load_word2vec_format('{}.words'.format(x)) for x in embeddings}

# %% Load SimLex/men
with open('SimLex-999/SimLex-999.txt', 'r') as f:
    data = f.read().strip().split('\n')
    data_sim = [itemgetter(*[0, 1, 3])(line.split()) for line in data][1:]

with open('SimLex-999/SimLex-converted.txt', 'w+') as f:
    for line in data_sim:
        f.write(' '.join(line) + '\n')

# %% Part 3
datasets = ['MEN/MEN_dataset_natural_form_full', 'SimLex-999/SimLex-converted.txt']

for model in models:
    for dataset in datasets:
        out = models[model].evaluate_word_pairs(dataset, delimiter=' ')
        print(out)

# %% LOAD GOOGLES
with open('questions-words.txt') as f:
    data = [category.split('\n')[:-1] for category in f.read().strip().split(': ')[1:]]
    data = {category[0]: [tuple(word.lower().split()) for word in category[1:]] for category in data}

# %%
for model_name, model in models.items():
    print('Evaluating model {}:'.format(model_name))
    total_acc = []
    total_mrr = []
    total_len = 0
    for category in data:
        acc = []
        mrr = []
        total = len(data[category])
        for i, (w1, w2, w3, w4) in enumerate(data[category]):
            if w4 in model:
                try:
                    ranking = model.most_similar(positive=[w2, w3], negative=[w1], topn=1000)
                    ranking = [word for word, _ in ranking]

                    # Collect stats
                    acc.append(ranking[0] == w4)
                    try:
                        mrr.append(1/(ranking.index(w4)+1))
                    except ValueError:
                        mrr.append(0)
                except KeyError:
                    continue

            # Print progress bar
            if i % 10 == 0 or i+1 == total:
                print('\r{:05d}/{:05d}'.format(i+1, total), end='')

        total_acc += acc
        total_mrr += mrr
        total_len += total

        print(' Accuracy: {:.4f} MRR: {:.4f} ({})'.format(sum(acc)/len(acc), sum(mrr)/len(mrr), category))
    print('{0:05d}/{0:05d} Accuracy: {1:.4f} MRR: {2:.4f} (overall)\n'.format(total_len, sum(total_acc)/len(total_acc), sum(total_mrr)/len(total_mrr)))
