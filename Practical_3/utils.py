import string, time, torch
from tqdm import tqdm

from collections import Counter

def create_vocab(top_size, vocab_file):
    with open('stopwords_english') as f: # List of stopwords obtained from nltk
        stop_words = f.read().split()
        stop_words += list(string.punctuation)
        stop_words += list(range(10))
        stop_words = set(stop_words)

    with open(vocab_file, 'r') as f:
        tokens = f.read().lower().split()
        tokens = [word for word in tokens if not word in stop_words]
        counts = Counter(tokens)

    if not top_size:
        top_size = len(counts)

    vocab = set(map(lambda x: x[0], counts.most_common(top_size)))
    vocab.add('<unk>')
    vocab_size = len(vocab)

    w2i = {k: v for v, k in enumerate(vocab)}
    i2w = {v: k for v, k in enumerate(vocab)}

    return vocab, vocab_size, w2i, i2w

def create_SG_dataset(data_file, window_size, w2i):
    with open('stopwords_english') as f: # List of stopwords obtained from nltk
        stop_words = f.read().split()
        stop_words += list(string.punctuation)
        stop_words += list(range(10))
        stop_words = set(stop_words)

    with open(data_file, 'r') as f:
        sentences = [line.lower().split() for line in f.readlines()]

    targets = []
    contexts = []
    for sentence in tqdm(sentences):
        for i in range(len(sentence)):
            for j in range(i-window_size, i+window_size+1):
                if i != j and j >= 0 and j < len(sentence):
                    if not (sentence[i] in stop_words or sentence[j] in stop_words):
                        if sentence[i] in w2i:
                            targets.append(w2i[sentence[i]])
                        else:
                            targets.append(w2i['<unk>'])
                        if sentence[j] in w2i:
                            contexts.append(w2i[sentence[j]])
                        else:
                            contexts.append(w2i['<unk>'])

    return torch.LongTensor(targets).cuda(), torch.LongTensor(contexts).cuda()

def save_as_glove(file_name, embeddings, i2w):
    with open(file_name, 'w+') as f:
        for i in i2w:
            vector = ' '.join(map(str, embeddings[i].tolist()))
            f.write('{} {}\n'.format(i2w[i], vector))
