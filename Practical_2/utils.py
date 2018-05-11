import string, time, torch

from collections import Counter

def create_vocab(vocab_size, vocab_file):
    with open('stopwords') as f: # List of stopwords obtained from nltk
        stop_words = f.read().split()

    with open(vocab_file, 'r') as f:
        tokens = f.read().lower().split()
        tokens = [word for word in tokens if not (word in stop_words or word in string.punctuation or word.isdigit())]
        counts = Counter(tokens)

    if not vocab_size:
        vocab_size = len(counts) + 1

    vocab = set(map(lambda x: x[0], counts.most_common(vocab_size-1)))
    vocab.add('<unk>')

    w2i = {k: v for v, k in enumerate(vocab)}
    i2w = {v: k for v, k in enumerate(vocab)}

    return vocab, w2i, i2w

def create_BSG_dataset(data_file, window_size, vocab, w2i):
    with open('stopwords') as f: # List of stopwords obtained from nltk
        stop_words = f.read().split()

    with open(data_file, 'r') as f:
        sentences = [line.lower().split() for line in f.readlines()]

    targets = []
    contexts = []

    for sentence in sentences:
        for i in range(len(sentence)):
            context = []
            for j in range(i-window_size, i+window_size+1):
                if i != j:
                    if j >= 0 and j < len(sentence) and sentence[j] in vocab:
                        context.append(w2i[sentence[j]])
                    else:
                        context.append(w2i['<unk>'])
            if context and not (sentence[i] in stop_words or sentence[i] in string.punctuation or sentence[i].isdigit()):
                if sentence[i] in vocab:
                    targets.append([w2i[sentence[i]]])
                else:
                    targets.append([w2i['<unk>']])
                contexts.append(context)

    return torch.LongTensor(targets).cuda(), torch.LongTensor(contexts).cuda()
