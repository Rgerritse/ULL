import string, time, torch

from collections import Counter

def create_vocab(top_size, vocab_file, required_words):
    with open('stopwords') as f: # List of stopwords obtained from nltk
        stop_words = f.read().split()

    with open(vocab_file, 'r') as f:
        tokens = f.read().lower().split()
        tokens = [word for word in tokens if not (word in stop_words or word in string.punctuation or word.isdigit())]
        counts = Counter(tokens)

    if not top_size:
        top_size = len(counts)

    vocab = set(map(lambda x: x[0], counts.most_common(top_size)))
    vocab.add('<unk>')

    # Merge vocabulary with required words
    vocab = vocab.union(required_words)
    vocab_size = len(vocab)

    w2i = {k: v for v, k in enumerate(vocab)}
    i2w = {v: k for v, k in enumerate(vocab)}

    return vocab, vocab_size, w2i, i2w

def create_BSG_dataset(data_file, window_size, w2i):
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
                    if j >= 0 and j < len(sentence) and sentence[j] in w2i:
                        context.append(w2i[sentence[j]])
                    else:
                        context.append(w2i['<unk>'])
            if context and not (sentence[i] in stop_words or sentence[i] in string.punctuation or sentence[i].isdigit()):
                if sentence[i] in w2i:
                    targets.append([w2i[sentence[i]]])
                else:
                    targets.append([w2i['<unk>']])
                contexts.append(context)

    return torch.LongTensor(targets).cuda(), torch.LongTensor(contexts).cuda()

def create_SG_dataset(data_file, window_size, w2i):
    with open('stopwords') as f: # List of stopwords obtained from nltk
        stop_words = f.read().split()

    with open(data_file, 'r') as f:
        sentences = [line.lower().split() for line in f.readlines()]

    targets = []
    contexts = []
    for sentence in sentences:
        for i in range(len(sentence)):
            for j in range(i-window_size, i+window_size+1):
                if i != j and j >= 0 and j < len(sentence):
                    if (not (sentence[i] in stop_words or sentence[i] in string.punctuation or sentence[i].isdigit() or
                        sentence[j] in stop_words or  sentence[j] in string.punctuation or sentence[j].isdigit())):

                        if sentence[i] in w2i:
                            targets.append(w2i[sentence[i]])
                        else:
                            targets.append(w2i['<unk>'])
                        if sentence[j] in w2i:
                            contexts.append(w2i[sentence[j]])
                        else:
                            contexts.append(w2i['<unk>'])

    return torch.LongTensor(targets).cuda(), torch.LongTensor(contexts).cuda()

def get_lst_vocab():
    vocab = set()
    with open('data/lst/lst.gold.candidates', 'r') as f:
        for line in f.readlines():
            line = line.strip().split('::')
            vocab.add(line[0].split('.')[0])
            for word in line[1].split(';'):
                if ' ' not in word:
                    vocab.add(word)
    return vocab
