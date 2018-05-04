import string
from subprocess import check_output
from torch.nn.functional import cosine_similarity
from operator import itemgetter

class Evaluator:
    def __init__(self, w2i, i2w, window_size):
        self.w2i = w2i
        self.i2w = i2w
        self.window_size = window_size
        self.sentences = self.load_data()

    def load_data(self):
        with open('stopwords', 'r') as f:
            stop_words = f.read().split()

        sentences = []
        with open('data/lst/lst_test.preprocessed', 'r') as f:
            data = f.read().strip().split('\n')
            data = [line.split('\t') for line in data]

        candidates = {}
        with open('data/lst/lst.gold.candidates', 'r') as f:
            candidates = f.read().strip().split('\n')
            candidates = [tuple(line.split('::')) for line in candidates]
            candidates = {word: [w for w in c.split(';') if not (' ' in w or w not in self.w2i)] for word, c in candidates}

        for sentence in data:
            pos = int(sentence[2])
            words = [w if w in self.w2i else '<unk>' for w in sentence[3].split()]
            word = words[pos]
            pre = [w for w in words[:pos] if not (w in stop_words or w in string.punctuation or w.isdigit())]
            post = [w for w in words[pos+1:] if not (w in stop_words or w in string.punctuation or w.isdigit())]
            context = pre[-min(len(pre), self.window_size):] + post[:min(len(post), self.window_size)]

            sentences.append((sentence[0], int(sentence[1]), word, context, candidates[sentence[0]]))

        return sentences

    def lst(self, embeddings):
        rankings = []
        for word_id, sentence_id, word, context, candidates in self.sentences:
            # Compute scores
            ranking = []
            for candidate in candidates:
                score = cosine_similarity(embeddings[self.w2i[word]], embeddings[self.w2i[candidate]], dim=0)
                for context_word in context:
                    score += cosine_similarity(embeddings[self.w2i[word]], embeddings[self.w2i[context_word]], dim=0)
                score /= (len(context) + 1)
                ranking.append((candidate, score))
            ranking.sort(key=itemgetter(1))
            rankings.append((word_id, sentence_id, ranking))
        self.write_rankings(rankings)

        # Get GAP
        output = check_output(['python', 'data/lst/lst_gap.py',
                               'data/lst/lst_test.gold', 'data/lst/lst.out',
                               'data/lst/lst.result', 'no-mwe']).decode('utf-8')

        return float(output.split()[1])

    def write_rankings(self, rankings):
        out = ''
        for word_id, sentence_id, ranking in rankings:
            out += '#RESULT\t{} {}\t'.format(word_id, sentence_id)
            for word, score in ranking:
                out += '{} {}\t'.format(word, score)
            out = out[:-1] + '\n'

        with open('data/lst/lst.out', 'w+') as f:
            f.write(out)
