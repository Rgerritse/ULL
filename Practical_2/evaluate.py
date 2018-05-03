import nltk, string
#nltk.download('stopwords')
from nltk.corpus import stopwords
from subprocess import check_output

class Evaluator:
    def __init__(self, w2i, i2w, window_size):
        self.w2i = w2i
        self.i2w = i2w
        self.window_size = window_size
        self.sentences = self.load_data()

    def load_data(self):
        stop_words = set(stopwords.words('english') + list(string.punctuation))

        sentences = []
        with open('data/lst/lst_test.preprocessed', 'r') as f:
            data = f.read().strip().split('\n')
            data = [line.split('\t') for line in data]

        candidates = {}
        with open('data/lst/lst.gold.candidates', 'r') as f:
            candidates = f.read().strip().split('\n')
            candidates = [tuple(line.split('::')) for line in candidates]
            candidates = {word: [w for w in c.split(';') if ' ' not in w] for word, c in candidates}

        for sentence in data:
            pos = int(sentence[2])
            words = sentence[3].split()
            word = words[pos]
            pre = [w for w in words[:pos] if w not in stop_words]
            post = [w for w in words[pos+1:] if w not in stop_words]
            context = pre[-min(len(pre), self.window_size):] + post[:min(len(post), self.window_size)]

            sentences.append((sentence[0], int(sentence[1]), word, context, candidates[sentence[0]]))

        return sentences

    def lst(self, embeddings):
        rankings = []
        for word_id, sentence_id, word, context, candidates in self.sentences:
            rankings.append((word_id, sentence_id, [('operate', -2), ('john', -4)]))

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
