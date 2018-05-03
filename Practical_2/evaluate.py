import nltk
try:
    from nltk.corpus import stopwords
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords

class Evaluator:
    def __init__(self, w2i, i2w, window_size):
        self.w2i = w2i
        self.i2w = i2w
        self.window_size = window_size
        self.sentences = self.loadData()

    def loadData(self):
        stop_words = set(stopwords.words('english'))

        sentences = []
        with open('data/lst/lst_test.preprocessed', 'r') as f:
            data = f.read().strip().split('\n')
            data = [line.split('\t') for line in data]

        for sentence in data:
            pos = int(sentence[2])
            words = sentence[3].split()
            word = words[pos]
            pre = [w for w in words[:pos] if w not in stop_words]
            post = [w for w in words[pos+1:] if w not in stop_words]
            context = pre[-min(len(pre), self.window_size):] + post[:min(len(post), self.window_size)]

            return



    def lst(self, embeddings):
        pass

ev = Evaluator({},{}, 2)
