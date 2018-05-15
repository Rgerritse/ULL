import string, torch
from subprocess import check_output
from torch.nn.functional import cosine_similarity
from operator import itemgetter

class Evaluator:
    def __init__(self, w2i, i2w, window_size):
        self.w2i = w2i
        self.i2w = i2w
        self.window_size = window_size
        self.sentences = []

        self.load_data()

    def load_data(self):
        with open('stopwords_english', 'r') as f:
            filter_words = f.read().split() + list(string.punctuation)

        with open('data/lst/lst.gold.candidates', 'r') as f:
            candidates = f.read().strip().split('\n')
            candidates = [tuple(line.split('::')) for line in candidates]
            candidates = {word: [self.w2i[w] for w in c.split(';') if not ' ' in w] for word, c in candidates}

        with open('data/lst/lst_test.preprocessed', 'r') as f:
            data = f.read().strip().split('\n')
            data = [line.split('\t') for line in data]

            for sentence in data:
                word_id = sentence[0]
                sentence_id = int(sentence[1])
                pos = int(sentence[2])
                words = [self.w2i[w] if (w in self.w2i and w not in filter_words and not w.isdigit()) else self.w2i['<unk>'] for w in sentence[3].split()]
                word = words[pos]
                context = words[max(0, pos - self.window_size):min(len(words), pos + self.window_size)]

                self.sentences.append((word_id, sentence_id, word, context, candidates[word_id]))

    def write_rankings(self, rankings):
        out = ''
        for word_id, sentence_id, ranking in rankings:
            out += '#RESULT\t{} {}\t'.format(word_id, sentence_id)
            for word, score in ranking:
                out += '{} {}\t'.format(self.i2w[word], score)
            out = out[:-1] + '\n'

        with open('data/lst/lst.out', 'w+') as f:
            f.write(out)

class SGEvaluator(Evaluator):
    def __init__(self, w2i, i2w, window_size):
        super().__init__(w2i, i2w, window_size)

    def lst(self, embeddings):
        rankings = []
        for word_id, sentence_id, word, context, candidates in self.sentences:
            # Compute scores
            ranking = []
            for candidate in candidates:
                emb_word = embeddings[word]
                emb_candidate = embeddings[candidate]

                score = cosine_similarity(emb_word, emb_candidate, dim=0)
                for context_word in context:
                    score += cosine_similarity(emb_word, embeddings[context_word], dim=0)
                score /= (len(context) + 1)
                ranking.append((candidate, score))
            ranking.sort(key=itemgetter(1), reverse=True)
            rankings.append((word_id, sentence_id, ranking))
        self.write_rankings(rankings)

        # Get GAP
        output = check_output(['python', 'data/lst/lst_gap.py',
                               'data/lst/lst_test.gold', 'data/lst/lst.out',
                               'data/lst/lst.result', 'no-mwe']).decode('utf-8')

        return float(output.split()[1])

class BSGEvaluator(Evaluator):
    def __init__(self, w2i, i2w, window_size):
        super().__init__(w2i, i2w, window_size)


    def lst(self, encoder, priorMu, priorSigma):
        rankings = []
        for word_id, sentence_id, word, context, candidates in self.sentences:
            # Pad context with unknowns
            context += [self.w2i['<unk>']]*(2*self.window_size-len(context))

            # Compute scores
            ranking = []
            mu_posterior, sigma_posterior = encoder(torch.LongTensor([word]).cuda(), torch.LongTensor([context]).cuda())
            for candidate in candidates:
                mu_prior = priorMu(torch.LongTensor([candidate]).cuda())
                sigma_prior = priorSigma(torch.LongTensor([candidate]).cuda())

                kl = ((sigma_prior / sigma_posterior).log() + (sigma_posterior.pow(2) + (mu_posterior-mu_prior).pow(2))/(2*sigma_prior.pow(2)) - 0.5).sum()
                ranking.append((candidate, kl))
            ranking.sort(key=itemgetter(1))
            rankings.append((word_id, sentence_id, ranking))
        self.write_rankings(rankings)

        # Get GAP
        output = check_output(['python', 'data/lst/lst_gap.py',
                               'data/lst/lst_test.gold', 'data/lst/lst.out',
                               'data/lst/lst.result', 'no-mwe']).decode('utf-8')

        return float(output.split()[1])
