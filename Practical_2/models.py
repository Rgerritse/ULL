import torch.nn as nn
import torch, sys

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size, vocab_size)

    def forward(self, data):
        out = self.embeddings(data)
        out = self.fc1(out)
        return out

class BayesianEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, window):
        super(BayesianEncoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.fc1 = nn.Linear(2 * embed_size, embed_size)
        self.fc2 = nn.Linear(2 * embed_size, embed_size)
        self.m = nn.Linear(2 * embed_size, 2 * embed_size)
        self.window = window

    def forward(self, target, contexts):
        target_emb = self.embeddings(target)
        target_emb = target_emb.repeat(1, 2 * self.window, 1)
        contexts_emb = self.embeddings(contexts)
        cat_emb = torch.cat((target_emb, contexts_emb), 2)
        proj_emb = self.m(cat_emb)
        relu_emb = self.relu(proj_emb)
        sum_emb = torch.sum(relu_emb, 1)
        mu_emb  = self.fc1(sum_emb)
        sigma_emb = self.softplus(self.fc2(sum_emb))

        return mu_emb, sigma_emb

class BayesianDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(BayesianDecoder, self).__init__()
        self.affine = nn.Linear(embed_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        return self.softmax(self.affine(z))

class PriorMu(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(PriorMu, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_size)

    def forward(self, word):
        return self.emb(word).squeeze()

class PriorSigma(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(PriorSigma, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.softplus = nn.Softplus()

    def forward(self, word):
        return self.softplus(self.emb(word)).squeeze()

class EmbedAlignEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbedAlignEncoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, embed_size, bidirectional = True, batch_first=True)
        self.mu_fc1 = nn.Linear(embed_size, embed_size)
        self.mu_fc2 = nn.Linear(embed_size, embed_size)
        self.sigma_fc1 = nn.Linear(embed_size, embed_size)
        self.sigma_fc2 = nn.Linear(embed_size, embed_size)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.embed_size = embed_size

    def forward(self, sentence):
        emb = self.embeddings(sentence)
        lstm_out, (hn, cn) = self.lstm(emb)
        lstm_1, lstm_2 = torch.split(lstm_out, self.embed_size, dim=2)
        lstm_out = lstm_1 + lstm_2
        mus = self.mu_fc1(lstm_out)
        mus = self.relu(mus)
        mus = self.mu_fc1(mus)
        sigmas = self.sigma_fc1(lstm_out)
        sigmas = self.relu(sigmas)
        sigmas = self.sigma_fc2(sigmas)
        sigmas = self.softplus(sigmas)

        return mus, sigmas

class EmbedAlignDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbedAlignDecoder, self).__init__()
        self.fc1 = nn.Linear(embed_size, embed_size)
        self.fc2 = nn.Linear(embed_size, vocab_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return self.softmax(out)

class EmbedAlignELBO(nn.Module):
    def __init__(self, embed_size):
        super(EmbedAlignELBO, self).__init__()
        self.embed_size = embed_size

    def forward(self, sentence_en, sentence_fr, mus, sigmas, decoded_en, decoded_fr):
        sentence_en = sentence_en.unsqueeze(2)
        sum = decoded_en.gather(2, sentence_en).log().sum()

        for batch in range(decoded_fr.size(0)):
            for fr_pos, word_fr in enumerate(sentence_fr):
                sum_en = 0
                for en_pos in range(len(sentence_en)):
                    sum_en += (1/len(sentence_en)) * decoded_fr[batch][en_pos][fr_pos]
                sum += sum_en.log()

        kl = ((1 / sigmas).log() + (sigmas.pow(2) + mus.pow(2))/2 - 0.5).sum()

        return kl - sum

class ELBO(nn.Module):
    def __init__(self, embed_size):
        super(ELBO, self).__init__()
        self.embed_size = embed_size

    def forward(self, context, mu_lambda, sigma_lambda, mu_x, sigma_x, decoded):
        sum = decoded.gather(1, context).log().sum()
        kl = ((sigma_x / sigma_lambda).log() + (sigma_lambda.pow(2) + (mu_lambda-mu_x).pow(2))/(2*sigma_x.pow(2)) - 0.5).sum()
        return kl - sum
