import torch.nn as nn
import torch

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
        return self.emb(word)

class PriorSigma(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(PriorSigma, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.softplus = nn.Softplus()

    def forward(self, word):
        return self.softplus(self.emb(word))

class ELBO(nn.Module):
    def __init__(self, embed_size):
        super(ELBO, self).__init__()
        self.embed_size = embed_size

    def forward(self, context, mu_lambda, sigma_lambda, mu_x, sigma_x, decoded):
        losses = 0

        for batch_id, context in enumerate(context):
            sum = 0
            for word_id in context:
                sum += torch.log(decoded[batch_id][word_id])

            kl = 0
            for dim in range(self.embed_size):
                kl += torch.log(sigma_x[batch_id][0][dim]/sigma_lambda[batch_id][dim])
                kl += (sigma_lambda[batch_id][dim].pow(2) + (mu_lambda[batch_id][dim]-mu_x[batch_id][0][dim]).pow(2))/(2*sigma_x[batch_id][0][dim].pow(2)) - 0.5

            losses += -sum + kl
        return losses
