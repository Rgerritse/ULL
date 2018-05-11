# %% Imports
import torch, os, time
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from evaluate import Evaluator
from collections import defaultdict
import string
import numpy as np

# %% Parameters
embed_size = 100
learning_rate = 0.001
num_epochs = 1
batch_size = 64
window = 2
model_name = 'skipgram'
top_size = 10000 # Top n words to use for training, all other words are mapped to <unk>, use None if you do not want to map any word to <unk>
unk = "<unk>"

# %% Construct vocabulary
with open("stopwords") as fsw: # List of stopwords obtained from nltk
    stopWords = fsw.read().split()

counts = defaultdict(lambda: 0)
with open("data/hansards/training.en") as f:
    tokens = f.read().lower().split()
    for word in tokens:
        if not (word in stopWords or word in string.punctuation or word.isdigit()): # Filter stopwords, punctuation and digits
            counts[word] += 1;

sortedCounts = sorted(counts.items(), key = lambda kv: kv[1],reverse=True)[:top_size]
if top_size != None and top_size <= len(sortedCounts):
    vocab = set(map(lambda x: x[0], sortedCounts[:top_size]))
else:
    vocab = set(map(lambda x: x[0], sortedCounts))

if top_size != None:
    vocab.add(unk)
vocab_size = len(vocab)

w2i = {k: v for v, k in enumerate(vocab)}
i2w = {v: k for v, k in enumerate(vocab)}

# %% Dataset creation
with open("data/hansards/training.en") as f:
    sentences = [line.lower().split() for line in f.readlines()]

data = []
labels = []
for sentence in sentences:
    for i in range(len(sentence)):
        for j in range(i-window, i+window+1):
            if i != j and j>=0 and j<len(sentence):
                if (not (sentence[i] in stopWords or sentence[i] in string.punctuation or sentence[i].isdigit() or
                    sentence[j] in stopWords or  sentence[j] in string.punctuation or sentence[j].isdigit())):
                    # data.append(sentence[i])
                    # labels.append(sentence[j])
                    if sentence[i] in vocab:
                        data.append(w2i[sentence[i]])
                    else:
                        data.append(w2i[unk])
                    if sentence[j] in vocab:
                        labels.append(w2i[sentence[j]])
                    else:
                        labels.append(w2i[unk])

# %% Network Definition
class Net(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Net, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size, vocab_size)

    def forward(self, data):
        out = self.embeddings(data)
        out = self.fc1(out)
        return out

# %% Load Evaluator
evaluator = Evaluator(w2i, i2w, window)

# %% Train
train_data = torch.utils.data.TensorDataset(torch.LongTensor(data), torch.LongTensor(labels))
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

net = Net(vocab_size, embed_size).cuda()
loss_fn = nn.CrossEntropyLoss()
# opt = torch.optim.SGD(net.parameters(), learning_rate)
opt = torch.optim.Adam(net.parameters(), learning_rate)

# Check for saved checkpoint
saved_epoch = 0
lst_scores = []
train_errors = []

checkpoints = [cp for cp in sorted(os.listdir('checkpoints')) if model_name in cp]
if checkpoints:
    state = torch.load('checkpoints/{}'.format(checkpoints[-1]))
    saved_epoch = state['epoch'] + 1
    lst_scores = state['lst_err']
    train_errors = state['train_err']
    net.load_state_dict(state['state_dict'])
    opt.load_state_dict(state['optimizer'])

# %%
for epoch in range(saved_epoch, num_epochs):
    start_time = time.time()
    num_batches = len(train_loader)
    total_loss = 0
    for batch, (inputs, targets) in enumerate(train_loader):
        inputs = Variable(inputs).cuda()
        targets = Variable(targets).cuda()

        opt.zero_grad()
        outputs = net(inputs)

        loss = loss_fn(outputs, targets)
        total_loss += loss.data.item()
        loss.backward()
        opt.step()

        pace = (batch+1)/(time.time() - start_time)
        print('\r[Epoch {:03d}/{:03d}] Batch {:06d}/{:06d} [{:.1f}/s] '.format(epoch+1, num_epochs, batch+1, num_batches, pace), end='')

    # Calculate LST score
    total_loss /= len(train_loader)
    score = evaluator.lst(net.embeddings.weight.data)
    print('Time: {:.1f}s Loss: {:.3f} LST: {:.6f}'.format(time.time() - start_time, total_loss, score))

    lst_scores.append(score)
    train_errors.append(total_loss)

    # Save checkpoint
    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': opt.state_dict(),
        'lst_err': lst_scores,
        'train_err': train_errors
    }
    torch.save(state, 'checkpoints/{}-{}'.format(model_name, epoch))

# %% Get embeddings
import importlib, evaluate
importlib.reload(evaluate)
evaluator = evaluate.Evaluator(w2i, i2w, window)
x = evaluator.lst(net.embeddings.weight.data)
x

#%% Data Creation Bayesian Skip Gram
with open("data/hansards/training.en") as f:
    sentences = [line.lower().split() for line in f.readlines()]

targets = []
contexts = []

for sentence in sentences:
    for i in range(len(sentence)):
        context = []
        for j in range(i-window, i+window+1):
            if i != j:
                if j>=0 and j<len(sentence):
                    if sentence[j] in vocab:
                        context.append(w2i[sentence[j]])
                    else:
                        context.append(w2i[unk])
                else:
                    context.append(w2i[unk])
        if context != [] and not (sentence[i] in stopWords or sentence[i] in string.punctuation or sentence[i].isdigit()):
            if sentence[i] in vocab:
                targets.append([w2i[sentence[i]]])
            else:
                targets.append([w2i[unk]])
            contexts.append(context)

# %% Bayesian SkipGram Network Definition
class BayesianEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(BayesianEncoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.fc1 = nn.Linear(2 * embed_size, embed_size)
        self.fc2 = nn.Linear(2 * embed_size, embed_size)
        self.m = nn.Linear(2 * embed_size, 2 * embed_size)

    def forward(self, target, contexts):
        target_emb = self.embeddings(target)
        target_emb = target_emb.repeat(1, 2 * window, 1)
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
        self.softmax = nn.Softmax()

    def forward(self, z):
        print(z.size())
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
    def __init__(self):
        super(ELBO, self).__init__()

    def forward(self, context, mu_lambda, sigma_lambda, mu_x, sigma_x, decoded):
        sum = 0
        for word_id in context:
            sum += torch.log(decoded[word_id])

        kl = 0
        for dim in range(embed_size):
            kl += torch.log(sigma_x[dim]/sigma_lambda[dim]) + (sigma_lambda[dim].pow(2) + (mu_lambda[dim]-mu_x[dim]).pow(2))/(2*sigma_x[dim].pow(2)) - 0.5

        return -sum + kl

# %% Train
model_name = 'BSK'

train_data = torch.utils.data.TensorDataset(torch.LongTensor(targets), torch.LongTensor(contexts))
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

encoder = BayesianEncoder(vocab_size, embed_size).cuda()
decoder = BayesianDecoder(vocab_size, embed_size).cuda()
priorMu = PriorMu(vocab_size, embed_size).cuda()
priorSigma = PriorSigma(vocab_size, embed_size).cuda()
ELBO_loss = ELBO().cuda()

models = nn.ModuleList()
models.append(encoder)
models.append(decoder)
models.append(priorMu)
models.append(priorSigma)

opt = torch.optim.Adam(models.parameters(), learning_rate)

# Check for saved checkpoint
saved_epoch = 0
lst_scores = []
train_errors = []

# checkpoints = [cp for cp in sorted(os.listdir('checkpoints')) if model_name in cp]
# if checkpoints:
#     state = torch.load('checkpoints/{}'.format(checkpoints[-1]))
#     saved_epoch = state['epoch'] + 1
#     lst_scores = state['lst_err']
#     train_errors = state['train_err']
#     net.load_state_dict(state['state_dict'])
#     opt.load_state_dict(state['optimizer'])

# %% WERKT NOG NIET
for epoch in range(saved_epoch, num_epochs):
    start_time = time.time()
    num_batches = len(train_loader)
    total_loss = 0
    for batch, (b_targets, b_contexts) in enumerate(train_loader):
        if batch == 0:
            b_targets = Variable(b_targets).cuda()
            b_contexts = Variable(b_contexts).cuda()

            opt.zero_grad()

            #
            (mu_lambda, sigma_lambda) = encoder(b_targets, b_contexts)
            epsilon = torch.FloatTensor((np.random.normal(0, 1, (batch_size, embed_size)))).cuda()
            z = (mu_lambda) + epsilon * (sigma_lambda)
            decoded = decoder(z)
            mu_x = priorMu(b_targets)
            sigma_x = priorSigma(b_targets)


            loss = ELBO_loss(b_contexts, mu_lambda, sigma_lambda, mu_x, sigma_x, decoded)
            total_loss += loss.data.item()
            loss.backward()
            opt.step()

        pace = (batch+1)/(time.time() - start_time)
        print('\r[Epoch {:03d}/{:03d}] Batch {:06d}/{:06d} [{:.1f}/s] '.format(epoch+1, num_epochs, batch+1, num_batches, pace), end='')

# epsilon = torch.FloatTensor((np.random.normal(0, 1, (batch_size, 2 * embed_size)))).cuda()
# z = mu_emb + epsilon * sigma_emb
# out = self.fc3(z)

    # Calculate LST score
    # total_loss /= len(train_loader)
    # score = evaluator.lst(net.embeddings.weight.data)
    # print('Time: {:.1f}s Loss: {:.3f} LST: {:.6f}'.format(time.time() - start_time, total_loss, score))
    #
    # lst_scores.append(score)
    # train_errors.append(total_loss)
    #
    # # Save checkpoint
    # state = {
    #     'epoch': epoch,
    #     'state_dict': net.state_dict(),
    #     'optimizer': opt.state_dict(),
    #     'lst_err': lst_scores,
    #     'train_err': train_errors
    # }
    # torch.save(state, 'checkpoints/{}-{}'.format(model_name, epoch))

#%%
