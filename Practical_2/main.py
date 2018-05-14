# %% Imports
import importlib, models
import torch, os, time
import torch.utils.data
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.autograd import Variable
from evaluate import Evaluator
from collections import defaultdict
import string
from utils import create_vocab, create_EA_dataset

# %% Parameters
embed_size = 100
learning_rate = 0.001
num_epochs = 100
batch_size = 64
window = 5
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

# %% Trainmodel_name = 'BSK'
importlib.reload(models)

train_data = torch.utils.data.TensorDataset(torch.LongTensor(targets).cuda(), torch.LongTensor(contexts).cuda())
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

encoder = models.BayesianEncoder(vocab_size, embed_size, window).cuda()
decoder = models.BayesianDecoder(vocab_size, embed_size).cuda()
priorMu = models.PriorMu(vocab_size, embed_size).cuda()
priorSigma = models.PriorSigma(vocab_size, embed_size).cuda()
ELBO_loss = models.ELBO(embed_size).cuda()

modules = nn.ModuleList()
modules.append(encoder)
modules.append(decoder)
modules.append(priorMu)
modules.append(priorSigma)

opt = torch.optim.Adam(modules.parameters(), learning_rate)

# Check for saved checkpoint
normal = torch.distributions.normal.Normal(0, 1)
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

# %% WERKT MISSCHIEN
for epoch in range(saved_epoch, num_epochs):
    start_time = time.time()
    num_batches = len(train_loader)
    total_loss = 0
    for batch, (b_targets, b_contexts) in enumerate(train_loader):
        opt.zero_grad()

        (mu_lambda, sigma_lambda) = encoder(b_targets, b_contexts)
        z = (mu_lambda) + normal.sample((b_targets.size(1), embed_size)).cuda() * (sigma_lambda)
        decoded = decoder(z)
        mu_x = priorMu(b_targets)
        sigma_x = priorSigma(b_targets)

        loss = ELBO_loss(b_contexts, mu_lambda, sigma_lambda, mu_x, sigma_x, decoded)
        total_loss += loss.data.item()
        loss.backward()
        opt.step()

        pace = (batch+1)/(time.time() - start_time)
        print('\r[Epoch {:03d}/{:03d}] Batch {:06d}/{:06d} [{:.1f}/s] '.format(epoch+1, num_epochs, batch+1, num_batches, pace), end='')


    # Calculate LST score10000
    total_loss /= len(train_loader)
    # score = evaluator.lst(net.embeddings.weight.data)
    score = 0
    print('Time: {:.1f}s Loss: {:.3f} LST: {:.6f}'.format(time.time() - start_time, total_loss, score))
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

#%% Create Vocabularies and Data EmbedAlign
required_words = []
vocab_en, vocab_size_en, w2i_en, i2w_en = create_vocab(top_size, "data/hansards/training.en", required_words, 'stopwords_english')
vocab_fr, vocab_size_fr, w2i_fr, i2w_fr = create_vocab(top_size, "data/hansards/training.fr", required_words, 'stopwords_french')

data_en = create_EA_dataset("data/hansards/training.en", vocab_en, w2i_en)
data_fr = create_EA_dataset("data/hansards/training.fr", vocab_fr, w2i_fr)

#%%

sorted_data_en = sorted(data_en, key=len, reverse=True)
longest = len(sorted_data_en[0])
batches = [data_en[i:i + batch_size] for i in range(0, len(data_en), batch_size)]
# pack_data_en = pack_sequence(sorted_data_en, batch_first=True)

# %% Initiaze models Embed Align
importlib.reload(models)

# train_data = torch.utils.data.TensorDataset(torch.LongTensor(targets).cuda(), torch.LongTensor(contexts).cuda())
# train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

encoder = models.EmbedAlignEncoder(vocab_size_en, embed_size)
# decoder = models.BayesianDecoder(vocab_size, embed_size).cuda()
# priorMu = models.PriorMu(vocab_size, embed_size).cuda()
# priorSigma = models.PriorSigma(vocab_size, embed_size).cuda()
# ELBO_loss = models.ELBO(embed_size).cuda()

modules = nn.ModuleList()
modules.append(encoder)
# modules.append(decoder)
# modules.append(priorMu)
# modules.append(priorSigma)

opt = torch.optim.Adam(modules.parameters(), learning_rate)

# Check for saved checkpoint
normal = torch.distributions.normal.Normal(0, 1)
saved_epoch = 0
lst_scores = []
train_errors = []

#%%
print(data_en[i])

#%% Train models Embed Align
# for epoch in range(saved_epoch, num_epochs):
for epoch in range(saved_epoch, 1):
    start_time = time.time()
    # num_batches = len(train_loader)
    total_loss = 0
    for i in range(len(data_en)):
        if i == 0:
            encoder(sentence_en)
            # print(sentence_en)
        # if i == 0:
            # sorted_batch = sorted(batch, key=len, reverse=True)
            # pack_batch = pack_sequence(sorted_batch)
            # encoder(batch)
            # print(len(batch))

            # seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    # for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        # seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    # return seq_tensor
            # pace = (i+1)/(time.time() - start_time)
            # print('\r[Epoch {:03d}/{:03d}] Batch {:06d}/{:06d} [{:.1f}/s] '.format(epoch+1, num_epochs, i+1, len(data_en), pace), end='')
    # for i in range(len(data_en)):
        # if i ==0:
            # sentence_en = torch.LongTensor(data_en[i])
            # sentence_fr = torch.LongTensor(data_fr[i])
            # opt.zero_grad()

            # encoder(sentence_en)

            # (mu_lambda, sigma_lambda) = encoder(b_targets, b_contexts)
            # z = (mu_lambda) + normal.sample((b_targets.size(1), embed_size)).cuda() * (sigma_lambda)
            # decoded = decoder(z)
            # mu_x = priorMu(b_targets)
            # sigma_x = priorSigma(b_targets)

            # loss = ELBO_loss(b_contexts, mu_lambda, sigma_lambda, mu_x, sigma_x, decoded)
            # total_loss += loss.data.item()
            # loss.backward()
            # opt.step()




    # Calculate LST score10000
    # total_loss /= len(train_loader)
    # score = evaluator.lst(net.embeddings.weight.data)
    # score = 0
    # print('Time: {:.1f}s Loss: {:.3f} LST: {:.6f}'.format(time.time() - start_time, total_loss, score))

#%%
import itertools


seqs = ['ghatmasala','nicela','c-pakodas']

# make <pad> idx 0
vocab = ['<pad>'] + sorted(list(set(flatten(seqs))))

# make model
embed = nn.Embedding(len(vocab), 10).cuda()
lstm = nn.LSTM(10, 5, batch_first=True).cuda()

vectorized_seqs = [[vocab.index(tok) for tok in seq]for seq in seqs]
vectorized_seqs = sorted(vectorized_seqs, key=len, reverse=True)
vectorized_seqs = [torch.cuda.LongTensor(i) for i in vectorized_seqs]
# seq_lengths = torch.cuda.LongTensor(list(map(len, vectorized_seqs)))


# padded_sequence = pad_sequence(vectorized_seqs, batch_first=True)
/
# seq_tensor = embed(padded_sequence)
# packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)
# pad_packed_sequence(packed_input, batch_first=True)
# get the length of each seq in your batch
# seq_lengths = torch.cuda.LongTensor(list(map(len, vectorized_seqs)))
#
# # dump padding everywhere, and place seqs on the left.
# # NOTE: you only need a tensor as big as your longest sequence
# seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long().cuda()
# for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
# 	seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
#
# # SORT YOUR TENSORS BY LENGTH!
# seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
# seq_tensor = seq_tensor[perm_idx]
#
# # utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
# # Otherwise, give (L,B,D) tensors
# seq_tensor = seq_tensor # (B,L,D) -> (L,B,D)
#
# print(seq_tensor.size())
# # embed your sequences
# seq_tensor = embed(seq_tensor)
# print(seq_tensor.size())
# # pack them up nicely
# print(seq_tensor)
# packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy())
#
# # throw them through your LSTM (remember to give batch_first=True here if you packed with it)
# packed_output, (ht, ct) = lstm(packed_input)
#
# # unpack your output if required
# output, _ = pad_packed_sequence(packed_output)
# print(output)

# Or if you just want the final hidden state
#%%
print(torch.LongTensor([[1,2]]))
