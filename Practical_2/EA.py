# %% Imports
import importlib, models
import torch, os, time,sys
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
learning_rate = 0.01
num_epochs = 100
batch_size = 64
window = 5
model_name = 'EA'
top_size = 5000 # Top n words to use for training, all other words are mapped to <unk>, use None if you do not want to map any word to <unk>
unk = "<unk>"

#%% Create Vocabularies and Data EmbedAlign
required_words = []
vocab_en, vocab_size_en, w2i_en, i2w_en = create_vocab(top_size, "data/hansards/training.en", required_words, 'stopwords_english')
vocab_fr, vocab_size_fr, w2i_fr, i2w_fr = create_vocab(top_size, "data/hansards/training.fr", required_words, 'stopwords_french')

data_en = create_EA_dataset("data/hansards/training.en", vocab_en, w2i_en)
data_fr = create_EA_dataset("data/hansards/training.fr", vocab_fr, w2i_fr)

# %% Initiaze models Embed Align
importlib.reload(models)

# train_data = torch.utils.data.TensorDataset(torch.LongTensor(targets).cuda(), torch.LongTensor(contexts).cuda())
# train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

encoder = models.EmbedAlignEncoder(vocab_size_en, embed_size)
decoder_en = models.EmbedAlignDecoder(vocab_size_en, embed_size)
decoder_fr = models.EmbedAlignDecoder(vocab_size_fr, embed_size)
ELBO_loss = models.EmbedAlignELBO(embed_size)

modules = nn.ModuleList()
modules.append(encoder)
modules.append(decoder_en)
modules.append(decoder_fr)
modules = modules.cuda()

# Check for saved checkpoint
opt = torch.optim.Adam(modules.parameters(), learning_rate)
normal = torch.distributions.normal.Normal(0, 1)
saved_epoch = 0
lst_scores = []
train_errors = []

# Check for saved checkpoint
checkpoints = [cp for cp in sorted(os.listdir('checkpoints')) if '-'.join(cp.split('-')[:-1]) == model_name]
if checkpoints:
    state = torch.load('checkpoints/{}'.format(checkpoints[-1]))
    saved_epoch = state['epoch'] + 1
    lst_scores = state['lst_err']
    train_errors = state['train_err']
    modules.load_state_dict(state['state_dict'])
    opt.load_state_dict(state['optimizer'])


#%% Train models Embed Align
# for epoch in range(saved_epoch, num_epochs):
for epoch in range(saved_epoch, num_epochs):
    start_time = time.time()
    # num_batches = len(train_loader)
    total_loss = 0
    for batch in range(len(data_en[:2000])):
        opt.zero_grad()

        sentence_en = torch.stack([data_en[batch]]).cuda()
        sentence_fr = torch.stack([data_fr[batch]]).cuda()

        mus, sigmas = encoder(sentence_en)
        z = mus + normal.sample((mus.size(0), mus.size(1), embed_size)).cuda() * sigmas
        decoded_en = decoder_en(z)
        decoded_fr = decoder_fr(z)

        # Calculate Loss and train
        loss = ELBO_loss(sentence_en, sentence_fr, mus, sigmas, decoded_en, decoded_fr)
        total_loss += loss.data.item()
        loss.backward()
        opt.step()

        pace = (batch+1)/(time.time() - start_time)
        print('\r[Epoch {:03d}/{:03d}] Batch {:06d}/{:06d} [{:.1f}/s] '.format(epoch+1, num_epochs, batch+1, len(data_en), pace), end='')
    # Calculate LST score
    total_loss /= 2000
    score = 0
    print('Time: {:.1f}s Loss: {:.3f} LST: {:.6f}'.format(time.time() - start_time, total_loss, score))

    lst_scores.append(score)
    train_errors.append(total_loss)

    # Save checkpoint
    state = {
        'epoch': epoch,
        'state_dict': modules.state_dict(),
        'optimizer': opt.state_dict(),
        'lst_err': lst_scores,
        'train_err': train_errors
    }
    torch.save(state, 'checkpoints/{}-{}'.format(model_name, epoch+1))

# %%
encoder.lstm
s = torch.stack([data_en[2]])
mus, sigmas = encoder(s)

# %%
lstm = nn.LSTM(5, 10, batch_first=True, bidirectional=True).cuda()
data = [[[-1.5, 3.5, 0.56, 0.43, 0.67], [-1.5, 3.5, 0.56, 0.43, 0.67]]]
data = torch.Tensor(data).cuda()
data[:][:]
type(data)
print(lstm(data))

#%%
a = torch.LongTensor([1,2,3,4])
print(a.size())
a.repeat(4,1)
