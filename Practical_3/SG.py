# %% Imports
import models, torch, os, time
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import utils, importlib
importlib.reload(utils)
from utils import create_vocab, create_SG_dataset, save_as_glove

# %% Parameters
model_name = 'SGeuroparl'
dataset = 'data/europarl/training.en'
embedding_dest = 'SentEval/pretrained/SG.txt'
embed_size = 128
learning_rate = 0.001
num_epochs = 100
batch_size = 100
window_size = 5
top_size = 25000 # Top n words to use for training, all other words are mapped to <unk>, use None if you do not want to map any word to <unk>

# %% Construct vocabulary and load dataset
vocab, vocab_size, w2i, i2w = create_vocab(top_size, dataset)
targets, contexts = create_SG_dataset(dataset, window_size, w2i)

# %% Save training dataset
data = {
    'vocab': vocab,
    'w2i': w2i,
    'i2w': i2w,
    'targets': targets,
    'contexts': contexts
}
torch.save(data, 'checkpoints/data-{}-nof'.format(top_size))

# %% Load training dataset
data = torch.load('checkpoints/data-{}-nof'.format(top_size))
vocab = data['vocab']
vocab_size = len(vocab)
w2i = data['w2i']
i2w = data['i2w']
targets = data['targets']
contexts = data['contexts']

# %% Train
train_data = torch.utils.data.TensorDataset(targets, contexts)
train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)

net = models.SkipGram(vocab_size, embed_size).cuda()
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), learning_rate)
saved_epoch = 0
train_errors = []

# Check for saved checkpoint
checkpoints = [cp for cp in sorted(os.listdir('checkpoints')) if '-'.join(cp.split('-')[:-1]) == model_name]
if checkpoints:
    state = torch.load('checkpoints/{}'.format(checkpoints[-1]))
    saved_epoch = state['epoch'] + 1
    train_errors = state['train_err']
    net.load_state_dict(state['state_dict'])
    opt.load_state_dict(state['optimizer'])

for epoch in range(saved_epoch, num_epochs):
    start_time = time.time()
    num_batches = len(train_loader)
    total_loss = 0
    for batch, (target, context) in enumerate(train_loader):
        target = target.cuda()
        context = context.cuda()
        opt.zero_grad()
        output = net(target)

        loss = loss_fn(output, context)
        total_loss += loss.data.item()
        loss.backward()
        opt.step()

        pace = (batch+1)/(time.time() - start_time)
        print('\r[Epoch {:03d}/{:03d}] Batch {:06d}/{:06d} [{:.1f}/s] '.format(epoch+1, num_epochs, batch+1, num_batches, pace), end='')

    # Calculate LST score
    total_loss /= len(train_loader)
    print('Time: {:.1f}s Loss: {:.3f}'.format(time.time() - start_time, total_loss))

    train_errors.append(total_loss)

    # Save checkpoint
    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': opt.state_dict(),
        'train_err': train_errors
    }
    torch.save(state, 'checkpoints/{}-{}'.format(model_name, epoch))

# %% Save embeddings
save_as_glove(embedding_dest, net.embeddings.weight, i2w)
