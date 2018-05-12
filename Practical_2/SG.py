# %% Imports
import models, torch, os, time
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from evaluate import Evaluator
from utils import create_vocab, create_SG_dataset, get_lst_vocab

# %% Parameters
model_name = 'SG'
dataset = 'data/hansards/training.en'
embed_size = 100
learning_rate = 0.001
num_epochs = 100
batch_size = 64
window_size = 5
top_size = 10000 # Top n words to use for training, all other words are mapped to <unk>, use None if you do not want to map any word to <unk>

# %% Construct vocabulary and load dataset
lst_words = get_lst_vocab()
vocab, vocab_size, w2i, i2w = create_vocab(top_size, dataset, lst_words)
targets, contexts = create_SG_dataset(dataset, window_size, w2i)

# %% Load Evaluator
evaluator = Evaluator(w2i, i2w, window_size)

# %% Train
train_data = torch.utils.data.TensorDataset(targets, contexts)
train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)

net = models.SkipGram(vocab_size, embed_size).cuda()
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), learning_rate)
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
    net.load_state_dict(state['state_dict'])
    opt.load_state_dict(state['optimizer'])

for epoch in range(saved_epoch, num_epochs):
    start_time = time.time()
    num_batches = len(train_loader)
    total_loss = 0
    for batch, (inputs, targets) in enumerate(train_loader):
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
