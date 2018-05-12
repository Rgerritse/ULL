# %% Imports
import models, torch, os, time
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from evaluate import BayesianEvaluator
from utils import create_vocab, create_BSG_dataset, get_lst_vocab

# %% Parameters
model_name = 'BSG'
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
targets, contexts = create_BSG_dataset(dataset, window_size, w2i)

# %% Initialise models
train_data = TensorDataset(targets, contexts)
train_loader = DataLoader(train_data, batch_size, shuffle=True)

encoder = models.BayesianEncoder(vocab_size, embed_size, window_size)
decoder = models.BayesianDecoder(vocab_size, embed_size)
priorMu = models.PriorMu(vocab_size, embed_size)
priorSigma = models.PriorSigma(vocab_size, embed_size)
ELBO_loss = models.ELBO(embed_size)

modules = nn.ModuleList()
modules.append(encoder)
modules.append(decoder)
modules.append(priorMu)
modules.append(priorSigma)
modules = modules.cuda()

# %% Load Evaluator
evaluator = BayesianEvaluator(w2i, window_size)

# %% Train
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

    # Calculate LST score
    total_loss /= len(train_loader)
    score = evaluator.lst(encoder, priorMu, priorSigma)
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
import importlib, evaluate
importlib.reload(evaluate)
from evaluate import BayesianEvaluator
evaluator = BayesianEvaluator(w2i, window_size)
