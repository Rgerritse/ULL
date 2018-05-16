# %% Imports
import torch, os, time,sys, random, string, importlib, models, evaluate
import torch.utils.data
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.autograd import Variable
from utils import create_vocab, create_EA_dataset, get_lst_vocab

# %% Parameters
embed_size = 100
learning_rate = 0.0001
num_epochs = 10
batch_size = 12
window = 5
model_name = 'EA'
top_size = 10000 # Top n words to use for training, all other words are mapped to <unk>, use None if you do not want to map any word to <unk>

#%% Create Vocabularies and Data EmbedAlign
required_words = get_lst_vocab()
vocab_en, vocab_size_en, w2i_en, i2w_en = create_vocab(top_size, "data/hansards/training.en", required_words, 'stopwords_english')
vocab_fr, vocab_size_fr, w2i_fr, i2w_fr = create_vocab(top_size, "data/hansards/training.fr", required_words, 'stopwords_french')

data_en = create_EA_dataset("data/hansards/training.en", vocab_en, w2i_en)
data_fr = create_EA_dataset("data/hansards/training.fr", vocab_fr, w2i_fr)

data_length = list(map(len, data_en))

# Create batches
batches = [(data_en[i:i + batch_size], data_fr[i:i + batch_size], data_length[i:i + batch_size]) for i in range(0, len(data_en), batch_size)][:10000]

# %% Initiaze models Embed Align
importlib.reload(models)
importlib.reload(evaluate)

encoder = models.EmbedAlignEncoder(vocab_size_en, embed_size)
decoder_en = models.EmbedAlignDecoder(vocab_size_en, embed_size)
decoder_fr = models.EmbedAlignDecoder(vocab_size_fr, embed_size)
ELBO_loss = models.EmbedAlignELBO(embed_size)

modules = nn.ModuleList()
modules.append(encoder)
modules.append(decoder_en)
modules.append(decoder_fr)
modules = modules.cuda()

evaluator = evaluate.EAEvaluator(w2i_en, i2w_en)

#%% Train models Embed Align

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

for epoch in range(saved_epoch, num_epochs):
    start_time = time.time()
    total_loss = 0
    for batch, (b_en, b_fr, b_l) in enumerate(batches):
        # Prepare batch
        data = zip(b_en, b_fr, b_l)
        data = sorted(data, key=lambda x: x[2], reverse=True)

        b_en, b_fr, b_l = map(list, zip(*data))
        b_en = pad_sequence(b_en, batch_first=True)
        b_l = torch.cuda.LongTensor(b_l)

        # Train
        opt.zero_grad()

        mus, sigmas = encoder(b_en, b_l)
        z = mus + normal.sample((mus.size(0), mus.size(1), embed_size)).cuda() * sigmas
        decoded_en = decoder_en(z)
        decoded_fr = decoder_fr(z)

        # Calculate Loss and train
        loss = ELBO_loss(b_en, b_fr, b_l, mus, sigmas, decoded_en, decoded_fr)
        total_loss += loss.data.item()
        loss.backward()
        opt.step()

        pace = (batch+1)/(time.time() - start_time)
        print('\r[Epoch {:03d}/{:03d}] Batch {:06d}/{:06d} [{:.1f}/s] '.format(epoch+1, num_epochs, batch+1, len(batches), pace), end='')

    # Calculate LST score
    total_loss /= len(batches)
    score = evaluator.lst(encoder)
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
