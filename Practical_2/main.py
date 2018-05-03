# %% Imports
import torch, os, time
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from evaluate import Evaluator

# %% Parameters
embed_size = 100
learning_rate = 0.01
num_epochs = 10
batch_size = 512
window = 2
model_name = 'skipgram'

# %% Data selection
word_set = set()

with open("data/lst/lst.gold.candidates") as fc:
    for line in fc.readlines():
        split_line = line.strip().split('::')
        word_set.add(split_line[0].split('.')[0])
        for word in split_line[1].split(';'):
            if len(word.split()) == 1:
                word_set.add(word)

with open("data/lst/lst_test.preprocessed") as fp:
    for line in fp.readlines():
        split_line = line.strip().split()
        i = int(split_line[2])
        sentence = split_line[3:]
        for j in range(i-window, i+window+1):
            if j>=0 and j<len(sentence):
                word_set.add(sentence[j])

# %% Dataset creation
with open("data/hansards/training.en") as f:
    tokens = [line.split() for line in f.readlines()]

vocab = list(set([token for sentence in tokens for token in sentence]))
vocab_size = len(vocab)
w2i = {k: v for v, k in enumerate(vocab)}
i2w = {v: k for v, k in enumerate(vocab)}

data = []
labels = []
for sentence in tokens:
    for i in range(len(sentence)):
        for j in range(i-window, i+window+1):
            if i != j and j>=0 and j<len(sentence):
                if sentence[i] in word_set:
                    data.append(w2i[sentence[i]])
                    labels.append(w2i[sentence[j]])

# %% Network Definition
class Net(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Net, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        out = self.embeddings(data)
        out = self.fc1(out)
        out = self.softmax(out)
        return out

# %% Load Evaluator
evaluator = Evaluator(w2i, i2w, window)

# %% Train
train_data = torch.utils.data.TensorDataset(torch.LongTensor(data), torch.LongTensor(labels))
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

net = Net(vocab_size, embed_size).cuda()
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), learning_rate)

# Check for saved checkpoint
saved_epoch = 0
checkpoints = [cp for cp in sorted(os.listdir('checkpoints')) if model_name in cp]
if checkpoints:
    state = torch.load('checkpoints/{}'.format(checkpoints[-1]))
    saved_epoch = state['epoch'] + 1
    net.load_state_dict(state['state_dict'])
    opt.load_state_dict(state['optimizer'])

for epoch in range(saved_epoch, num_epochs):
    start_time = time.time()
    num_batches = len(train_loader)
    for batch, (inputs, targets) in enumerate(train_loader):
        inputs = Variable(inputs).cuda()
        targets = Variable(targets).cuda()

        opt.zero_grad()
        outputs = net(inputs)

        loss = loss_fn(outputs, targets)
        opt.step()

        pace = (batch+1)/(time.time() - start_time)
        print('\r[Epoch {:03d}/{:03d}] Batch {:06d}/{:06d} [{:.1f}/s] '.format(epoch+1, num_epochs, batch+1, num_batches, pace), end='')

    # Calculate LST score
    score = evaluator.lst(net.embeddings.weight.data)
    print('Time: {:.1f}s Score: {:.6f}'.format(time.time() - start_time, score))

    # Save checkpoint
    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': opt.state_dict()
    }
    torch.save(state, 'checkpoints/{}-{}'.format(model_name, epoch))

# %% Get embeddings
<<<<<<< HEAD
embeddings = net.embeddings.weight
vocab_size

#%%
=======
embeddings = net.embeddings.weight.data
print(embeddings)
time.time() - start_time
>>>>>>> da9b9228c3e2b6d9560f3bc80edf3d562aac1b31
