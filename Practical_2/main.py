# %% Imports
import torch, os, time
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from evaluate import Evaluator
from collections import defaultdict
import string

# %% Parameters
embed_size = 100
learning_rate = 0.01
num_epochs = 10
batch_size = 512
window = 2
model_name = 'skipgram'
top_size = 1000 # Top n words to use for training, all other words are mapped to <unk>, use None if you do not want to map any word to <unk>
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

# %% Dataset creation
with open("data/hansards/training.en") as f:
    sentences = [line.lower().split() for line in f.readlines()]

w2i = {k: v for v, k in enumerate(vocab)}
i2w = {v: k for v, k in enumerate(vocab)}

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
embeddings = net.embeddings.weight.data
print(embeddings)
time.time() - start_time
