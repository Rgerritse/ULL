# %% Imports
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

# %% Parameters
embed_size = 300
learning_rate = 0.01
num_epochs = 2
batch_size = 1
window = 2

# %% Dataset creation
with open("data/wa/dev.en") as f:
    tokens = f.read().split()

vocab = list(set(tokens))
vocab_size = len(vocab)
w2i = {k: v for v, k in enumerate(vocab)}
i2w = {v: k for v, k in enumerate(vocab)}

data = []
labels = []
for i in range(len(tokens)):
    for j in range(i-window, i+window+1):
        if i != j and j>=0 and j<len(tokens):
            data.append(w2i[tokens[j]])
            labels.append(w2i[tokens[i]])

X = Variable(torch.LongTensor(data))
T = Variable(torch.LongTensor(labels))

train_data = torch.utils.data.TensorDataset(X, T)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)

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

# %% Train
net = Net(vocab_size, embed_size)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), learning_rate)

for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        if i % 10 == 0:
            print('\rEpoch {} {}'.format(epoch, i+1), end='')

        opt.zero_grad()
        outputs = net(inputs)

        loss = loss_fn(outputs, targets)
        opt.step()

# %% Get embeddings
embeddings = net.embeddings.weight
