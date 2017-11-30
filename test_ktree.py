import time,resource
import numpy as np
from ktree import ktree,KTreeOptions,utils

def ktree_clustering(order,selection,ds):
    # order = 10 # order: order of K-tree default:6 i.e. the maximum number of a node's children.  Default order is 5.
    # ds = np.loadtxt("data/data.csv", skiprows=1, delimiter='\t',usecols=range(1,3))
    # ds = np.loadtxt("data/generated_data_1000.csv")
    # print ('Shape: ',ds.shape)
    N,d = ds.shape # N: number of examples, d: dimension of examples


    options = KTreeOptions() #sets options for ktree_clustering
    options.order = order
    options.weighted = True
    options.distance = "euclidean"
    options.reinsert = True

    t0 = time.time()
    k = ktree(data=ds, options=options)   # returns k-Node object
    timeElapsed = time.time() - t0

    print "ktree Number of clusters: ",len(k.root),' in ',timeElapsed #k.root has all clusters with list of clusters in one key of dict.
    # print
    # print("Max_ram_usage: %.2f MB.\n" % (float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024))

    utils.save_ktree(k,'save_ktree.pickle') #Save Pickle
    # k = utils.load_ktree('save_ktree.pickle') #Read Pickle
    results = {}
    results['n_clusters']=len(k.root)
    # results['timeElapsed']=timeElapsed

    return results
dataset = np.loadtxt("data/generated_data_10000.csv")  #loads data into numpy n dimensional array
a =ktree_clustering(1000,1,dataset)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

model = nn.DataParallel(Net())
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.NLLLoss().cuda()

model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    input_var = Variable(data.cuda())
    target_var = Variable(target.cuda())

    print('Getting model output')
    output = model(input_var)
    print('Got model output')

    loss = criterion(output, target_var)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('Finished')
