import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin, init
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import DynamicEdgeConv, global_max_pool
from torch_geometric.utils import intersection_and_union as i_and_u
from torch_geometric.utils import to_dense_batch
from DiffGCN import DiffGCNBlock
import sys
from pointnet2_classification import MLP
from shapenet2 import ShapeNet2
from mgpool import mgunpool
import numpy as np
from torch.autograd import Variable

category = 'Airplane'
print(torch.cuda.get_device_capability())
expname = 'airplane.pth'
print("Exp name:", expname)

path = '/home/cluster/users/erant_group/shapenet_segmentation'
savepath = '/home/cluster/users/erant_group/diffops/' + expname
batchSize = 20
npoints = 2048

if "slurm" in sys.argv:
    path = '/home/eliasof/meshfit/pytorch_geometric/data/shapenet'
    savepath = '/home/eliasof/meshfit/pytorch_geometric/checkpoints/' + expname

train_transform = T.Compose([
    T.RandomTranslate(0.01),
    T.RandomRotate(15, axis=0),
    T.RandomRotate(15, axis=1),
    T.RandomRotate(15, axis=2),
    T.FixedPoints(npoints, replace=False)
])
pre_transform, transform = T.NormalizeScale(), T.FixedPoints(npoints, replace=True)
train_dataset = ShapeNet2(path, categories=str(category), split='trainval', transform=transform,
                          pre_transform=pre_transform)
test_dataset = ShapeNet2(path, categories=str(category), split='test', transform=transform,
                         pre_transform=pre_transform)
train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True,
                          num_workers=6, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False,
                         num_workers=6)
eps = 1e-10


def one_hot_embedding(labels, num_classes=16):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels].cuda()


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d).reshape((1, d, d)).repeat(batchsize, 1, 1)
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1).contiguous()) - I, dim=(1, 2), p=2))
    return loss


def stnknn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, normals=None, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = stnknn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = stnknn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if normals is None:
        feature = torch.cat((x, feature - x), dim=3).permute(0, 3, 1, 2)
    else:
        normals = normals.contiguous()
        normals = normals.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((x, feature - x, normals), dim=3).permute(0, 3, 1, 2)

    return feature  # (batch_size, 2*num_dims, num_points, k)


class Transform_Net(nn.Module):
    def __init__(self, args=None, normals=False):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3
        if normals:
            self.initialFeatSize = 6
            self.outputSize = 3
        else:
            self.initialFeatSize = 2 * 128
            self.outputSize = 128
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(self.initialFeatSize, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, self.outputSize * self.outputSize)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(self.outputSize, self.outputSize))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)  # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)  # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)  # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, self.outputSize, self.outputSize)  # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class Net(torch.nn.Module):
    def __init__(self, out_channels, k=10, aggr='max'):
        super(Net, self).__init__()
        self.transform_net = STN3d()
        self.k = k
        self.conv0 = DiffGCNBlock(3, 64, 20, 1)
        self.conv1 = DiffGCNBlock(64, 64, 5, 2, pool=True)
        self.conv2 = DiffGCNBlock(64, 64, 5, 2, pool=True)
        self.conv3 = DiffGCNBlock(64, 128, 5, 2, pool=True)

        self.lin1 = MLP([64*3 + 128, 2048])

        self.mlp = Seq(MLP([2048 + 64*3 + 128 + 16, 512]), Dropout(0.5), MLP([512, 256]),
                       Dropout(0.5), MLP([256, 128]), Dropout(0.5), Lin(128, out_channels))

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        cat = data.category
        x0 = pos
        x0, _ = to_dense_batch(x0, batch)
        x0 = x0.transpose(2, 1).contiguous()
        t1 = self.transform_net(x0).contiguous()
        x0 = x0.transpose(2, 1).contiguous()
        x0 = torch.bmm(x0, t1).contiguous()
        x0 = x0.view(x0.shape[0] * x0.shape[1], -1).contiguous()
        pos = x0

        # Open conv. and feature transform:
        origbatch = batch.clone()
        x0, pos, batch = self.conv0(x0, pos, batch)

        x1, pos, batch, pooldata1 = self.conv1(x0, pos, batch)

        x2, pos, batch, pooldata2 = self.conv2(x1, pos, batch)

        x3, pos, batch, pooldata3 = self.conv3(x2, pos, batch)

        # Unpool:
        x3 = mgunpool(mgunpool(mgunpool(x3, *pooldata3), *pooldata2), *pooldata1)
        x2 = mgunpool(mgunpool(x2, *pooldata2), *pooldata1)
        x1 = mgunpool(x1, *pooldata1)

        out = self.lin1(torch.cat([x0, x1, x2, x3], dim=1))

        out = global_max_pool(out, origbatch)
        out = out.repeat_interleave(repeats=npoints, dim=0)
        onehot = one_hot_embedding(cat)
        onehot = onehot.repeat_interleave(repeats=npoints, dim=0)

        out = torch.cat([x0, x1, x2, x3, out, onehot], dim=1)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1), t1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(train_dataset.num_classes, k=20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
print(train_dataset.num_classes)
prev_test_acc = 0.0
if "continue" in sys.argv:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)
    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = 0
    acc = checkpoint['acc']
    prev_test_acc = checkpoint['test_acc']
    print("Continuing... current test acc:", prev_test_acc)


def train():
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    for i, data in enumerate(train_loader):

        data = data.to(device)
        optimizer.zero_grad()
        out, t1 = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.max(dim=1)[1].eq(data.y).sum().item()
        total_nodes += data.num_nodes

        if (i + 1) % 10 == 0:
            print('[{}/{}] Loss: {:.4f}, Train Accuracy: {:.4f}'.format(
                i + 1, len(train_loader), total_loss / 10,
                correct_nodes / total_nodes), flush=True)
            total_loss = correct_nodes = total_nodes = 0


def test(loader):
    model.eval()
    loader.dataset.y_mask = loader.dataset.y_mask[:, 0:test_dataset.num_classes]

    correct_nodes = total_nodes = 0
    intersections, unions, categories = [], [], []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out, _, _ = model(data)
        pred = out.max(dim=1)[1]
        correct_nodes += pred.eq(data.y).sum().item()
        total_nodes += data.num_nodes
        i, u = i_and_u(pred, data.y, test_dataset.num_classes, data.batch)
        intersections.append(i.to(torch.device('cpu')))
        unions.append(u.to(torch.device('cpu')))
        categories.append(data.category.to(torch.device('cpu')))

    category = torch.cat(categories, dim=0)
    unique_cats = torch.unique(category)
    intersection = torch.cat(intersections, dim=0)
    union = torch.cat(unions, dim=0)

    hist = torch.zeros(16)
    for j in range(len(loader.dataset)):
        hist[category[j]] += 1
    print(hist)

    ious = [[] for _ in range(16)]
    for j in range(len(loader.dataset)):
        i = intersection[j, loader.dataset.y_mask[category[j]]]
        u = union[j, loader.dataset.y_mask[category[j]]]
        iou = i.to(torch.float) / u.to(torch.float)
        iou[torch.isnan(iou)] = 1
        ious[category[j]].append(iou.mean().item())

    for cat in range(16):
        ious[cat] = torch.tensor(ious[cat]).mean().item()
    print("IOUS:", ious)
    print(unique_cats)
    ious = torch.tensor(ious)
    return correct_nodes / total_nodes, torch.tensor(ious[unique_cats]).mean().item()


print(model)
for epoch in range(1, 1001):
    train()
    acc, iou = test(test_loader)
    print('Epoch: {:02d}, Acc: {:.4f}, IoU: {:.4f}'.format(epoch, acc, iou))
    if prev_test_acc < iou:
        prev_test_acc = iou
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': acc,
            'test_acc': iou,
        }, savepath)
    print("Best IOU so far:", prev_test_acc)
