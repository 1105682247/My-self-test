# @title 默认标题文本
# 在第三层后添加pool实验结果
# !/usr/bin/env python
# coding: utf-8

# In[1]:


# prepare necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import init
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from ignite.metrics import Accuracy, Precision, Recall
import time, random, sys

# In[2]:


# data preprocessing

col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
             "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "score"]

attack_dict = {
    'normal': 'normal',

    'back': 'DoS',
    'land': 'DoS',
    'neptune': 'DoS',
    'pod': 'DoS',
    'smurf': 'DoS',
    'teardrop': 'DoS',
    'mailbomb': 'DoS',
    'apache2': 'DoS',
    'processtable': 'DoS',
    'udpstorm': 'DoS',

    'ipsweep': 'Probe',
    'nmap': 'Probe',
    'portsweep': 'Probe',
    'satan': 'Probe',
    'mscan': 'Probe',
    'saint': 'Probe',

    'ftp_write': 'R2L',
    'guess_passwd': 'R2L',
    'imap': 'R2L',
    'multihop': 'R2L',
    'phf': 'R2L',
    'spy': 'R2L',
    'warezclient': 'R2L',
    'warezmaster': 'R2L',
    'sendmail': 'R2L',
    'named': 'R2L',
    'snmpgetattack': 'R2L',
    'snmpguess': 'R2L',
    'xlock': 'R2L',
    'xsnoop': 'R2L',
    'worm': 'R2L',

    'buffer_overflow': 'U2R',
    'loadmodule': 'U2R',
    'perl': 'U2R',
    'rootkit': 'U2R',
    'httptunnel': 'U2R',
    'ps': 'U2R',
    'sqlattack': 'U2R',
    'xterm': 'U2R'
}

labels2 = ['normal', 'attack']
labels5 = ['normal', 'DoS', 'Probe', 'R2L', 'U2R']

df_train = pd.read_csv(r'E:/BaiduNetdiskDownload/data/NSL-KDD/KDDTrain+.txt', header=None, names=col_names)
df_test = pd.read_csv(r'E:/BaiduNetdiskDownload/data/NSL-KDD/KDDTest+.txt', header=None, names=col_names)

df_train = df_train.iloc[:, :42].copy()
df_test = df_test.iloc[:, :42].copy()

# 按照传统而言，将正常标记为0，攻击标记为1
df_train.loc[df_train.label != 'normal', 'label'] = 1
df_train.loc[df_train.label == 'normal', 'label'] = 0
df_test.loc[df_test.label != 'normal', 'label'] = 1
df_test.loc[df_test.label == 'normal', 'label'] = 0

df_train['label'] = pd.to_numeric(df_train['label'])
df_test['label'] = pd.to_numeric(df_test['label'])

x_train, y_train, x_test, y_test = df_train.iloc[:, :41].copy(), df_train['label'].copy(), df_test.iloc[:, :41].copy(), \
                                   df_test['label'].copy()

one_hot_x_train = x_train.iloc[:, 1:4].copy()
one_hot_x_test = x_test.iloc[:, 1:4].copy()
one_hot = OneHotEncoder()
one_hot_x_train = one_hot.fit_transform(one_hot_x_train)
one_hot_x_test = one_hot.transform(one_hot_x_test)

one_hot_x_train_df = pd.DataFrame(one_hot_x_train.toarray())
one_hot_x_test_df = pd.DataFrame(one_hot_x_test.toarray())
x_train = pd.concat([x_train, one_hot_x_train_df], axis=1)
x_train.drop(['flag', 'protocol_type', 'service'], axis=1, inplace=True)
# concat axis=1行对齐，axix=0列对齐
x_test = pd.concat([x_test, one_hot_x_test_df], axis=1)
# drop 删除指定标签数据，axis=0删除行，=1删除列，默认为0
x_test.drop(['flag', 'protocol_type', 'service'], axis=1, inplace=True)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# MIN-MAX归一化
minmax = MinMaxScaler()
x_train = minmax.fit_transform(x_train)
x_test = minmax.transform(x_test)

# 将训练集划分为7:3的train和test
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

x_train, y_train, x_test, y_test = torch.from_numpy(x_train).float(), torch.from_numpy(y_train), torch.from_numpy(
    x_test).float(), torch.from_numpy(y_test)


# In[3]:

# 设置s3
# 因此，为了重现结果，每次训练前都要先调用该函数
def set_seed(seed=1337):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# In[4]:
''' https://mp.weixin.qq.com/s?src=11&timestamp=1679972511&ver=4433&signature=O1*i6EALm9CALTHm221YCMDseOVNmluQxmXMe11FUTM0vXNwKiqqi4xG49j0E5btY7XMIaLbatKMugnwTz-nztY5jUeDT-hGFWSVs4IaHrTAExWIyk-0por7NV3hT21y&new=1
    https://mp.weixin.qq.com/s?src=11&timestamp=1679969390&ver=4433&signature=sU*YT-9T5YSEp2baWXKSNuoFcYggJRRcWFElekq6S3Ue1VB5qmVQ8-6BiZVfkJcpUop8YAOoLP*piZpbTOJ8jWpJ9s5sf7U2hQaWgUl*gxfT7hkU*cFeMOEMNDbCoGYX&new=1
    fpr假正率 = fp / (fp + tn)又称误判率
    真正率（TPR）=  TP/(TP+FN) 其实就是召回率
    TP – True Positive：实际为患癌，且判断为患癌（正确）
    FN – False Negative：实际为患癌，但判断为为患癌（错误）
    TN – True Negative：实际为未患癌，且判断为未患癌（正确）
    FP – False Positive：实际为未患癌，但判断为患癌（错误）'''


def getMetricsFromScoresbyRoc(labels, scores, print_now=False):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_ = roc_auc_score(labels, scores)

    # 通过相关公式概念，分别计算四种基础指标
    fp = fpr * (labels == 0).sum()  # 误判-
    tp = tpr * (labels == 1).sum()  # 正确+
    tn = (labels == 0).sum() - fp  # 正确-
    fn = (labels == 1).sum() - tp

    prescision = tp / (tp + fp)
    acc = (tn + tp) / labels.shape[0]

    f1 = (2 * tpr * prescision) / (tpr + prescision + 1e-20)  # F1值是精确率和召回率的调和均值，即F1=2PR/(P+R)，相当于精确率和召回率的综合评价指标。
    # 避免因为f1首元素为nan导致max和min等函数失效
    if np.isnan(f1[0]):
        f1[0] = 0
    bestF1 = np.argmax(f1)
    acc_, fpr_, precision_, recall_, f1_ = acc[bestF1], fpr[bestF1], prescision[bestF1], tpr[bestF1], f1[bestF1]
    if print_now:
        print(('acc:{:.6f}\t FPR:{:.6f}\t precision:{:.6f}\t recall:{:.6f}\t F1:{:.6f}\t AUC:{:.4f}\t')
              .format(acc_, fpr_, precision_, recall_, f1_, auc_), file=outFile, flush=True)
    return acc_, fpr_, precision_, recall_, f1_, auc_


# In[5]:


class Net_AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(122, 64)
        self.linear2 = nn.Linear(64, 32)
        self.delinear1 = nn.Linear(32, 64)
        self.delinear2 = nn.Linear(64, 122)
        self.lrelu = nn.LeakyReLU()
        self.rep_dim = 32

    def forward(self, x):
        x = self.linear1(x)
        x = self.lrelu(x)
        x_latent = self.linear2(x)  # 解码器输入
        x = self.lrelu(x_latent)
        x = self.delinear1(x)
        x = self.lrelu(x)
        x = self.delinear2(x)
        return x, x_latent


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(122, 64)
        self.linear2 = nn.Linear(64, 32)
        self.lrelu = nn.LeakyReLU()
        self.rep_dim = 32

    def forward(self, x):  # 前向传播
        x = self.linear1(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        return x


class Conv1dNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool1d(2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv1d(1, 16, 3, padding='same')
        self.bn1 = nn.BatchNorm1d(16, eps=1e-04, affine=False)
        self.conv2 = nn.Conv1d(16, 8, 3, padding='same')
        self.bn2 = nn.BatchNorm1d(8, eps=1e-04, affine=False)
        self.conv3 = nn.Conv1d(8, 4, 3, padding='same')
        self.bn3 = nn.BatchNorm1d(4, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(4 * 15, self.rep_dim)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


class Conv1dAE(nn.Module):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool1d(2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv1d(1, 16, 3, padding='same')
        self.bn1 = nn.BatchNorm1d(16, eps=1e-04, affine=False)
        self.conv2 = nn.Conv1d(16, 8, 3, padding='same')
        self.bn2 = nn.BatchNorm1d(8, eps=1e-04, affine=False)
        self.conv3 = nn.Conv1d(8, 4, 3, padding='same')
        self.bn3 = nn.BatchNorm1d(4, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(4 * 15, self.rep_dim)  # 全连接层
        self.fc2 = nn.Linear(self.rep_dim, 4 * 15)

        # Decoder
        self.deconv1 = nn.ConvTranspose1d(4, 4, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose1d(4, 8, 3, padding=1)
        self.bn5 = nn.BatchNorm1d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose1d(8, 16, 3, padding=1)
        self.bn6 = nn.BatchNorm1d(16, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose1d(16, 1, 3, padding=0)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))

        x = x.view(x.size(0), -1)
        latent = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(latent))
        x = x.view(x.size(0), 4, 15)

        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn6(x)), scale_factor=2)
        x = self.deconv4(x)

        x = x.view(x.size(0), -1)
        return x, latent


# In[6]:


class DeepSVDDTrainer():

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda:0',
                 n_jobs_dataloader: int = 0):

        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 2  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        self.acc_ = None
        self.precision_ = None
        self.recall_ = None
        self.F1_ = None
        self.fpr_ = None
        self.b_F1_epoch = -1
        self.b_F1 = 0
        self.list_precision = []
        self.list_recall = []
        self.list_f1 = []

    def train(self, net):

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_dataset = TensorDataset(x_train[y_train == 0], y_train[y_train == 0])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')  # 设置优化器

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)  # 设置衰减策略

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print('Initializing center c...', file=outFile, flush=True)
            self.c = self.init_center_c(train_loader, net)
            print('Center c initialized.', file=outFile, flush=True)

        # Training
        print('Starting training...', file=outFile, flush=True)
        start_time = time.time()
        net.train()

        for epoch in range(self.n_epochs):

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            # print last lr
            if epoch in self.lr_milestones:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_last_lr()[0]), file=outFile,
                      flush=True)
            for inputs, _ in train_loader:

                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    # loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                    loss = (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1

            scheduler.step()

            # test each epoch
            self.test(net)

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            self.b_F1, self.b_F1_epoch = (self.b_F1, self.b_F1_epoch) if self.b_F1 > self.F1_ else (self.F1_, epoch)
            print(('Epoch {}/{}\t Time:{:.3f}\t Loss:{:.8f}\t Test set AUC:{:.2f}\t'
                   + 'acc:{:.6f}\t  FPR:{:.6f}\t precision:{:.6f}\t recall:{:.6f}\t F1:{:.6f}\t R:{:.6f}\t')
                  .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches,
                          100. * self.test_auc, self.acc_, self.fpr_, self.precision_, self.recall_, self.F1_, self.R),
                  file=outFile, flush=True)

            # get the change of metrics
            self.list_precision.append(self.precision_)
            self.list_recall.append(self.recall_)
            self.list_f1.append(self.F1_)

        self.train_time = time.time() - start_time
        print('Training time: %.3f' % self.train_time, file=outFile, flush=True)
        print('Finished training. the biggest F1 is {:.6f} Epoch {}'.format(self.b_F1, self.b_F1_epoch + 1),
              file=outFile, flush=True)
        print('Training time: %.3f' % self.train_time)
        print('Finished training. the biggest F1 is {:.6f} Epoch {}'.format(self.b_F1, self.b_F1_epoch + 1))

        # fig_metric, ax_metric = plt.subplots()
        # x_local = list(range(self.n_epochs))
        # ax_metric.plot(x_local, self.list_precision, x_local, self.list_recall, x_local, self.list_f1)

        return net

    def test(self, net):

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        test_Dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_Dataset, batch_size=self.batch_size)

        # metrics ，如果是soft-boundary则使用sklearn.metrics计算
        if self.objective == 'soft-boundary':
            acc = Accuracy(device=self.device)
            precision = Precision(device=self.device)
            recall = Recall(device=self.device)
            F1 = precision * recall * 2 / (precision + recall + 1e-20)

        # Testing
        print('Starting testing...', file=outFile, flush=True)
        start_time = time.time()
        label_score = []
        net.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:

                inputs = inputs.to(self.device)
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - (self.R * 1.01) ** 2
                else:
                    scores = dist

                # Save triples of (idx, label, score) in a list
                label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        print('Testing time: %.3f' % self.test_time, file=outFile, flush=True)

        self.test_scores = label_score

        # Compute AUC
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        # 计算soft-boundary的metrics
        if self.objective == 'soft-boundary':
            pred = scores.copy()
            pred[pred < 0] = 0
            pred[pred > 0] = 1
            acc.update((torch.tensor(pred), torch.tensor(labels)))
            precision.update((torch.tensor(pred), torch.tensor(labels)))
            recall.update((torch.tensor(pred), torch.tensor(labels)))
            F1.update((torch.tensor(pred), torch.tensor(labels)))
            # 此处计算很有意思，都是库函数，但precision、recall、F1返回是tensor
            self.acc_, self.precision_ = acc.compute(), precision.compute().item()
            self.recall_, self.F1_ = recall.compute().item(), F1.compute().item()
            self.fpr_ = -1
            acc.reset(), precision.reset, recall.reset(), F1.reset()
        else:
            self.acc_, self.fpr_, self.precision_, self.recall_, self.F1_, _ = getMetricsFromScoresbyRoc(labels, scores)

    def init_center_c(self, train_loader: DataLoader, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for inputs, _ in train_loader:
                # get the inputs of the batch
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


# In[7]:

class AETrainer():

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        self.optimizer_name = 'adam'
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        self.test_scores = None
        self.list_f1 = []
        self.list_auc = []

    def train(self, ae_net):

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get train data loader
        train_dataset = TensorDataset(x_train[y_train == 0], y_train[y_train == 0])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        start_time = time.time()
        ae_net.train()

        for epoch in range(self.n_epochs):

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            # print last lr
            if epoch in self.lr_milestones:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_last_lr()[0]), file=outFile,
                      flush=True)
            for inputs, _ in train_loader:
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs, _ = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            scheduler.step()

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                  .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches), file=outFile, flush=True)
            self.test(ae_net)
            torch.save({'ae': ae_net.state_dict(), 'opt': optimizer.state_dict()}, prefolder + str(epoch + 1) + '.tar')

        pretrain_time = time.time() - start_time
        print('Pretraining time: %.3f' % pretrain_time, file=outFile, flush=True)
        print('Pretraining time: %.3f' % pretrain_time)
        print('Finished pretraining.', file=outFile, flush=True)

        # 比较AE的优劣
        self.list_f1, self.list_auc = np.array(self.list_f1), np.array(self.list_auc)
        if len(self.list_f1) > 1:
            lo = np.argmin(self.list_auc)
            self.list_f1 = np.delete(self.list_f1, lo)
            self.list_auc = np.delete(self.list_auc, lo)
        print('the max and avg f1 are {:.6f} and {:.6f}'.format(np.max(self.list_f1),
                                                                np.sum(self.list_f1) / len(self.list_f1)), file=outFile,
              flush=True)
        print('the max and avg auc are {:.4f} and {:.4f}'.format(np.max(self.list_auc),
                                                                 np.sum(self.list_auc) / len(self.list_auc)),
              file=outFile, flush=True)

        return ae_net

    def test(self, ae_net):

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get test data loader
        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        # Testing
        print('Testing autoencoder...', file=outFile, flush=True)
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        label_score = []
        ae_net.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs, _ = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)

                # Save triple of (label, score) in a list
                label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))

                loss_epoch += loss.item()
                n_batches += 1

        print('Test set Loss: {:.8f}'.format(loss_epoch / n_batches), file=outFile, flush=True)

        self.test_scores = label_score
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        auc = roc_auc_score(labels, scores)
        print('Test set AUC: {:.2f}%'.format(100. * auc), file=outFile, flush=True)
        _, _, _, _, f1, _ = getMetricsFromScoresbyRoc(labels, scores, print_now=True)
        self.list_auc.append(auc)
        self.list_f1.append(f1)
        test_time = time.time() - start_time
        print('Autoencoder testing time: %.3f' % test_time, file=outFile, flush=True)
        print('Autoencoder testing time: %.3f' % test_time)
        print('Finished testing autoencoder.', file=outFile, flush=True)


class JointTrainer(object):
    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda:0',
                 n_jobs_dataloader: int = 0, mu: float = 0.1):
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu
        self.mu = mu

        # Optimization parameters
        self.warm_up_n_epochs = 2  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        self.b_F1_epoch = -1
        self.b_F1 = 0
        self.list_precision = []
        self.list_recall = []
        self.list_f1 = []

    def joint_train(self, ae_net):

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get train data loader
        train_dataset = TensorDataset(x_train[y_train == 0], y_train[y_train == 0])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print('Initializing center c...')
            self.c = self.get_center_c(train_loader, ae_net)
            print('Center c initialized.')

        # Training
        print('Starting joint training...')
        start_time = time.time()
        ae_net.train()

        for epoch in range(self.n_epochs):

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            # print last lr
            if epoch in self.lr_milestones:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_last_lr()[0]), file=outFile,
                      flush=True)
            for inputs, _ in train_loader:

                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs, latent_out = ae_net(inputs)
                dist = torch.sum((latent_out - self.c) ** 2, dim=1)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                recon_loss = torch.mean(scores)

                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    # loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                    loss = (1 / self.nu) * torch.mean(
                        torch.max(torch.zeros_like(scores), scores)) + recon_loss * self.mu
                else:
                    loss = torch.mean(dist) + recon_loss * self.mu

                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1

            scheduler.step()

            # test each epoch
            self.joint_test(ae_net, time.time() - epoch_start_time, epoch, loss_epoch, n_batches)

        self.train_time = time.time() - start_time
        print('Training time: %.3f' % self.train_time, file=outFile, flush=True)
        print('Finished training. the biggest F1 is {:.6f} Epoch {}'.format(self.b_F1, self.b_F1_epoch + 1),
              file=outFile, flush=True)
        print('Training time: %.3f' % self.train_time)
        print('Finished training. the biggest F1 is {:.6f} Epoch {}'.format(self.b_F1, self.b_F1_epoch + 1))

        fig_metric, ax_metric = plt.subplots()
        x_local = list(range(self.n_epochs))
        ax_metric.plot(x_local, self.list_precision, x_local, self.list_recall, x_local, self.list_f1)

        return ae_net

    def joint_test(self, ae_net, train_time: float = 0.1, epoch: int = 1, loss_epoch: float = 0.1, n_batches: int = 0):
        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get test data loader
        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        # metrics ，如果是soft-boundary则使用sklearn.metrics计算
        if self.objective == 'soft-boundary':
            acc = Accuracy(device=self.device)
            precision = Precision(device=self.device)
            recall = Recall(device=self.device)
            F1 = precision * recall * 2 / (precision + recall + 1e-20)

        # Testing
        print('Starting joint testing...', file=outFile, flush=True)
        start_time = time.time()
        label_score = []
        ae_net.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:

                inputs = inputs.to(self.device)
                _, outputs = ae_net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - (self.R * 1.01) ** 2
                else:
                    scores = dist

                # Save triples of (idx, label, score) in a list
                label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))

        test_time = time.time() - start_time
        print('Testing joint time: %.3f' % test_time, file=outFile, flush=True)

        self.test_scores = label_score

        # Compute AUC
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        # 计算soft-boundary的metrics
        if self.objective == 'soft-boundary':
            pred = scores.copy()
            pred[pred < 0] = 0
            pred[pred > 0] = 1
            acc.update((torch.tensor(pred), torch.tensor(labels)))
            precision.update((torch.tensor(pred), torch.tensor(labels)))
            recall.update((torch.tensor(pred), torch.tensor(labels)))
            F1.update((torch.tensor(pred), torch.tensor(labels)))
            # 此处计算很有意思，都是库函数，但precision、recall、F1返回是tensor
            acc_, precision_ = acc.compute(), precision.compute().item()
            recall_, F1_ = recall.compute().item(), F1.compute().item()
            fpr_ = -1
            acc.reset(), precision.reset, recall.reset(), F1.reset()
        else:
            acc_, fpr_, precision_, recall_, F1_, _ = getMetricsFromScoresbyRoc(labels, scores)

        print('Finished joint testing.', file=outFile, flush=True)

        # log epoch statistics
        self.b_F1, self.b_F1_epoch = (self.b_F1, self.b_F1_epoch) if self.b_F1 > F1_ else (F1_, epoch)

        print(('Epoch {}/{}\t Time:{:.3f}\t Loss:{:.8f}\t Test set AUC:{:.2f}\t'
               + 'acc:{:.6f}\t  FPR:{:.6f}\t precision:{:.6f}\t recall:{:.6f}\t F1:{:.6f}\t R:{:.6f}\t')
              .format(epoch + 1, self.n_epochs, train_time, loss_epoch / n_batches,
                      100. * self.test_auc, acc_, fpr_, precision_, recall_, F1_, self.R),
              file=outFile, flush=True)

        # get the change of metrics
        self.list_precision.append(precision_)
        self.list_recall.append(recall_)
        self.list_f1.append(F1_)

    def get_center_c(self, train_loader: DataLoader, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data.传递球心"""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for inputs, _ in train_loader:
                # get the inputs of the batch
                inputs = inputs.to(self.device)
                _, outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


# In[8]:


class DeepSVDD(object):
    """A class for the Deep SVDD method.

    Attributes:
        objective: A string specifying the Deep SVDD objective (either 'one-class' or 'soft-boundary').
        nu: Deep SVDD hyperparameter nu (must be 0 < nu <= 1).
        R: Hypersphere radius R.
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network \phi.
        ae_net: The autoencoder network corresponding to \phi for network weights pretraining.
        trainer: DeepSVDDTrainer to train a Deep SVDD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SVDD network.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
    """

    def __init__(self, objective: str = 'one-class', nu: float = 0.1, mu: float = 1):
        """Inits DeepSVDD with one of the two objectives and hyperparameter nu."""

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.nu = nu
        self.mu = mu
        self.R = 0.0  # hypersphere radius R
        self.c = None  # hypersphere center c

        self.ae_net = Conv1dAE()  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.net = Conv1dNet()  # neural network \phi
        self.trainer = None
        self.optimizer_name = None

        self.joint_optimizer_name = None
        self.joint_trainer = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

    def train(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda:0',
              n_jobs_dataloader: int = 0):
        """Trains the Deep SVDD model on the training data."""

        self.optimizer_name = optimizer_name
        self.trainer = DeepSVDDTrainer(self.objective, self.R, self.c, self.nu, optimizer_name, lr=lr,
                                       n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size,
                                       weight_decay=weight_decay, device=device, n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.net = self.trainer.train(self.net)
        self.R = float(self.trainer.R.cpu().data.numpy())  # get float
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get list
        self.results['train_time'] = self.trainer.train_time

    def test(self, device: str = 'cuda:0', n_jobs_dataloader: int = 0):
        """Tests the Deep SVDD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSVDDTrainer(self.objective, self.R, self.c, self.nu,
                                           device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(self.net)
        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def pretrain(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda:0',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the Deep SVDD network \phi via autoencoder."""

        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(self.ae_net)
        # self.ae_trainer.test(self.ae_net)
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SVDD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def joint_train(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
                    lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6,
                    device: str = 'cuda:0', n_jobs_dataloader: int = 0):
        self.joint_optimizer_name = optimizer_name
        self.joint_trainer = JointTrainer(self.objective, self.R, self.c, self.nu, optimizer_name, lr=lr,
                                          n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size,
                                          weight_decay=weight_decay, device=device, n_jobs_dataloader=n_jobs_dataloader,
                                          mu=self.mu)
        # Get the model
        self.ae_net = self.joint_trainer.joint_train(self.ae_net)
        self.R = float(self.joint_trainer.R.cpu().data.numpy())  # get float
        self.c = self.joint_trainer.c.cpu().data.numpy().tolist()  # get list


# In[9]:


# fold
import sys

outFile = open('E:/BaiduNetdiskDownload/data/output/output_prelr_16-8-4_32', 'w')
# outFile = sys.stdout
prefolder = r'E:/BaiduNetdiskDownload/data/output/state_dict_prelr_16-8-4_32/'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.seterr(divide='ignore', invalid='ignore')
list_best_F1 = []
list_best_F1_epoch = []

set_seed()
deep_svdd = DeepSVDD('one-class', mu=0.1)
deep_svdd.pretrain(lr_milestones=(2,), n_epochs=140)
for preepoches in range(140):
    set_seed()
    deep_svdd = DeepSVDD('one-class', mu=0.1)
    deep_svdd.net.load_state_dict(torch.load(prefolder + str(preepoches + 1) + '.tar')['ae'], strict=False)
    set_seed()
    deep_svdd.train(lr_milestones=(), n_epochs=30)
    list_best_F1.append(deep_svdd.trainer.b_F1)
    list_best_F1_epoch.append(deep_svdd.trainer.b_F1_epoch)

# 打印网格搜索实验结果信息
print(list_best_F1, file=outFile)
print(list_best_F1_epoch, file=outFile)
print('the best pretrain epoches and F1 are {} and {:.6f}'.format(np.argmax(list_best_F1) + 1, max(list_best_F1)),
      file=outFile, flush=True)
print('the best pretrain epoches and F1 are {} and {:.6f}'.format(np.argmax(list_best_F1) + 1, max(list_best_F1)))
_, ax_best_f1 = plt.subplots()
ax_best_f1.plot(list_best_F1)
