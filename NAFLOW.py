import math

from matplotlib.patches import Rectangle
from sklearn.neighbors import KDTree
import random
import warnings
import torch
from sklearn import svm
import numpy as np
import pandas as pd
from scipy.optimize import minimize, fsolve
from scipy.spatial.distance import cdist
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from scipy.optimize import minimize
from sklearn.svm import SVC
from torch import nn, optim
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import accuracy_score as Accuracy
from sklearn.metrics import precision_score as Precision
from sklearn.metrics import recall_score as Recall
from sklearn.metrics import f1_score as F1_measure
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as AUC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

"----------------------------集成类----------------------------"


class NaturalNeighborSearch:
    def __init__(self, X: np.array, y: np.array = None):
        self.data = X
        self.labels = y if y is not None else np.zeros(len(X))
        self.nan_edges = set()  # 自然邻居边集合
        self.nan_num = {}  # 每个点的自然邻居数量
        self.knn = {}  # 每个点的k近邻
        self.nan_neighbors = {}  # 每个点的自然邻居
        self.r_optimal = 1  # 最优的r值
        self.history = []  # 记录迭代历史

    def _initialize(self):
        """初始化数据结构"""
        n_samples = len(self.data)
        for i in range(n_samples):
            self.knn[i] = set()
            self.nan_neighbors[i] = set()
            self.nan_num[i] = 0

    def find_knn(self, point_idx: int, r: int, tree: KDTree):
        """找到指定点的r近邻（排除自身）"""
        if r >= len(self.data):
            r = len(self.data) - 1

        distances, indices = tree.query([self.data[point_idx]], r + 1)
        # 移除自身，返回r个最近邻
        return np.array([idx for idx in indices[0] if idx != point_idx])

    def count_zero_neighbors(self):
        """统计没有自然邻居的点的数量"""
        return sum(1 for count in self.nan_num.values() if count == 0)

    def search_natural_neighbors(self, max_r: int = 10, convergence_threshold: float = 0.01):
        """
        搜索自然邻居

        参数:
            max_r: 最大搜索半径
            convergence_threshold: 收敛阈值

        返回:
            zero_neighbors: 没有自然邻居的点索引
            nan_num: 每个点的自然邻居数量
        """
        if len(self.data) <= 1:
            return [], {}

        tree = KDTree(self.data)
        self._initialize()

        r = 2
        prev_zero_count = len(self.data)
        converged = False

        while not converged and r <= max_r:
            # 当前迭代的新边
            new_edges = 0

            for i in range(len(self.data)):
                knn_indices = self.find_knn(i, r, tree)
                if len(knn_indices) == 0:
                    continue

                # 第r个最近邻
                rth_neighbor = knn_indices[-1]
                self.knn[i].add(rth_neighbor)

                # 检查互反关系
                if i in self.knn[rth_neighbor] and (i, rth_neighbor) not in self.nan_edges:
                    self.nan_edges.add((i, rth_neighbor))
                    self.nan_edges.add((rth_neighbor, i))
                    self.nan_neighbors[i].add(rth_neighbor)
                    self.nan_neighbors[rth_neighbor].add(i)
                    self.nan_num[i] += 1
                    self.nan_num[rth_neighbor] += 1
                    new_edges += 1

            current_zero_count = self.count_zero_neighbors()

            # 记录迭代历史
            self.history.append({
                'r': r,
                'zero_neighbors': current_zero_count,
                'total_edges': len(self.nan_edges) // 2,
                'new_edges': new_edges
            })

            # 收敛条件
            zero_ratio = current_zero_count / len(self.data)  ##自然邻居为0的是数量占总数据数量的比例
            improvement_ratio = (prev_zero_count - current_zero_count) / prev_zero_count if prev_zero_count > 0 else 0

            if (zero_ratio < convergence_threshold or
                    improvement_ratio < 0.01 or  # 改善很小
                    current_zero_count == 0):
                converged = True
                self.r_optimal = r
            else:
                prev_zero_count = current_zero_count
                r += 1

        zero_neighbors = [i for i, count in self.nan_num.items() if count == 0]
        # print(self.knn)
        # print(self.nan_neighbors)
        return zero_neighbors, self.nan_neighbors, self.knn


class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, input_layer, num_units=128):
        super(MLPDiffusion, self).__init__()

        # 定义每一层
        self.fc1 = nn.Linear(input_layer + 2, num_units)  # 输入层到第一个隐藏层
        self.relu1 = nn.ReLU()  # 第一个激活函数
        self.fc2 = nn.Linear(num_units, 256)  # 第一个隐藏层到第二个隐藏层
        self.relu2 = nn.ReLU()  # 第二个激活函数
        self.fc3 = nn.Linear(256, num_units)  # 第二个隐藏层到第三个隐藏层
        self.relu3 = nn.ReLU()  # 第三个激活函数
        self.fc4 = nn.Linear(num_units, input_layer)  # 隐藏层到输出层

    def forward(self, x, weight, t):
        x = torch.cat((x, weight,t), dim=1)
        x = self.fc1(x)  # [batch_size, num_units]
        x = self.relu1(x)
        x = self.fc2(x)  # [batch_size, 256]
        x = self.relu2(x)

        # 第二个隐藏层
        x = self.fc3(x)  # [batch_size, num_units]
        x = self.relu3(x)

        # 输出层
        x = self.fc4(x)  # [batch_size, 2]
        return x


class LambdaOptimizer(nn.Module):
    def __init__(self, input_dim=2):
        super(LambdaOptimizer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lamda, t):
        x = torch.cat((x, lamda, t), dim=1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x



# def vectorized_nearest_positive_loss(anchors, positives, negatives, temperature=0.1):
#     all_samples = torch.cat([positives, negatives], dim=0)
#     sim_matrix = torch.cosine_similarity(
#         anchors.unsqueeze(1),
#         all_samples.unsqueeze(0),
#         dim=2
#     )
#
#     scaled_sim_matrix = sim_matrix / temperature
#     num_positives = positives.shape[0]
#     scaled_positive_sims = scaled_sim_matrix[:, :num_positives]
#     scaled_nearest_pos_sims, _ = torch.max(scaled_positive_sims, dim=1)
#     log_denominator = torch.logsumexp(scaled_sim_matrix, dim=1)
#     log_probs = scaled_nearest_pos_sims - log_denominator
#     loss = -torch.mean(log_probs)
#     return loss

def vectorized_nearest_positive_loss(anchors, positives, negatives, temperature=0.1):
    all_samples = torch.cat([positives, negatives], dim=0)
    sim_matrix = torch.cosine_similarity(
        anchors.unsqueeze(1),
        all_samples.unsqueeze(0),
        dim=2
    )
    scaled_sim_matrix = sim_matrix / temperature
    num_positives = positives.shape[0]
    scaled_positive_sims = scaled_sim_matrix[:, :num_positives]
    # scaled_nearest_pos_sims, _ = torch.max(scaled_positive_sims, dim=1)
    # log_denominator = torch.logsumexp(scaled_sim_matrix, dim=1)
    log_sum_pos = torch.logsumexp(scaled_positive_sims, dim=1)
    # 分母：所有样本
    log_denominator = torch.logsumexp(scaled_sim_matrix, dim=1)
    log_probs = log_sum_pos - log_denominator
    loss = -torch.mean(log_probs)
    return loss

def distribute_by_weight_probability(weights, total):
    weights = np.array(weights, dtype=float)
    # 归一化
    p = weights / weights.sum()
    # 计算每个类别的基础数量
    counts = np.floor(p * total).astype(int)
    remainder = total - counts.sum()

    # 剩余名额按照权重概率分配
    if remainder > 0:
        # 使用权重作为概率进行抽样
        additional_indices = np.random.choice(
            len(weights),
            size=remainder,
            p=p,  # 按权重概率抽样
            replace=True
        )
        for idx in additional_indices:
            counts[idx] += 1
    # print(counts.sum())
    return counts


class NAFLOW:
    def __init__(self, data_path, epochs, num_steps, batch_size_ratio=0.5):
        self.epochs = epochs
        self.data_path = data_path
        self.num_steps = num_steps
        self.batch_size_ratio = batch_size_ratio  # batch_size相对于少数类样本数量的比例
        self.load_data()

    def load_data(self):
        data1 = pd.read_csv(self.data_path, header=None)
        y_label = data1.iloc[:, -1].astype(str).str.strip()
        le = LabelEncoder()
        le.fit(y_label)
        labeldata = np.array(le.transform(y_label)).reshape(-1, 1)
        columnstestdata = data1.shape[1] - 1
        data2 = pd.concat([data1.iloc[:, 0:columnstestdata], pd.DataFrame(labeldata)], axis=1)
        data2.columns = [i for i in range(0, columnstestdata + 1)]

        self.data = data2
        self.X = data2.values[:, :-1]
        self.y = data2.values[:, -1]

        # scaler = MinMaxScaler()
        # self.X = scaler.fit_transform(self.X)

        # Z-Score标准化
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        self.minority_data = self.X[self.y == 1]
        self.majority_data = self.X[self.y == 0]

        # 计算batch_size为少数类数据数量的一半
        base_batch_size = max(1, int(len(self.minority_data) * self.batch_size_ratio))
        nn_search = NaturalNeighborSearch(self.X, self.y)
        zero_neighbors, nan_neighbors, knn_neighbors = nn_search.search_natural_neighbors(max_r=6)

        ## 挑选出少数类数据自然近邻中包含多数类的少数类点 与 其中的多数类点
        overlap_maj = []  ## 存储重叠区域的多数类数据的索引
        for i in range(self.X.shape[0]):
            if self.y[i] == 1 and any(self.y[x] != 1 for x in list(nan_neighbors[i])):
                for x in list(nan_neighbors[i]):
                    if self.y[x] != 1:
                        overlap_maj.append(x)
        overlap_maj = np.unique(overlap_maj)

        # 计算overlap_maj的一半
        overlap_half = max(1, len(overlap_maj))

        # 取两者中的较小值作为最终的batch_size
        # self.batch_size = min(base_batch_size, overlap_half)
        self.batch_size = base_batch_size

        print(f"少数类样本数量: {len(self.minority_data)}")
        print(f"设置的batch_size: {self.batch_size}")

        # 创建数据加载器
        self._create_data_loaders()

    def _create_data_loaders(self):
        """创建批量数据加载器"""
        # 少数类数据
        minority_tensor = torch.tensor(self.minority_data, dtype=torch.float32)
        self.minority_loader = torch.utils.data.DataLoader(
            minority_tensor,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        # 多数类数据（用于负样本）
        majority_tensor = torch.tensor(self.majority_data, dtype=torch.float32)
        self.majority_loader = torch.utils.data.DataLoader(
            majority_tensor,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        print(f"少数类批次数量: {len(self.minority_loader)}")
        print(f"多数类批次数量: {len(self.majority_loader)}")

    def sample(self, model, num_generate, input_dim, weight, steps=100):
        dt = 1.0 / steps
        traj = []  # to store the trajectory
        z = torch.randn(num_generate, input_dim).to(device)

        for i in range(steps):
            t = torch.ones((z.shape[0], 1)).to(device) * (i / steps)
            v = model(z, weight, t)
            z = z + v * dt
            traj.append(z)
        # print(len(traj))
        return traj

    def train(self):
        ### 首先把边界数据寻找出来
        nn_search = NaturalNeighborSearch(self.X, self.y)
        zero_neighbors, nan_neighbors, knn_neighbors = nn_search.search_natural_neighbors(max_r=6)

        ## 挑选出少数类数据自然近邻中包含多数类的少数类点 与 其中的多数类点
        overlap_maj = []  ## 存储重叠区域的多数类数据的索引
        overlap_min = []  ## 存储重叠区域的多数类数据的索引
        for i in range(self.X.shape[0]):
            if self.y[i] == 1 and any(self.y[x] != 1 for x in list(nan_neighbors[i])):
                overlap_min.append(i)
                for x in list(nan_neighbors[i]):
                    if self.y[x] != 1:
                        overlap_maj.append(x)
        overlap_maj = np.unique(overlap_maj)

        print(f"重叠区域多数类样本数量: {len(overlap_maj)}")
        maj_no_noise = []
        for i in range(self.X.shape[0]):
            if self.y[i] == 0 and i not in overlap_maj:
                maj_no_noise.append(i)
        self.majority_data_no_noise = self.X[maj_no_noise]    ###获取去掉噪声的多数类数据  为了以后添加到损失中处理
        # print(self.majority_data_no_noise.shape)
        # 创建重叠区域多数类数据加载器
        if len(overlap_maj) > 0:
            ##主要多数类区域的数据设置 尽量保证数据的一致性  设置重叠区域的样本是少数类样本的偶数倍
            if len(overlap_maj) == (self.minority_data.shape[0]):  ##说明重叠区域的数据是是少数类样本的偶数倍
                overlap_maj_tensor1 = torch.tensor(self.X[overlap_maj], dtype=torch.float32)
                self.overlap_maj_loader = torch.utils.data.DataLoader(
                    overlap_maj_tensor1,
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=False
                )
                print(f"重叠区域多数类批次数量: {len(self.overlap_maj_loader)}")
                self.overlap = torch.tensor(self.X[overlap_maj], dtype=torch.float32).cpu().numpy()
            if len(overlap_maj) != (self.minority_data.shape[0]):  ##说明重叠区域的数据是是少数类样本的倍数
                numner1 = len(overlap_maj)
                if numner1<self.minority_data.shape[0]:##当重叠数据少于少数类数据的数量的时候  计算离着重叠数据最近的多数类数据进行填补
                    knn = NearestNeighbors(n_neighbors=self.minority_data.shape[0] - numner1)
                    knn.fit(self.majority_data)
                    query_samples = self.X[overlap_maj]
                    # print(query_samples)
                    distances, indices = knn.kneighbors(query_samples)
                    # print(self.majority_data[indices[0][0:3]])
                    overlap_maj2 = np.concatenate([overlap_maj, indices[0][0:]], axis=0)
                    overlap_maj_tensor2 = torch.tensor(self.X[overlap_maj2], dtype=torch.float32)
                    # print(overlap_maj_tensor2.shape)
                    self.overlap_maj_loader = torch.utils.data.DataLoader(
                        overlap_maj_tensor2,
                        batch_size=self.batch_size,
                        shuffle=False,
                        drop_last=False
                    )
                    self.overlap = overlap_maj_tensor2.cpu().numpy()
                    print(f"重叠区域多数类批次数量: {len(self.overlap_maj_loader)}")

                if numner1>self.minority_data.shape[0]:  ##如果重叠样本是多于少数类样本的个数的时候 去除部分重叠样本
                    knn = NearestNeighbors(n_neighbors=self.minority_data.shape[0])
                    knn.fit(self.X[overlap_maj])
                    query_samples = self.X[overlap_min]
                    # print('ss',query_samples)
                    distances, indices = knn.kneighbors(query_samples)
                    # print('aa',self.X[overlap_maj][indices[0][0:2]])
                    # overlap_maj3 = overlap_maj + indices[0][0:]
                    overlap_maj3 = indices[0][0:]
                    overlap_maj_tensor3 = torch.tensor(self.X[overlap_maj3], dtype=torch.float32)
                    # print(overlap_maj_tensor3.shape)
                    self.overlap_maj_loader = torch.utils.data.DataLoader(
                        overlap_maj_tensor3,
                        batch_size=self.batch_size,
                        shuffle=False,
                        drop_last=False
                    )
                    self.overlap = overlap_maj_tensor3.cpu().numpy()
                    print(f"重叠区域多数类批次数量: {len(self.overlap_maj_loader)}")

        else:
            print("没有找到重叠区域多数类样本，将使用普通多数类样本")
            knn = NearestNeighbors(n_neighbors=self.minority_data.shape[0])   ###此时没有重叠的样本  就不需要设置邻居是少数类数量加1来避免自身样本
            knn.fit(self.majority_data)
            query_samples = self.minority_data
            distances, indices = knn.kneighbors(query_samples)
            overlap_maj_tensor3 = torch.tensor(self.X[indices[0][0:]], dtype=torch.float32)
            # print(overlap_maj_tensor3.shape)
            self.overlap_maj_loader = torch.utils.data.DataLoader(
                overlap_maj_tensor3,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False
            )
            print(f"重叠区域多数类批次数量: {len(self.overlap_maj_loader)}")
            self.overlap_maj_loader = self.majority_loader
            self.overlap = overlap_maj_tensor3.cpu().numpy()

        ###根据自然近邻计算局部密度权重作为条件
        self.weight = []
        for i in range(self.X.shape[0]):
            if self.y[i] == 1:
                dist_NaN_maj = 0
                dist_NaN_min = 0
                NaN_neighbors = list(nan_neighbors[i])  ##寻找到第i个少数类数据的自然邻居
                NaN_maj = [i for i in NaN_neighbors if self.y[i] == 0]  ##寻找到第i个少数类数据的中的多数类自然邻居
                if len(NaN_maj) != 0:
                    for j1 in NaN_maj:
                        dist_NaN_maj += np.sqrt(np.sum(np.square(self.X[i] - self.X[j1])))
                if len(NaN_maj) == 0:
                    dist_NaN_maj = 0

                NaN_min = [i for i in NaN_neighbors if self.y[i] == 1]  ##寻找到第i个少数类数据的中的少数类自然邻居
                if len(NaN_min) != 0:
                    for j2 in NaN_min:
                        dist_NaN_min += np.sqrt(np.sum(np.square(self.X[i] - self.X[j2])))
                if len(NaN_min) == 0:
                    dist_NaN_min = 0

                if dist_NaN_min == 0:
                    self.weight.append(0)
                else:
                    weight1 = (dist_NaN_min / (len(NaN_min) + 1e-5)) / (
                            (dist_NaN_maj / (len(NaN_maj) + 1e-5)) + (dist_NaN_min / (len(NaN_min) + 1e-5)))
                    self.weight.append(weight1)

        # print(self.weight)
        self.normalized_weight = self.weight/sum(self.weight)  ###每个少数类数据的权重  数据到网络之中指导数据的生成


        ##权重样本
        weight_tensor = torch.tensor(np.array(self.weight/sum(self.weight)).reshape(-1,1), dtype=torch.float32)
        self.weight_tensor_loader = torch.utils.data.DataLoader(
            weight_tensor,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True
        )
        print(f"训练配置: epochs={self.epochs}, batch_size={self.batch_size}")

        ### 初始化模型
        model = MLPDiffusion(self.num_steps, input_layer=self.minority_data.shape[1])
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Lambda优化模型
        model1 = LambdaOptimizer(input_dim=self.minority_data.shape[1] + 2)
        model1 = model1.to(device)
        optimizer1 = torch.optim.SGD(model1.parameters(), lr=1e-6)

        # 初始 lambda 值 - 根据batch_size初始化
        lambda_0 = torch.tensor(np.random.rand(self.batch_size, 1),
                                requires_grad=False, dtype=torch.float).view(-1, 1).to(device)

        # 固定数据（用于Lambda优化）
        positive_samples = torch.tensor(self.minority_data, dtype=torch.float32).to(device)
        negative_samples = torch.tensor(self.majority_data, dtype=torch.float32).to(device)

        # 交替优化参数
        lambda_update_freq = 1  # 每1个epoch更新一次lambda
        lambda_steps = 2  # 每次lambda优化的步数

        for epoch in range(self.epochs):
            epoch_main_loss = 0
            epoch_lambda_loss = 0
            batch_count = 0

            # 创建数据迭代器
            minority_iter = iter(self.minority_loader)
            overlap_maj_iter = iter(self.overlap_maj_loader)
            weight_iter = iter(self.weight_tensor_loader)
            # print(min(len(self.minority_loader), len(self.overlap_maj_loader)))
            # 按批次训练
            for batch_idx in range(min(len(self.minority_loader), len(self.overlap_maj_loader))):
                try:
                    # 获取当前批次数据
                    minority_batch = next(minority_iter).to(device)
                    input_data2 = next(overlap_maj_iter).to(device)  # 直接从数据加载器获取
                    weighr_batch = next(weight_iter).to(device)
                    current_batch_size = minority_batch.size(0)

                    # ==================== 主模型训练 ====================
                    # 生成噪声和随机时间
                    z = torch.randn_like(minority_batch).to(device)
                    t = torch.rand(current_batch_size, dtype=torch.float).view(-1, 1).to(device)

                    # 构造加噪样本
                    x_t = (1 - t) * z + t * minority_batch

                    # 前向传播
                    v_pred = model(x_t, weighr_batch, t)
                    v_true = (minority_batch - z)
                    v_true2 = (input_data2 - z)

                    # 计算损失
                    loss = (v_true - v_pred).pow(2).mean() - ((torch.sqrt(lambda_0)) * (v_true2 - v_pred)).pow(2).mean()
                    # loss = (v_true - v_pred).pow(2).mean()

                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_main_loss += loss.item()
                    batch_count += 1

                    # ==================== 交替优化lambda ====================
                    if batch_idx % lambda_update_freq == 0:
                        # 固定当前生成样本（不计算梯度）
                        with torch.no_grad():
                            fixed_augmented_samples = self.sample(model, current_batch_size,
                                                                  self.minority_data.shape[1], weighr_batch, 100)[-1]
                            # print('ss',fixed_augmented_samples.shape)

                        # 短期优化lambda
                        lambda_losses = []
                        # print('Lambda', lambda_0.detach().cpu().numpy())
                        for lambda_step in range(lambda_steps):
                            predicted_lambda = model1(fixed_augmented_samples, lambda_0, t)
                            loss1 = vectorized_nearest_positive_loss(fixed_augmented_samples,
                                                                     positive_samples, input_data2,
                                                                     predicted_lambda)

                            optimizer1.zero_grad()
                            loss1.backward()
                            # torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)  # 梯度裁剪
                            optimizer1.step()

                            lambda_losses.append(loss1.item())

                        # 更新lambda_0用于下一轮
                        lambda_0 = predicted_lambda.detach()

                        epoch_lambda_loss += np.mean(lambda_losses)

                except StopIteration:
                    break
            # print(batch_count)
            # 打印epoch统计信息
            if batch_count > 0:
                avg_main_loss = epoch_main_loss / batch_count
                avg_lambda_loss = epoch_lambda_loss / batch_count if epoch_lambda_loss > 0 else 0

                if epoch % 10 == 0:
                    print(f'Epoch {epoch}: 平均主损失={avg_main_loss:.4f}, '
                          f'平均Lambda损失={avg_lambda_loss:.4f}, '
                          f'Lambda均值={lambda_0.mean().item():.4f}')

        # ==================== 最终生成和可视化 ====================
        with torch.no_grad():
            # 生成与少数类样本数量相同的合成数据
            num_to_generate = len(self.majority_data_no_noise) - len(self.minority_data)
            # zero_count = len([x for x in self.normalized_weight if x == 0])
            # print(zero_count)
            # print(self.weight)
            result_list = distribute_by_weight_probability(self.normalized_weight,num_to_generate)  ##每个样本应该生成多少个数据的列表
            # print(result_list)
            # print(sum(result_list))
            weight_generate = []  ##需要生成数据的权重
            for i in range(len(result_list)):
                # if result_list[i] == 0:
                #     weight_generate.append(0)
                if result_list[i] != 0:
                    for j in range(result_list[i]):
                        weight_generate.append(self.normalized_weight[i])
            # print(weight_generate)
            weight_generate = np.array(weight_generate).reshape(-1, 1)
            # print(weight_generate.shape)
            x_seq = self.sample(model, num_to_generate, self.minority_data.shape[1],
                                torch.tensor(weight_generate, dtype=torch.float32).to(device), 100)[
                -1].detach().cpu().numpy()

        # 可视化

        plt.scatter(self.majority_data_no_noise[:, 0], self.majority_data_no_noise[:, 1], alpha=0.5, label='majority class',
                    facecolors='deepskyblue', edgecolors='blue',marker='^')
        plt.scatter(self.minority_data[:, 0], self.minority_data[:, 1], alpha=0.4, c='green', label='minority class',marker='x')
        # plt.scatter(self.overlap[:, 0], self.overlap[:, 1], alpha=0.4, c='black', label='over',
        #             marker='x')
        # rect1 = Rectangle((4.025, 1.86), 0.13, 0.05,  # (x, y), width, height
        #                   angle=48,  # 旋转45度
        #                   alpha=0.15,
        #                   edgecolor='red',
        #                   facecolor='red',
        #                   linewidth=1.5,
        #                   linestyle='--',
        #                   label='overlapping subset')
        #
        # plt.gca().add_patch(rect1)
        # # print(self.minority_data.shape)
        plt.scatter(x_seq[:, 0], x_seq[:, 1], alpha=0.4, label='synthetic data', c='red',
                    marker='*')
        # plt.scatter(np.array(newsample)[:,0],
        #             np.array(newsample)[:,1],alpha=0.4, label='synthetic data', c='black',marker='*')
        # plt.scatter(self.X[overlap_maj[:100]][:,0], self.X[overlap_maj[:100]][:,1], alpha=0.4, label='synthetic data', facecolors='black', edgecolors='black',
        #             marker='o')
        # plt.xticks([-4.3, -3.7])
        # plt.yticks([0.6, 1.4])
        # # plt.scatter(noise[:, 0], noise[:, 1], alpha=0.5, label='noise', facecolors='green', edgecolors='green', )
        # # print(noise.shape)
        # print(x_seq)
        # plt.legend(loc='upper right')
        plt.legend()
        plt.title('NAFLOW')
        plt.savefig(r'D:\pythonproject\pythonProject\NAFLOW\结果/fig4.eps', format='eps',
                    dpi=300, bbox_inches='tight')
        # # plt.plot(noise, x_seq, label='synthetic data1')
        plt.show()

        ###等高线图
        plt.scatter(self.majority_data[:, 0], self.majority_data[:, 1], alpha=0.5, label='majority class',
                    facecolors='deepskyblue', edgecolors='blue',marker='^')
        # plt.scatter(self.majority_data_no_noise[:, 0], self.majority_data_no_noise[:, 1], alpha=0.5, label='majority class',
        #             facecolors='deepskyblue', edgecolors='blue')
        scatter = plt.scatter(self.minority_data[:, 0], self.minority_data[:, 1],
                              c=self.normalized_weight,
                              cmap='Reds',  # 可以使用 'viridis', 'plasma', 'inferno' 等
                              alpha=0.4,  label='minority class',marker='x')

        # 添加颜色条
        cbar = plt.colorbar(scatter, label='Normalized Weight')
        cbar.set_label('Normalized Weight', fontsize=10)
        plt.legend()
        plt.savefig(r'D:\pythonproject\pythonProject\NAFLOW\结果/fig3.pdf', format='pdf',
                    dpi=500, bbox_inches='tight')
        plt.show()



        oversample_X = np.vstack((self.majority_data_no_noise, self.minority_data, x_seq))
        oversample_y = np.vstack((np.array([0] * (self.majority_data_no_noise.shape[0])).reshape(-1, 1),
                                  np.array([1] * (self.majority_data_no_noise.shape[0])).reshape(-1, 1)))
        # print(oversample_X.shape[0])
        # print(oversample_y)

        # 返回合成数据
        return oversample_X, oversample_y


if __name__ == "__main__":
    # 设置batch_size为少数类数据数量的一半
    model = NAFLOW(data_path=r'D:\pythonproject\pythonProject\NAFLOW\Imbalanced data\glass\glass0.dat',
                      epochs=5000, num_steps=100, batch_size_ratio=1)
    train_data , label = model.train()




