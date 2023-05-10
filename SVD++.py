import numpy as np
from numpy import linalg as la
import time
from split_train_test import split_data2, mk_train_matrix2
import matplotlib.pyplot as plt
import random
from QRPI import quantum_inspired,test
'''
author:huang
svd++ algorithm
'''


def find_lcseque(s1, s2):
    # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    # d用来记录转移方向
    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = 'ok'
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = 'left'
            else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    # print(np.array(d))
    s = []
    while m[p1][p2]:  # 不为None时
        c = d[p1][p2]
        if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1 - 1])
            p1 -= 1
            p2 -= 1
        if c == 'left':  # 根据标记，向左找下一个
            p2 -= 1
        if c == 'up':  # 根据标记，向上找下一个
            p1 -= 1
    s.reverse()
    return s


class SVDPP:
    def __init__(self, mat, K=20):
        self.mat = np.array(mat)
        self.K = K
        self.bi = {}
        self.bu = {}
        self.qi = {}
        self.pu = {}
        self.avg = np.mean(self.mat[:, 2])
        self.y = {}
        self.u_dict = {}
        for i in range(self.mat.shape[0]):
            uid = self.mat[i, 0]
            iid = self.mat[i, 1]
            self.u_dict.setdefault(uid, [])
            self.u_dict[uid].append(iid)
            self.bi.setdefault(iid, 0)
            self.bu.setdefault(uid, 0)
            self.qi.setdefault(iid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))
            self.pu.setdefault(uid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))
            self.y.setdefault(iid, np.zeros((self.K, 1)) + .1)

    def predict(self, uid, iid):  # 预测评分的函数
        # setdefault的作用是当该用户或者物品未出现过时，新建它的bi,bu,qi,pu及用户评价过的物品u_dict，并设置初始值为0
        self.bi.setdefault(iid, 0)
        self.bu.setdefault(uid, 0)
        self.qi.setdefault(iid, np.zeros((self.K, 1)))
        self.pu.setdefault(uid, np.zeros((self.K, 1)))
        self.y.setdefault(uid, np.zeros((self.K, 1)))
        self.u_dict.setdefault(uid, [])
        u_impl_prf, sqrt_Nu = self.getY(uid, iid)
        rating = self.avg + self.bi[iid] + self.bu[uid] + np.sum(self.qi[iid] * (self.pu[uid] + u_impl_prf))  # 预测评分公式
        # 由于评分范围在1到5，所以当分数大于5或小于1时，返回5,1.
        if rating > 5:
            rating = 5
        if rating < 1:
            rating = 1
        return rating

    # 计算sqrt_Nu和∑yj
    def getY(self, uid, iid):
        Nu = self.u_dict[uid]
        I_Nu = len(Nu)
        sqrt_Nu = np.sqrt(I_Nu)
        y_u = np.zeros((self.K, 1))
        if I_Nu == 0:
            u_impl_prf = y_u
        else:
            for i in Nu:
                y_u += self.y[i]
            u_impl_prf = y_u / sqrt_Nu

        return u_impl_prf, sqrt_Nu

    def train(self, steps=3, gamma=0.04, Lambda=0.15):  # 训练函数，step为迭代次数。
        print('train data size', self.mat.shape)
        for step in range(steps):
            print('step', step + 1, 'is running')
            KK = np.random.permutation(self.mat.shape[0])  # 随机梯度下降算法，kk为对矩阵进行随机洗牌
            rmse = 0.0
            for i in range(self.mat.shape[0]):
                j = KK[i]
                uid = self.mat[j, 0]
                iid = self.mat[j, 1]
                rating = self.mat[j, 2]
                predict = self.predict(uid, iid)
                u_impl_prf, sqrt_Nu = self.getY(uid, iid)
                eui = rating - predict
                rmse += eui ** 2
                self.bu[uid] += gamma * (eui - Lambda * self.bu[uid])
                self.bi[iid] += gamma * (eui - Lambda * self.bi[iid])
                self.pu[uid] += gamma * (eui * self.qi[iid] - Lambda * self.pu[uid])
                self.qi[iid] += gamma * (eui * (self.pu[uid] + u_impl_prf) - Lambda * self.qi[iid])
                for j in self.u_dict[uid]:
                    self.y[j] += gamma * (eui * self.qi[j] / sqrt_Nu - Lambda * self.y[j])

            gamma = 0.93 * gamma
            print('rmse is', np.sqrt(rmse / self.mat.shape[0]))

    def test(self, test_data):  # gamma以0.93的学习率递减

        test_data = np.array(test_data)
        print('test data size', test_data.shape)
        rmse = 0.0
        for i in range(test_data.shape[0]):
            uid = test_data[i, 0]
            iid = test_data[i, 1]
            rating = test_data[i, 2]
            eui = rating - self.predict(uid, iid)
            rmse += eui ** 2
        print('rmse of test data is', np.sqrt(rmse / test_data.shape[0]))

    def test2(self, test_data):
        test_data = np.array(test_data)
        test_data2 = np.zeros([944, 1683])
        result_data = np.zeros([944, 1683])
        for i in range(944):
            for j in range(1683):
                test_data2[i, j] = 0
                result_data[i, j] = 0
        for i in range(test_data.shape[0]):
            test_data2[test_data[i, 0], test_data[i, 1]] = test_data[i, 2]
            result_data[test_data[i, 0], test_data[i, 1]] = self.predict(test_data[i, 0], test_data[i, 1])
        LCS = 0
        count = 0
        for i in range(1,944):
            k = 10
            a = np.array(result_data[i])
            a_index = a.argsort()[::-1][0:k]
            print(a_index)
            b = np.array(test_data2[i])
            b_index = b.argsort()[::-1][0:k]
            index = []
            for j in range(0, len(b_index)):
                if b[b_index[j]] != 0:
                    index.append(j)
            b_index = b_index[index]
            print(b_index)
            s = find_lcseque(a_index, b_index)
            if len(b_index) != 0:
                LCS += len(s) / len(b_index)
                count += 1
                print(len(s) / len(b_index))
        print(LCS/count)
        return LCS/count


def getMLData():  # 获取训练集和测试集的函数
    import re
    f = open("./data/ml-100k/u1.base", 'r')
    lines = f.readlines()
    f.close()
    data = []
    for line in lines:
        list = re.split('\t|\n', line)
        if int(list[2]) != 0:
            data.append([int(i) for i in list[:3]])
    train_data = data
    f = open("./data/ml-100k/u1.test", 'r')
    lines = f.readlines()
    f.close()
    data = []
    for line in lines:
        list = re.split('\t|\n', line)
        if int(list[2]) != 0:
            data.append([int(i) for i in list[:3]])
    test_data = data

    return train_data, test_data


def getMLData2(split_ratio):  # 获取训练集和测试集的函数
    import re
    f = open("./data/ml-100k/u.data", 'r')
    lines = f.readlines()
    f.close()
    data = []
    data2 = []
    count = 0
    for line in lines:
        list = re.split('\t|\n', line)
        if int(list[2]) != 0:
            c = random.randint(1, 2)
            if c == 1:
                data.append([int(i) for i in list[:3]])
            else:
                data2.append([int(i) for i in list[:3]])
        count += 1
        if count == int(len(lines)*split_ratio):
            break
    train_data = data
    test_data = data2
    return train_data, test_data


def svd_test():
    result = []
    for i in range(2, 11):
        split_ratio = i / 10
        train_data, test_data = getMLData2(split_ratio)
        a = SVDPP(train_data, 30)
        a.train()
        result.append(a.test2(test_data))
    return result


if __name__ == '__main__':
    # train_data, test_data = getMLData2(0.2)
    # a = SVDPP(train_data, 30)
    # a.train()
    # a.test2(test_data)
    result1 = svd_test()
    result = []
    x = []
    for i in range(2, 11):
        split_ratio = i / 10
        df, df_train, df_test, mu = split_data2(split_ratio)
        A, B = mk_train_matrix2(df, df_train, mu)
        np.save("A_movies_small.npy", A)
        np.save("A_movies_small_test.npy", B)
        A = np.load('A_movies_small.npy')
        m, n = np.shape(A)
        p = 1000
        rank = 10
        X = quantum_inspired(A, p, rank)
        # for i in X[1]:
        #     print(i)
        result.append(test(X))
        x.append(split_ratio)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(x, result, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='QRPI')
    plt.plot(x, result1, 'ro-', color='red', alpha=0.8, linewidth=1, label='SVD++')
    plt.legend(loc="upper right")
    plt.xlabel('数据集规模')
    plt.ylabel('LCS分数')
    plt.show()
