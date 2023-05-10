import numpy as np
from numpy import linalg as la
import time
from split_train_test import split_data2, mk_train_matrix2
import matplotlib.pyplot as plt
import split_train_test
import heapq
from multiprocessing import Pool


def ls_probs(m, n, A):
    tic = time.time()
    row_norms = np.zeros(m)
    sum = 0
    for i in range(m):
        row_norms[i] = la.norm(A[i, :]) ** 2
        sum += row_norms[i]
    A_Frobenius = np.sqrt(sum)

    LS_prob_rows = np.zeros(m)
    for i in range(m):
        LS_prob_rows[i] = row_norms[i] / A_Frobenius ** 2

    LS_prob_columns = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            LS_prob_columns[i][j] = A[i, j] ** 2 / row_norms[i]
    toc = time.time()

    rt_LS_probs = toc - tic
    print("2范数采样时间 : " + str(rt_LS_probs))
    return row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius


def sample_C(A, m, n, r, c, row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius):
    # 采样i和j
    tic = time.time()
    rows = np.random.choice(m, r, replace=True, p=LS_prob_rows)
    columns = np.zeros(c, dtype=int)
    for j in range(c):
        i = np.random.choice(rows, replace=True)
        columns[j] = np.random.choice(n, 1, p=LS_prob_columns[i])

    # 生成行标准化过的R
    R = np.zeros((r, n))
    for i in range(r):
        R[i, :] = A[rows[i], :] * A_Frobenius / (np.sqrt(r) * np.sqrt(row_norms[rows[i]]))

    # 生成双标准化过的C
    C = np.zeros((r, c))
    column_norms = np.zeros(c)
    for i in range(r):
        for j in range(c):
            C[i, j] = A[rows[i], columns[j]] * A_Frobenius / (np.sqrt(r) * np.sqrt(row_norms[rows[i]]))
            column_norms[j] += C[i, j] ** 2

    for j in range(c):
        C[:, j] = C[:, j] * A_Frobenius / (np.sqrt(c) * np.sqrt(column_norms[j]))

    w, sigma, vh = la.svd(C, full_matrices=False)
    toc = time.time()
    rt_sampling_C = toc - tic
    print("生成R、C并完成C的svd的时间 ：" + str(rt_sampling_C))
    return w, sigma, vh, rows, R, LS_prob_columns


def quantum_inspired(A, p, rank):
    t1 = time.time()
    m, n = np.shape(A)
    LS = ls_probs(m, n, A)
    row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius = LS[0:4]
    svd_C = sample_C(A, m, n, p, p, row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius)
    w, sigma, vh, rows, R = svd_C[0:5]
    sigma2 = np.diag(sigma[0:rank])
    U = np.zeros((rank, p))
    for i in range(0, rank):
        U[i] = w.T[i]
    V = np.dot(np.dot(R.T, U.T), np.linalg.inv(sigma2))
    Dk = np.dot(np.dot(A, V), V.T)
    # Dk = np.zeros([m,n])
    # for i in range(0, rank):
    #     U = np.zeros((1, p))
    #     U[0] = w.T[i]
    #     V = np.true_divide(np.dot(R.T, U.T), sigma[i])
    #     dk = np.dot(np.dot(A, V), V.T)
    #     Dk = Dk + dk
    t2 = time.time()
    print("总时间 : " + str(t2 - t1))
    return Dk


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


def FM_recall():
    movies_train = np.load('A_movies_small.npy')
    movies_test = np.load('A_movies_small_test.npy')
    m, n = np.shape(movies_train)
    p = 1000
    rank = 10
    result = quantum_inspired(movies_train, p, rank)
    recall_array = []
    for i in range(0, m):
        k = 100
        a = np.array(result[i])
        a_index = a.argsort()[::-1][0:n - 1]
        c_index = []
        for j in a_index:
            if movies_test[i][j] != 0:
                c_index.append(j)
            if len(c_index) == k:
                break
        recall_array.append(c_index)
    print(recall_array)
    return recall_array


def test(X):
    movies_test = np.load('A_movies_small_test.npy')
    m, n = np.shape(X)
    count = 0
    print(m,n)
    for i in range(0, m):
        k = 10
        a = np.array(X[i])
        a_index = a.argsort()[::-1][0:n-1]
        c_index = []
        for j in a_index:
            if movies_test[i][j] != 0:
                c_index.append(j)
            if len(c_index) == k:
                break
        # print(a_index)
        # print(c_index)
        b = np.array(movies_test[i])
        b_index = b.argsort()[::-1][0:k]
        index = []
        for j in range(0, len(b_index)):
            if b[b_index[j]] != 0:
                index.append(j)
        b_index = b_index[index]
        s = find_lcseque(c_index, b_index)
        # s = find_lcseque(np.sort(c_index), np.sort(b_index))
        if len(b_index) != 0:
            lcs = len(s)/len(b_index)
            count += lcs
            print(lcs)
        else:
            m -= 1
    print(count/m)
    return count/m
    # mu = 3.581564453029317
    # m,n = np.shape(movies_test)
    # count = 0
    # len = 0
    # for i in range(5):
    #     for j in range(0, n):
    #         if movies_test[i][j] != 0:
    #             print(movies_test[i][j], X[i][j])
    #             count = count + abs(movies_test[i][j] - X[i][j])
    #             len += 1
    # print(count/len)


if __name__ == '__main__':
    FM_recall()


# if __name__ == '__main__':
#     # df, df_train, df_test, mu = split_train_test.split_data()
#     # A = split_train_test.mk_train_matrix(df, df_train, mu)
#     # np.save("A_movies_small.npy", A)
#     result = []
#     x = []
#     for i in range(2, 11):
#         split_ratio = i/10
#         df, df_train, df_test, mu = split_data2(split_ratio)
#         A, B = mk_train_matrix2(df, df_train, mu)
#         np.save("A_movies_small.npy", A)
#         np.save("A_movies_small_test.npy", B)
#         A = np.load('A_movies_small.npy')
#         m,n = np.shape(A)
#         p = 1000
#         rank = 10
#         X = quantum_inspired(A, p, rank)
#     # for i in X[1]:
#     #     print(i)
#         result.append(test(X))
#         x.append(split_ratio)
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
#     # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
#     plt.plot(x, result, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='QRPI')
#     # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
#     plt.legend(loc="upper right")
#     plt.xlabel('数据集规模')
#     plt.ylabel('LCS分数')
#     plt.show()
    # split_train_test.res_evaluation(X, df_test)
# LS_file = open('sample_C_File1.txt', "w+")
# LS_file.write(str(sampc))
# LS_file.close()
