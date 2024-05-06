import numpy as np
import pandas as pd
from utils import read_csv
import math

def get_fusion_sim (k1, k2):

    MD = pd.read_csv("../Data/MD_A.csv", index_col=0)


    DS = pd.read_csv("../Data/DS.csv", index_col=0)


    m = MD.shape[0]
    T = []
    for i in range(m):
        T.append(np.where(MD.iloc[i] == 1))
    Fs = []
    for ti in range(m):
        for tj in range(m):
            Ti_Tj, Tj_Ti = S_fun1(DS, T[ti][0], T[tj][0])
            FS_i_j = FS_fun1(Ti_Tj, Tj_Ti, T[ti][0], T[tj][0])
            Fs.append(FS_i_j)
    Fs = np.array(Fs).reshape(MD.shape[0], MD.shape[0])
    Fs = pd.DataFrame(Fs)
    for index, rows in Fs.iterrows():
        for col, rows in Fs.iteritems():
            if index == col:
                Fs.loc[index, col] = 1
    Fs

    rm, rt = r_func(MD)
    sim_m1, sim_d1 = Gau_sim(MD, rm, rt)


    MD_c = MD.copy()
    MD_c.columns = range(0, MD.shape[1])
    MD_c.index = range(0, MD.shape[0])
    MD_c = np.array(MD_c)


    sim_m2, sim_d2 = cos_sim(MD_c)
    sim_m3, sim_d3 = sig_kr(MD_c)


    m1 = new_normalization1(sim_m1)
    m2 = new_normalization1(sim_m2)
    m3 = new_normalization1(sim_m3)

    Sm_1 = KNN_kernel1(sim_m1, k1)
    Sm_2 = KNN_kernel1(sim_m2, k1)
    Sm_3 = KNN_kernel1(sim_m3, k1)

    Pm = Updating1(Sm_1, Sm_2, Sm_3, m1, m2, m3)
    Pm_final = (Pm + Pm.T)/2
    Pm_final = InSm(Pm_final, Fs, 0.28)



    d1 = new_normalization1(sim_d1)
    d2 = new_normalization1(sim_d2)
    d3 = new_normalization1(sim_d3)


    Sd_1 = KNN_kernel1(sim_d1, k2)
    Sd_2 = KNN_kernel1(sim_d2, k2)
    Sd_3 = KNN_kernel1(sim_d3, k2)

    Pd = Updating1(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T)/2
    Pd_final = InSm(Pd_final, DS, 0.28)


    # Pm_final = (Fs+sim_m1+sim_m2+sim_m3)/4
    # Pd_final = (DS+sim_d1+sim_d2+sim_d3)/4

    return Pm_final, Pd_final


def new_normalization1(w):
    m = w.shape[0]
    p = np.zeros([m, m])
    for i in range(m):
        for j in range(m):
            if i == j:
                p[i][j] = 1/2
            elif np.sum(w[i, :])-w[i, i]>0:
                p[i][j] = w[i, j]/(2*(np.sum(w[i, :])-w[i, i]))
    return p


def KNN_kernel1(S, k):
    n = S.shape[0]
    S_knn = np.zeros([n, n])
    for i in range(n):
        sort_index = np.argsort(S[i, :])
        for j in sort_index[n-k:n]:
            if np.sum(S[i, sort_index[n-k:n]])>0:
                S_knn[i][j] = S[i][j] / (np.sum(S[i, sort_index[n-k:n]]))
    return S_knn



def Updating1 (S1,S2,S3, P1,P2,P3):
    it = 0
    P = (P1+P2+P3)/3
    dif = 1
    while dif > 0.0000001:
        it = it + 1
        P111 =np.dot(np.dot(S1, (P2+P3)/2), S1.T)
        P111 = new_normalization1(P111)
        P222 =np.dot(np.dot(S2, (P1+P3)/2), S2.T)
        P222 = new_normalization1(P222)
        P333 = np.dot(np.dot(S3, (P1+P2)/2), S3.T)
        P333 = new_normalization1(P333)

        P1 = P111
        P2 = P222
        P3 = P333

        P_New = (P1+P2+P3)/3
        dif = np.linalg.norm(P_New-P)/np.linalg.norm(P)
        P = P_New
    print("Iter numb1", it)
    return P

def Updating2 (S1,S2, P1,P2):
    it = 0
    P = (P1+P2)/2
    dif = 1
    while dif > 0.0000001:
        it = it + 1
        P111 =np.dot(np.dot(S1, P2), S1.T)
        P111 = new_normalization1(P111)
        P222 =np.dot(np.dot(S2, P1), S2.T)
        P222 = new_normalization1(P222)
        # P333 = np.dot(np.dot(S3, (P1+P2)/2), S3.T)
        # P333 = new_normalization1(P333)

        P1 = P111
        P2 = P222
        # P3 = P333

        P_New = (P1+P2)/2
        dif = np.linalg.norm(P_New-P)/np.linalg.norm(P)
        P = P_New
    print("Iter numb1", it)
    return P

def InSm(sm1, sm2, w):
    simm = w * sm1 + (1-w) * sm2
    return simm


# In[2]:


# functional similarity
def S_fun1(DDsim, T0, T1):
    DDsim = np.array(DDsim)
    T0_T1 = []
    if len(T0) != 0 and len(T1) != 0:
        for ti in T0:
            m_ax = []
            for tj in T1:
                m_ax.append(DDsim[ti][tj])
            T0_T1.append(max(m_ax))
    if len(T0) == 0 or len(T1) == 0:
        T0_T1.append(0)
    T1_T0 = []
    if len(T0) != 0 and len(T1) != 0:
        for tj in T1:
            m_ax = []
            for ti in T0:
                m_ax.append(DDsim[tj][ti])
            T1_T0.append(max(m_ax))
    if len(T0) == 0 or len(T1) == 0:
        T1_T0.append(0)
    return T0_T1, T1_T0

# 计算Fs
def FS_fun1(T0_T1, T1_T0, T0, T1):
    a = len(T1)
    b = len(T0)
    S1 = sum(T0_T1)
    S2 = sum(T1_T0)
    FS = []
    if a != 0 and b != 0:
        Fsim = (S1+S2)/(a+b)
        FS.append(Fsim)
    if a == 0 or b == 0:
        FS.append(0)
    return FS


# In[3]:


# Gaussian interaction profile kernel similarity
def r_func(MD):
    m = MD.shape[0]
    n = MD.shape[1]
    EUC_MD = np.linalg.norm(MD, ord=2, axis=1, keepdims=False)
    EUC_DL = np.linalg.norm(MD.T, ord=2, axis=1, keepdims=False)
    EUC_MD = EUC_MD**2
    EUC_DL = EUC_DL**2
    sum_EUC_MD = np.sum(EUC_MD)
    sum_EUC_DL = np.sum(EUC_DL)
    rl = 1 / ((1 / m) * sum_EUC_MD)
    rt = 1 / ((1 / n) * sum_EUC_DL)
    return rl, rt


def Gau_sim(MD, rl, rt):
    MD = np.mat(MD)
    DL = MD.T
    m = MD.shape[0]
    n = MD.shape[1]
    c = []
    d = []
    for i in range(m):
        for j in range(m):
            b_1 = MD[i] - MD[j]
            b_norm1 = np.linalg.norm(b_1, ord=None, axis=1, keepdims=False)
            b1 = b_norm1**2
            b1 = math.exp(-rl * b1)
            c.append(b1)
    for i in range(n):
        for j in range(n):
            b_2 = DL[i] - DL[j]
            b_norm2 = np.linalg.norm(b_2, ord=None, axis=1, keepdims=False)
            b2 = b_norm2**2
            b2 = math.exp(-rt * b2)
            d.append(b2)
    GMM = np.mat(c).reshape(m, m)
    GDD = np.mat(d).reshape(n, n)
    return GMM, GDD


# In[4]:


#cosine similarity
def cos_sim(MD):
    m = MD.shape[0]
    n = MD.shape[1]
    cos_MS1 = []
    cos_DS1 = []
    for i in range(m):
        for j in range(m):
            a = MD[i,:]
            b = MD[j,:]
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            if a_norm!=0 and b_norm!=0:
                cos_ms = np.dot(a,b)/(a_norm * b_norm)
                cos_MS1.append(cos_ms)
            else:
                cos_MS1.append(0)

    for i in range(n):
        for j in range(n):
            a1 = MD[:,i]
            b1 = MD[:,j]
            a1_norm = np.linalg.norm(a1)
            b1_norm = np.linalg.norm(b1)
            if a1_norm!=0 and b1_norm!=0:
                cos_ds = np.dot(a1,b1)/(a1_norm * b1_norm)
                cos_DS1.append(cos_ds)
            else:
                cos_DS1.append(0)

    cos_MS1 = np.array(cos_MS1).reshape(m, m)
    cos_DS1 = np.array(cos_DS1).reshape(n, n)
    return cos_MS1,cos_DS1


# In[5]:


#sigmoid function kernel similarity
def sig_kr(MD):
    m = MD.shape[0]
    n = MD.shape[1]
    sig_MS1 = []
    sig_DS1 = []
    for i in range(m):
        for j in range(m):
            a = MD[i,:]
            b = MD[j,:]
            z = (1/m)*(np.dot(a,b))
            sig_ms = math.tanh(z)
            sig_MS1.append(sig_ms)

    for i in range(n):
        for j in range(n):
            a1 = MD[:,i]
            b1 = MD[:,j]
            z1 = (1/n)*(np.dot(a1,b1))
            sig_ds = math.tanh(z1)
            sig_DS1.append(sig_ds)

    sig_MS1 = np.array(sig_MS1).reshape(m, m)
    sig_DS1 = np.array(sig_DS1).reshape(n, n)
    return sig_MS1, sig_DS1






























def GIP_kernel(association):

    nc = association.shape[0]
    matrix = np.zeros((nc, nc))
    r = getGosiR(association)
    for i in range(nc):
        for j in range(nc):
            temp_up = np.square(np.linalg.norm(association[i, :] - association[j, :]))
            if r == 0:
                matrix[i][j] = 0
            elif i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e ** (-temp_up / r)
    return matrix


def getGosiR(association):

    nc = association.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(association[i, :])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r

def sim_thresholding(matrix: np.ndarray, threshold):
    matrix_copy = matrix.copy()
    matrix_copy[matrix_copy >= threshold] = 1
    matrix_copy[matrix_copy < threshold] = 0
    print(f"rest links: {np.sum(np.sum(matrix_copy))}")
    return matrix_copy








