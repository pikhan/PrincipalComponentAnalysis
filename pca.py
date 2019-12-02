import numpy as np


def compute_Z(X, centering=True, scaling=False):
    sample_num = len(X)
    feature_num = len(X[0])
    feature_avg = np.array([feature_num])
    feature_avg = np.zeros(feature_avg)  # initialize averages @ 0
    feature_var = np.array([feature_num])
    feature_var = np.zeros(feature_var)  # initialize variance @ 0
    feature_stdev = np.array([feature_num])
    Z = X

    sigma = 0
    for f in range(feature_num):
        for s in range(sample_num):
            sigma += X[s][f]
        sigma /= sample_num
        feature_avg[f] = sigma
        sigma = 0

    for f in range(feature_num):
        for s in range(sample_num):
            sigma += (X[s][f] - feature_avg[f]) * (X[s][f] - feature_avg[f])
        sigma /= sample_num - 1     # sample var -> n - 1
        feature_var[f] = sigma
        sigma = 0
    feature_stdev = np.sqrt(feature_var)

    for i in range(sample_num):
        for j in range(feature_num):
            if centering is True:
                Z[i][j] -= feature_avg[j]
            if scaling is True:
                Z[i][j] /= feature_stdev[j]

    return Z


def compute_covariance_matrix(Z):
    ZT = np.transpose(Z)
    return np.matmul(ZT, Z)


def find_pcs(COV):
    return np.linalg.eig(COV)


def project_data(Z, PCS, L, k, var):
    ZT = np.transpose(Z)
    adj = PCS[0]
    return np.matmul(adj, ZT)
