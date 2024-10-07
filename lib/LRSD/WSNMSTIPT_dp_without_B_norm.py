###reproduced code from matlab code
###chaoxiao
###202200326

import numpy as np
import cv2
import os

import torch

def prox_non_neg_l1(X, tau):
    X_hat = X - tau
    X_hat[X_hat<0]=0
    return X_hat

def WSNMSTIPT_dp_without_B_norm(tenD, B_hat, Lambda, mu, beta, rho, opts=None):
    # Solve the WSNM problem
    # ---------------------------------------------
    # Input:
    # tenD       -    n1*n2*n3 tensor
    # lambda  -    >0, parameter
    # mu-      regelarization parameter
    # p    -    key parameter of Schattern p norm
    #B_hat  - estimated bacground by the network
    #
    # Output:
    # tenB       -    n1*n2*n3 tensor
    # tenT       -    n1*n2*n3 tensor
    # change     -    change of objective function value
    ######params
    tol = 1e-7
    max_iter = 500
    rho = rho
    max_mu = 1e7
    # beta = 100
    beta = beta
    DEBUG = 1
    normD = torch.linalg.norm(tenD[:])
    ##initialize
    dim = [tenD.shape[0], tenD.shape[1], tenD.shape[2]]
    weightTenT = torch.ones_like(tenD).to(tenD.device)
    N = torch.zeros_like(tenD).to(tenD.device)
    tenT = torch.zeros_like(tenD).to(tenD.device)
    tenB = torch.zeros_like(tenD).to(tenD.device)
    Y1 = torch.zeros_like(tenD).to(tenD.device)
    Y2 = torch.zeros_like(tenD).to(tenD.device)
    ##
    preTnnT = 0
    NOChange_counter = 0
    # weightTenT = np.ones_like(tenD)
    n = min(dim[0], dim[1])
    change = torch.zeros([1, max_iter])

    for iter in range(max_iter):
        #Step 1: Update low-rank background tensor B
        # process_variable = -tenT+tenD-Y1/mu-N
        tenB = 1 / 2 * (tenD - tenT - N + Y1 / mu + (B_hat - Y2 / mu))
        #Step 3:  Update sparse target tensor T
        tenT = prox_non_neg_l1(tenD-tenB-N+ Y1 / mu, weightTenT * Lambda / mu)
        weightTenT = 1 / (abs(tenT) + 0.01)   #%enhance sparsity
        # Step 4: Update N
        N = (mu * (tenD - tenB - tenT) + Y1) / (2 * beta + mu)
        # Step 5: Update Y1, Y2, Y3
        dY1 = tenD-(tenB + tenT + N)
        Y1 = Y1 + mu * dY1
        dY2 = tenB-B_hat
        Y2 = Y2 + mu*dY2
        # Step 6: Update mu
        mu = min(rho * mu, max_mu)
        stopCriterion = torch.linalg.norm(tenD[:] - tenB[:] - tenT[:]-N[:]) / normD
        change[0, iter] = stopCriterion
        if (stopCriterion < tol) or (iter >= 100):
            break
    out = {}
    out['B'] = tenB
    out['T'] = tenT
    out['N'] = N
    return out

def tubalrank(X,tol=None):
    '''
    % The tensor tubal rank of a 3 way tensor
    %
    % X     -    n1*n2*n3 tensor
    % trank -    tensor tubal rank of X
    %
    % version 2.0 - 14/06/2018
    %
    % Written by Canyi Lu (canyilu@gmail.com)
    %
    %
    % References:
    % Canyi Lu, Tensor-Tensor Product Toolbox. Carnegie Mellon University.
    % June, 2018. https://github.com/canyilu/tproduct.
    %
    % Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin and Shuicheng
    % Yan, Tensor Robust Principal Component Analysis with A New Tensor Nuclear
    % Norm, arXiv preprint arXiv:1804.03728, 2018
    '''
    X = np.fft.fft(X, axis=2)
    n1, n2, n3 = X.shape
    s = np.zeros(min(n1, n2))
    #% i=1
    u, s0, v = np.linalg.svd(X[:,:,0].astype(np.float), full_matrices=False)
    s = s+s0
    #% i=2,...,halfn3
    halfn3 = round(n3 / 2+1e-9)
    for i in range(1,halfn3):
        u, si, v = np.linalg.svd(X[:, :, i].astype(np.float), full_matrices=False)
        s += si*2 #特征值是两倍
    #% if n3 is even
    if n3%2==0:
        i = halfn3 + 1
        u, si, v = np.linalg.svd(X[:, :, i].astype(np.float), full_matrices=False)
        s += si
    s = s/n3
    if tol is None:
        tol = max(n1, n2) * np.eps(max(s))
    trank = sum(s > tol)
    return trank

def prox_non_neg_l1(X, tau):
    X_hat=X-tau
    X_hat[X - tau<=0] = 0
    return X_hat

def prox_tnn1(Y,rho,p):
    '''
    The proximal operator of the tensor nuclear norm of a 3 way tensor
    min_X rho*||X||_*+0.5*||X-Y||_F^2
    Y     -    n1*n2*n3 tensor
    X     -    n1*n2*n3 tensor
    tnn   -    tensor nuclear norm of X
    trank -    tensor tubal rank of X
    4/16/2022
    Rewritten by ChaoXiao matlab version is written by Canyi Lu
    References:
    Canyi Lu, Tensor-Tensor Product Toolbox. Carnegie Mellon University.
    June, 2018. https://github.com/canyilu/tproduct.
    Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin and Shuicheng
    Yan, Tensor Robust Principal Component Analysis with A New Tensor Nuclear
    Norm, arXiv preprint arXiv:1804.03728, 2018
    '''
    n1, n2, n3 = Y.shape
    X = np.zeros_like(Y)
    Y = np.fft.fft(Y,axis=2)
    tnn = 0
    trank = 0
    #% first frontal slice
    U,S,V = np.linalg.svd(Y[:,:,0].astype(np.float), full_matrices=False)
    tempDiagS,svp=IterativeWSNM(S, rho, p)
    X[:, :, 0] = np.dot(np.dot(U[:, :svp], np.diag(tempDiagS[:svp])), V[:svp, :])
    tnn = tnn + sum(S)
    trank = max(trank, svp)
    #% i=2,...,halfn3
    halfn3 = round(n3/2+1e-9)
    for i in range(1, halfn3):
        U, S, V = np.linalg.svd(Y[:,:,i].astype(np.float), full_matrices=False)
        tempDiagS, svp = IterativeWSNM(S, rho, p)
        X[:,:, i] = np.dot(np.dot(U[:, :svp],np.diag(tempDiagS[:svp])),V[:svp, :])
        tnn = tnn + sum(S) * 2
        trank = max(trank,svp)
        X[:,:, n3-i] = np.conj(X[:,:, i])
    #% if n3 is even
    if n3%2==0:
        i = halfn3
        U, S, V = np.linalg.svd(Y[:,:,i].astype(np.float), full_matrices=False)
        tempDiagS, svp = IterativeWSNM(S, rho, p)
        X[:,:, i] = np.dot(np.dot(U[:, :svp],np.diag(tempDiagS[:svp])),V[:svp, :])
        tnn = tnn + sum(S)
        trank = max(trank,svp)
    tnn = tnn / n3
    X = np.fft.ifft(X, axis=2)#这里得到的是虚数，需要将其转换为实数
    X = abs(X)
    return X,tnn,trank

def IterativeWSNM( SigmaY, C, p ):
    '''
    % weighted schatten p-norm minimization
    % Objective function:
    % min |A|^{p}_w,p + |E|_1, s.t. A + E = D
    % w_i = C*sqrt(m*n)/(SigmaX_i + eps);

    % input: SigmaY is the singular value of Y (maybe partial)
    % C = C*sqrt(m*n)/mu;

    % here, we use GISA to iteratively solve the following lp-based
    % minimization:
    % X_{k+1} = argmin_X |X|^{p}_w,p + \mu/2 ||X - (Y - E_{k+1} + \mu^{-1} L_k)||_F
    '''
    p1 = p
    Temp = SigmaY.copy()
    s = SigmaY.copy()
    s1 = np.zeros_like(s)
    eps = 1e-7
    for i in range(3):
        W_Vec = C / ((Temp) + eps)
        [s1, svp] = solve_Lp_w(s, W_Vec, p1)
        Temp = s1
    SigmaX = s1
    return SigmaX, svp

def solve_Lp_w( y, Lambda, p ):
    '''
    % Modified by Dr. xie yuan
    % lambda here presents the weights vector
    '''
    J = 2
    #% tau is generalized thresholding vector
    tau   =  (2*Lambda*(1-p))**(1/(2-p)) + p*Lambda*(2*(1-p)*Lambda)**((p-1)/(2-p))
    x     =   np.zeros_like(y)
    #% i0表示 thresholding 后非0的个数
    i0    =   np.argwhere( abs(y)>tau)
    svp   =   len(i0)
    if len(i0) >= 1:
        #    % lambda  =   lambda(i0);
        y0    =   y[i0]
        t     =   abs(y0)
        lambda0 = Lambda[i0]
        for j  in range(J):
            t = abs(y0) - p * lambda0* (t)**(p - 1)
        x[i0] = np.sign(y0)* t
    return x, svp

