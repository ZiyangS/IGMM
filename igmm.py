import sys
import time
import copy
import numpy as np
from scipy.stats import gamma, wishart
from scipy.stats import multivariate_normal as mv_norm
from numpy.linalg import inv, det, slogdet
from scipy import special
from ars import ARS


# the maximum positive integer for use in setting the ARS seed
maxsize = sys.maxsize


class Sample:
    """Class for defining a single sample"""
    def __init__(self, mu, s, pi, lam, r, beta, w, alpha, k):
        self.mu = np.reshape(mu, (1, -1))
        self.s = np.reshape(s, (1, -1))
        self.pi = np.reshape(pi, (1, -1))
        self.lam = lam
        self.r = r
        self.beta = beta
        self.w = w
        self.k = k
        self.alpha = alpha

class Samples:
    """Class for generating a collection of samples"""
    def __init__(self, N, D):
        self.sample = []
        self.N = N
        self.D = D

    def __getitem__(self, key): 
        return self.sample[key]

    def addsample(self, S):
        return self.sample.append(S)

def igmm_full_cov_sampler(Y, cov_type="full", Nsamples=2000, Nint=100, anneal=False):
    """
    infinite gaussian mixture model with full or diagonal covariance matrix
    using Gibbs sampling
    input:
        Y : the input dataset
        Nsamples : the number of Gibbs samples
        Nint : the samples used for evaluating the tricky integral
        anneal : perform simple siumulated annealing
    output:
        Samp : the output samples
        Y : the input dataset
    """
    # compute some data derived quantities
    N, D = Y.shape
    muy = np.mean(Y, axis=0)
    covy = np.cov(Y.transpose())
    inv_covy = inv(covy) if D > 1 else np.reshape(1.0/covy,(1,1))

    # initialise a single sample
    Samp = Samples(Nsamples, D)

    c = np.zeros(N)            # initialise the stochastic indicators
    pi = np.zeros(1)           # initialise the weights
    mu = np.zeros((1, D))      # initialise the means
    s = np.zeros((1, D * D))    # initialise the precisions
    n = np.zeros(1)            # initialise the occupation numbers

    # set first mu to the mean of all data
    mu[0,:] = muy
    # set first pi to 1, because only one component initially
    pi[0] = 1.0

    # draw beta from prior
    # y = (beta - D + 1)^(-1) , y is subject to gamma dist (Rasmussen 2000), eq 11 (Rasmussen 2006)
    # gamma dist (Rasmussen 2000) use scale and mean, is different. alpha need to change to 1/2*alpha
    # theta parameter change to mean, theta*(1/alpha). So theta of Rasmussen's should be 2*theta/a
    # For multivariate part, mean of Gamma is 1/D but 1. So we need "/float(D)"
    temp_para_y = draw_gamma(0.5, 2.0/float(D))
    beta = np.squeeze(float(D) - 1.0 + 1.0/temp_para_y)

    # draw w from prior
    # w is subject to Wishart(D, covariance matrix/D), eq 11 (Rasmussen 2006)
    # the df(degrees of freedom) is dimensions number D, scale matrix is covariance matrix/D
    w = draw_wishart(D, covy/float(D))

    # draw s from prior
    # Sj is subject to Wishart(beta, (beta*w)-1), eq 8 (Rasmussen 2006)
    s[0,:] = np.squeeze(np.reshape(draw_wishart(float(beta), inv(beta*w)), (D*D, -1)))
    n[0] = N                   # initially, all samples are in the only component

    # draw lambda from prior
    # lambda is subject to Multivariate Guassian(muy, covy), eq 13 (Rasmussen 2006)
    lam = draw_MVNormal(mean=muy, cov=covy)

    # draw r from prior
    # r is subject to Wishart(D, (D*covy)^(-1)), eq 13 (Rasmussen 2006)
    r = draw_wishart(D, inv(D * covy))

    # draw alpha from prior
    # (alpha)^(-1) is subject to Rasmussen's paper's gamma distribution, scale is 1, mean is 1, eq 14 (Rasmussen 2006)
    # so (alphs)^(-1) is sujbect to gamma distribution, scale is 1/2, and theta is 2
    alpha = 1.0/draw_gamma(0.5, 2.0)
    k = 1                                       # set only 1 component
    S = Sample(mu, s, pi, lam, r, beta, w, alpha, k)    # define the sample

    Samp.addsample(S)                           # add the sample
    print('{}: initialised parameters'.format(time.asctime()))

    # loop over samples
    z = 1
    oldpcnt = 0
    while z < Nsamples:
        # define simulated annealing temperature
        G = max(1.0, float(0.5*Nsamples)/float(z + 1)) if anneal else 1.0

        # recompute muy and covy
        muy = np.mean(Y, axis=0)
        covy = np.cov(Y, rowvar=0)
        inv_covy = inv(covy) if D > 1 else np.reshape(1.0/covy,(1, 1))

        # calculate yj bar, for each represented muj value, eq 4 (Rasmussen 2000)
        yj_bar = [np.sum(Y[np.argwhere(c == j), : ], 0)/nj for j, nj in enumerate(n)]
        mu = np.zeros((k, D))
        j = 0
        # draw muj from posterior (depends on sj, c, lambda, r), eq 4 (Rasmussen 2000)
        for yb, nj, sj in zip(yj_bar, n, s):
            sj = np.reshape(sj, (D, D))
            # To get posterior of muj, fist calculate cov and mean, and then sampling
            muj_cov = inv(nj*sj + r)
            muj_mean = np.dot(muj_cov, nj*np.dot(sj, np.squeeze(yb)) + np.dot(r, lam))
            mu[j,:] = draw_MVNormal(mean=muj_mean, cov=muj_cov, size=1)
            j += 1

        # draw lambda from posterior (depends on mu, k, and r), eq 5 (Rasmussen 2000)
        lam_cov = inv(inv_covy + k*r)
        lam_mean = np.dot(lam_cov, np.dot(inv_covy, muy) + np.dot(r, np.sum(mu, axis=0)))
        lam = draw_MVNormal(mean=lam_mean, cov=lam_cov, size=1)

        # draw r from posterior (depnds on k, mu, and lambda), eq 5 (Rasmussen 2000)
        temp_para_sum = np.zeros((D, D))
        for muj in mu:
            temp_para_sum += np.outer((muj - lam), np.transpose(muj - lam))
        r = draw_wishart(k + D, inv(D*covy + temp_para_sum))

        # draw alpha from posterior (depends on k, N), eq 15 (Rasmussen 2000)
        # Because its not standard form, using ARS to sampling
        alpha = draw_alpha(k, N)

        # draw sj from posterior (depends on mu, c, beta, w), eq 8 (Rasmussen 2000)
        for j,nj in enumerate(n):
            temp_para_sum = np.zeros((D, D))
            idx = np.argwhere(c == j)
            yj = np.reshape(Y[idx, : ], (idx.shape[0], D))
            for yi in yj:
                temp_para_sum += np.outer((mu[j, : ] - yi), np.transpose(mu[j, : ] - yi))
            temp_s = draw_wishart(beta + nj, inv(beta*w + temp_para_sum))
            s[j, :] = np.reshape(temp_s, (1, D * D))

        # compute the unrepresented probability - apply simulated annealing, eq 17 (Rasmussen 2000)
        p_unrep = (alpha/(N - 1.0 + alpha)) * integral_approx(Y, lam, r, beta, w, G, size=Nint)
        p_indicators_prior = np.outer(np.ones(k + 1), p_unrep)

        # for the represented components, eq 17 (Rasmussen 2000)
        for j in range(k):
            # n-i,j : the number of oberservations, excluding yi, that are associated with component j
            nij = n[j] - (c == j).astype(int)
            # only apply to indices where we have multi occupancy, the condition 1
            idx = np.argwhere(nij > 0)
            temp_sj = G*np.reshape(s[j,:], (D, D))     # apply simulated annealing to this parameter
            Q = np.array([np.dot(np.squeeze(Y[i, : ] - mu[j, : ]), np.dot(np.squeeze(Y[i, : ]-mu[j, : ]), temp_sj))\
                          for i in idx])
            p_indicators_prior[j, idx] = nij[idx]/(N - 1.0 + alpha)*np.reshape(np.exp(-0.5 * Q), idx.shape) \
                                         *np.sqrt(det(temp_sj))

        # stochastic indicator (we could have a new component)
        c = np.hstack(draw_indicator(p_indicators_prior))

        # draw w from posterior (depends on k, beta, D, sj), eq 9 (Rasmussen 2000)
        w = draw_wishart(k*beta + D, inv(D*inv_covy + beta*np.reshape(np.sum(s, 0), (D, D))))

        # draw beta from posterior (depends on k, s, w), eq 9 (Rasmussen 2000)
        # Because its not standard form, using ARS to sampling.
        beta = draw_beta(k, s, w)[0]

        # sort out based on new stochastic indicators
        nij = np.sum(c == k)        # see if the *new* component has occupancy
        if nij > 0:
            # draw from priors and increment k
            newmu = draw_MVNormal(mean=lam, cov=inv(r))
            news = draw_wishart(float(beta), inv(beta*w))
            mu = np.concatenate((mu, np.reshape(newmu, (1, D))))
            s = np.concatenate((s, np.reshape(news,(1, D*D))))
            k = k + 1
        # find the associated number for every components
        n = np.array([np.sum(c == j) for j in range(k)])

        # find unrepresented components
        badidx = np.argwhere(n == 0)
        Nbad = len(badidx)

        # remove unrepresented components
        if Nbad > 0:
            mu = np.delete(mu, badidx, axis=0)
            s = np.delete(s, badidx, axis=0)
            # if the unrepresented compont removed is in the middle, make the sequential component indicators change
            for cnt, i in enumerate(badidx):
                idx = np.argwhere(c >= (i - cnt))
                c[idx] = c[idx] - 1
            k -= Nbad        # update component number

        # recompute n
        n = np.array([np.sum(c == j) for j in range(k)])

        # recompute pi
        pi = n.astype(float)/np.sum(n)

        pcnt = int(100.0 * z / float(Nsamples))
        if pcnt > oldpcnt:
            print('{}: %--- {}% complete ----------------------%'.format(time.asctime(), pcnt))
            oldpcnt = pcnt

        # add sample
        S = Sample(mu, s, pi, lam, r, beta, w, alpha, k)
        newS = copy.deepcopy(S)
        Samp.addsample(newS)
        z += 1

    return Samp, Y


def integral_approx(y, lam, r, beta, w, G=1, size=100):
    """
    estimates the integral, eq 17 (Rasmussen 2000)
    """
    temp = np.zeros(len(y))
    inv_betaw = inv(beta * w)

    inv_r = inv(r)
    i = 0
    bad = 0
    while i < size:
        mu = mv_norm.rvs(mean=lam, cov=inv_r, size=1)
        s = draw_wishart(float(beta), inv_betaw)
        try:
            temp += mv_norm.pdf(y, mean=np.squeeze(mu), cov=G*inv(s))
        except:
            bad += 1
            pass
        i += 1
    return temp/float(size)


def log_p_alpha(alpha, k=1, N=1):
    """
    the log of eq15 (Rasmussen 2000)
    """
    return (k - 1.5)*np.log(alpha) - 0.5/alpha + special.gammaln(alpha) - special.gammaln(N + alpha)


def log_p_alpha_prime(alpha, k=1, N=1):
    """
    the derivative (wrt alpha) of the log of eq 15 (Rasmussen 2000)
    """
    return (k - 1.5)/alpha + 0.5/(alpha*alpha) + special.psi(alpha) - special.psi(alpha + N)


def log_p_beta(beta,k=1,s=1,w=1,D=1,logdet_w=1,cumculative_sum_equation=1):
    """
    The log of the second part of eq 9 (Rasmussen 2000)
    """
    return -1.5*np.log(beta - D + 1.0) \
        - 0.5*D/(beta - D + 1.0) \
        + 0.5*beta*k*D*np.log(0.5*beta) \
        + 0.5*beta*k*logdet_w \
        + 0.5*beta*cumculative_sum_equation \
        - k * special.multigammaln(0.5*beta, D)

def log_p_beta_prime(beta,k=1,s=1,w=1,D=1,logdet_w=1,cumculative_sum_equation=1):
    """
    The derivative (wrt beta) of the log of eq 9 (Rasmussen 2000)
    """
    psi = 0.0
    for j in range(1,D+1):
        psi += special.psi(0.5*beta + 0.5*(1.0 - j))
    return -1.5/(beta - D + 1.0) \
        + 0.5*D/(beta - D + 1.0)**2 \
        + 0.5*k*D*(1.0 + np.log(0.5*beta)) \
        + 0.5*k*logdet_w \
        + 0.5*cumculative_sum_equation \
    - 0.5*k*psi

# def draw_gamma_ras(a, theta, size=1):
#     """
#     returns Gamma distributed samples according to the Rasmussen (2000) definition
#     """
#     return gamma.rvs(0.5 * a, loc=0, scale=2.0 * theta / a, size=size)


def draw_gamma(a, theta, size=1):
    """
    returns Gamma distributed samples
    """
    return gamma.rvs(a, loc=0, scale=theta, size=size)


def draw_wishart(df, scale, size=1):
    """
    returns Wishart distributed samples
    """
    return wishart.rvs(df=df, scale=scale, size=size)


def draw_MVNormal(mean=0, cov=1, size=1):
    """
    returns multivariate normally distributed samples
    """
    return mv_norm.rvs(mean=mean, cov=cov, size=size)


def draw_alpha(k, N, size=1):
    """
    draw alpha from posterior (depends on k, N), eq 15 (Rasmussen 2000), using ARS
    Make it robust with an expanding range in case of failure
    """
    ars = ARS(log_p_alpha, log_p_alpha_prime, xi=[0.1, 5], lb=0, ub=np.inf, k=k, N=N)
    return ars.draw(size)


def draw_beta(k, s, w, size=1):
    """
    draw beta from posterior (depends on k, s, w), eq 9 (Rasmussen 2000), using ARS
    Make it robust with an expanding range in case of failure
    """
    D = w.shape[0]

    # compute Determinant of w, det(w)
    logdet_w = slogdet(w)[1]
    # compute cumulative sum j from i to k, [ log(det(sj))- trace(w * sj)]
    cumculative_sum_equation = 0
    for sj in s:
        sj = np.reshape(sj, (D, D))
        cumculative_sum_equation += slogdet(sj)[1]
        cumculative_sum_equation -= np.trace(np.dot(w, sj))

    lb = D
    ars = ARS(log_p_beta, log_p_beta_prime, xi=[lb + 0.1, lb + 1000], lb=lb, ub=float("inf"), \
              k=k, s=s, w=w, D=D, logdet_w=logdet_w, cumculative_sum_equation=cumculative_sum_equation)


    return ars.draw(size)


def draw_indicator(pvec):
    """
    draw stochastic indicator values from multinominal distributions, check wiki
    """
    res = np.zeros(pvec.shape[1])
    # loop over each data point
    for j in range(pvec.shape[1]):
        c = np.cumsum(pvec[ : ,j])        # the cumulative un-scaled probabilities
        R = np.random.uniform(0, c[-1], 1)        # a random number
        r = (c - R)>0                     # truth table (less or greater than R)
        y = (i for i, v in enumerate(r) if v)    # find first instant of truth
        try:
            res[j] = y.__next__()           # record component index
        except:                 # if no solution (must have been all zeros)
            res[j] = np.random.randint(0, pvec.shape[0]) # pick uniformly
    return res
