import sys
import numpy as np
from scipy.stats import gamma, wishart, norm, invgamma
from scipy.stats import multivariate_normal as mv_norm
from numpy.linalg import inv, det, slogdet
from scipy import special
from ars import ARS

# the maximum positive integer for use in setting the ARS seed
maxsize = sys.maxsize


def integral_approx_full_cov(y, lam, r, beta, w, G=1, size=100):
    """
    estimates the integral, eq 17 (Rasmussen 2000)
    the covariance matrix of the model is full cov
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


def integral_approx_diagonal_cov(y, lam, r, beta, w, G=1, size=100):
    """
    estimates the integral, eq 17 (Rasmussen 2000)
    the covariance matrix of the model is diagonal cov
    """
    N, D = y.shape
    temp = np.zeros(len(y))

    inv_r = inv(r)
    i = 0
    bad = 0
    while i < size:
        mu = mv_norm.rvs(mean=lam, cov=inv_r, size=1)
        s = np.diag([np.squeeze(draw_gamma(beta[d]/2 , 2/(beta[d]*w[d]))) for d in range(D)])
        try:
            temp_para = mv_norm.pdf(y, mean=np.squeeze(mu), cov=G*inv(s))
            temp += temp_para
        except:
            bad += 1
            pass
        i += 1
    return temp/float(size)


def log_p_alpha(alpha, k, N):
    """
    the log of eq15 (Rasmussen 2000)
    """
    return (k - 1.5)*np.log(alpha) - 0.5/alpha + special.gammaln(alpha) - special.gammaln(N + alpha)


def log_p_alpha_prime(alpha, k, N):
    """
    the derivative (wrt alpha) of the log of eq 15 (Rasmussen 2000)
    """
    return (k - 1.5)/alpha + 0.5/(alpha*alpha) + special.psi(alpha) - special.psi(alpha + N)


def log_p_beta_full_cov(beta,k=1,s=1,w=1,D=1,logdet_w=1,cumculative_sum_equation=1):
    """
    The log of the second part of eq 9 (Rasmussen 2000)
    the covariance matrix of the model is full cov
    """
    return -1.5*np.log(beta - D + 1.0) \
        - 0.5*D/(beta - D + 1.0) \
        + 0.5*beta*k*D*np.log(0.5*beta) \
        + 0.5*beta*k*logdet_w \
        + 0.5*beta*cumculative_sum_equation \
        - k*special.multigammaln(0.5*beta, D)


def log_p_beta_prime_full_cov(beta,k=1,s=1,w=1,D=1,logdet_w=1,cumculative_sum_equation=1):
    """
    The derivative (wrt beta) of the log of eq 9 (Rasmussen 2000)
    the covariance matrix of the model is full cov
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


def log_p_beta_diagonal_cov(beta,k=1,w=1,D=1,cumculative_sum_equation=1):
    """
    The log of the second part of eq 9 (Rasmussen 2000)
    the covariance matrix of the model is diagonal cov
    """
    return -k*special.gammaln(beta/2) \
        - 0.5/beta \
        + 0.5*(beta*k-3)*np.log(beta/2) \
        + 0.5*beta*cumculative_sum_equation


def log_p_beta_prime_diagonal_cov(beta,k=1,w=1,D=1,cumculative_sum_equation=1):
    """
    The derivative (wrt beta) of the log of eq 9 (Rasmussen 2000)
    the covariance matrix of the model is diagonal cov
    """
    return -k*special.psi(0.5*beta) \
        + 0.5/beta**2 \
        + 0.5*k*np.log(0.5*beta) \
        + (k*beta -3)/beta \
        + 0.5*cumculative_sum_equation


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


def draw_invgamma(a, theta, size=1):
    """
    returns inverse Gamma distributed samples
    """
    return invgamma.rvs(a, loc=0, scale=theta, size=size)


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


def draw_beta_full_cov(k, s, w, size=1):
    """
    draw beta from posterior (depends on k, s, w), eq 9 (Rasmussen 2000), using ARS
    the covariance matrix of the model is full cov
    Make it robust with an expanding range in case of failure
    """
    D = w.shape[0]

    # compute Determinant of w, det(w)
    logdet_w = slogdet(w)[1]
    # compute cumculative sum j from i to k, [ log(det(sj))- trace(w * sj)]
    cumculative_sum_equation = 0
    for sj in s:
        sj = np.reshape(sj, (D, D))
        cumculative_sum_equation += slogdet(sj)[1]
        cumculative_sum_equation -= np.trace(np.dot(w, sj))
    lb = D
    ars = ARS(log_p_beta_full_cov, log_p_beta_prime_full_cov, xi=[lb + 1000, lb + 1000], lb=lb, ub=float("inf"), \
              k=k, s=s, w=w, D=D, logdet_w=logdet_w, cumculative_sum_equation=cumculative_sum_equation)
    return ars.draw(size)

def draw_beta_diagonal_cov(k, s, w, d, D, size=1):
    """
    draw beta from posterior (depends on k, s, w), eq 9 (Rasmussen 2000), using ARS
    the covariance matrix of the model is diagonal cov
    Make it robust with an expanding range in case of failure
    """
    # compute cumculative sum j from i to k, [ log(sj) + log(w) - w*sj ]
    # 0.5*beta*cumculative_sum_equation
    cumculative_sum_equation = 0
    for sj in s:
        sj = np.reshape(sj, (D, D))
        cumculative_sum_equation += np.log(sj[d, d])
        cumculative_sum_equation += np.log(w[d])
        cumculative_sum_equation -= w[d]*sj[d, d]
    lb = D
    ars = ARS(log_p_beta_diagonal_cov, log_p_beta_prime_diagonal_cov, xi=[lb + 15], lb=lb, ub=float("inf"), \
              k=k, w=w, D=D, cumculative_sum_equation=cumculative_sum_equation)
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
