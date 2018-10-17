import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from scipy.stats import norm, uniform, gamma, chi2, wishart
from scipy.stats import multivariate_normal as mv_norm
from numpy.linalg import inv,eig,det,slogdet
from scipy import special
import copy
import time
import sys
from ars import ARS

# the maximum positive integer for use in setting the ARS seed
maxsize = sys.maxsize


class Sample:
    """Class for defining a single sample"""
    def __init__(self,mu,s,pi,lam,r,beta,w,alpha,k):
        self.mu = np.reshape(mu,(1,-1))
        self.s = np.reshape(s,(1,-1))
        self.pi = np.reshape(pi,(1,-1))
        self.lam = lam
        self.r = r
        self.beta = beta
        self.w = w
        self.k = k
        self.alpha = alpha

class Samples:
    """Class for generating a collection of samples"""
    def __init__(self,N,nd):
        self.sample = []
        self.N = N
        self.nd = nd

    def __getitem__(self, key): 
        return self.sample[key]

    def addsample(self,S):
        return self.sample.append(S)

# the sampler
def igmm_sampler(Y, Nsamples, Nint=1, anneal=False):
    """Takes command line args and computes samples from the joint posterior
    using Gibbs sampling

    input:
        Y - the input dataset
        Nsamples - the number of Gibbs samples
        Nint - the samples used for evaluating the tricky integral
        anneal - perform simple siumulated annealing

    output:
        Samp - the output samples

    """
    # compute some data derived quantities
    N,nd = Y.shape
    muy = np.mean(Y, axis=0)
    covy = np.cov(Y.transpose())
    inv_covy = inv(covy) if nd>1 else np.reshape(1.0/covy,(1,1))

    # initialise a single sample
    Samp = Samples(Nsamples,nd)

    c = np.zeros(N)            # initialise the stochastic indicators
    pi = np.zeros(1)           # initialise the weights
    mu = np.zeros((1,nd))      # initialise the means
    s = np.zeros((1,nd*nd))    # initialise the precisions
    n = np.zeros(1)            # initialise the occupation numbers

    mu[0,:] = muy              # set first mu to the mean of all data
    pi[0] = 1.0                # only one component so pi=1

    # draw beta from prior
    # y = (beta - D + 1)^ (-1) , y is subject to gamma dist(Rasmussen 2000)
    # gamma dist (Rasmussen 2000) use scale and mean, is different. alpha need to change to 1/2*alpha
    # theta parameter change to mean, theta * 1/alpha. So theta of Rasmussen's should be 2*theta/a
    # For multivariate part, mean of Gamma is 1/nd not 1. So we need "/float(nd)"
    temp_y = drawGamma(0.5,2.0/float(nd))
    beta = np.squeeze(float(nd) - 1.0 + 1.0/temp_y)

    # draw w from prior
    # the df(degrees of freedom) should be dimensions nd, scale matrix should be covariance matrix / dimensions nd
    w = drawWishart(nd, covy/float(nd))

    # draw s from prior
    # according to the paper (Rasmussen 2006), Sj is subject to Wishart(beta, (beta * w)-1)
    s[0,:] = np.squeeze(np.reshape(drawWishart(float(beta),inv(beta*w)),(nd*nd,-1)))
    n[0] = N                   # all samples are in the only component

    # draw lambda from prior
    # the mean is muy, the variance matrix is covy
    lam = drawMVNormal(mean=muy,cov=covy)

    # draw r from prior
    # the df is dimensions nd, scale matrix is (nd * cov)^(-1)
    r = drawWishart(nd,inv(nd*covy))

    # draw alpha from prior
    # (alpha)^(-1) is subject to Rasmussen's paper's gamma distribution, scale is 1, mean is 1
    # so (alphs)^(-1) is sujbect to gamma distribution, scale is 1/2, and theta is 2
    alpha = 1.0/drawGamma(0.5,2.0)
    k = 1                                       # set only 1 component
    S = Sample(mu,s,pi,lam,r,beta,w,alpha,k)    # define the sample
    Samp.addsample(S)                           # add the sample
    print('{}: initialised parameters'.format(time.asctime()))

    # loop over samples
    z = 1
    oldpcnt = 0
    while z<Nsamples:
        # define simulated annealing temperature
        G = max(1.0,float(0.5*Nsamples)/float(z+1)) if anneal else 1.0

        # recompute muy and covy
        muy = np.mean(Y,axis=0)
        covy = np.cov(Y,rowvar=0)
        inv_covy = inv(covy) if nd>1 else np.reshape(1.0/covy,(1,1))

        # calculate yj bar, for each represented muj value
        ybarj = [np.sum(Y[np.argwhere(c==j),:],0)/nj for j,nj in enumerate(n)]
        mu = np.zeros((k,nd))
        j = 0
        # sampling muj from posterior (depends on sj, c, lambda, r).
        for yb, nj, sj in zip(ybarj, n, s):
            sj = np.reshape(sj,(nd, nd))
            # (Rasmussen 2000) equation (4): To get posterior of muj, fist calculate cov and mean, and then sampling
            muj_cov = inv(nj * sj + r)
            muj_mean = np.dot(muj_cov, nj * np.dot(sj, np.squeeze(yb)) + np.dot(r, lam))
            mu[j,:] = drawMVNormal(mean=muj_mean, cov=muj_cov, size=1)
            j += 1

        # sampling lambda from posterior (depends on mu vector, k, and r).
        # From (Rasmussen 2000) equation (5)
        lam_cov = inv(inv_covy + k*r)
        lam_mean = np.dot(lam_cov ,np.dot(inv_covy, muy) + np.dot(r, np.sum(mu, axis=0)))
        lam = drawMVNormal(mean=lam_mean,cov=lam_cov, size=1)

        # sampling r from posterior (depnds on k, mu, and lambda).
        # From (Rasmussen 2000) equation (5)
        temp = np.zeros((nd,nd))
        for muj in mu:
            temp += np.outer((muj-lam),np.transpose(muj-lam))
        r = drawWishart(k+nd,inv(nd*covy + temp))

        # sampling alpha from posterior (depends on k, N). Because its not standard form, using ARS to sampling.
        alpha = drawAlpha(k,N)

        # sampling sj from posterior (depends on mu, c, beta, w).
        # the equation is not included in paper (Rasmussen 2000)
        for j,nj in enumerate(n):
            temp = np.zeros((nd,nd))
            idx = np.argwhere(c==j)
            yj = np.reshape(Y[idx,:],(idx.shape[0],nd))
            for yi in yj:
                temp += np.outer((mu[j,:]-yi),np.transpose(mu[j,:]-yi))
            temp_s = drawWishart(beta + nj,inv(beta*w + temp))
            s[j,:] = np.reshape(temp_s,(1,nd*nd))

        # compute the unrepresented probability - apply simulated annealing
        p_unrep = (alpha/(N-1.0+alpha)) * IntegralApprox(Y,lam,r,beta,w,G,size=Nint)
        p_indicators_prior = np.outer(np.ones(k+1),p_unrep)

        # for the represented components
        # From (Rasmussen 2000) equation (17), the first line
        for j in range(k):
            # n-i,j : the number of oberservations, excluding yi, that are associated with component j
            nij = n[j] - (c==j).astype(int)
            # only apply to indices where we have multi occupancy, the condition 1
            idx = np.argwhere(nij>0)
            temp_sj = G*np.reshape(s[j,:],(nd,nd))     # apply simulated annealing to this parameter
            Q = np.array([np.dot(np.squeeze(Y[i,:]-mu[j,:]),np.dot(np.squeeze(Y[i,:]-mu[j,:]),temp_sj)) for i in idx])
            p_indicators_prior[j,idx] = nij[idx]/(N-1.0+alpha)*np.reshape(np.exp(-0.5*Q),idx.shape)*np.sqrt(det(temp_sj))

        # stochastic indicator (we could have a new component)
        c = np.hstack(drawIndicator(p_indicators_prior))

        # for w
        w = drawWishart(k*beta + nd,inv(nd*inv_covy + beta*np.reshape(np.sum(s,0),(nd,nd))))

        # from beta
        beta = drawBeta(k,s,w)[0]


        # sort out based on new stochastic indicators
        nij = np.sum(c==k)        # see if the *new* component has occupancy
        if nij>0:
            # draw from priors and increment k
            newmu = drawMVNormal(mean=lam,cov=inv(r))
            news = drawWishart(float(beta),inv(beta*w))
            mu = np.concatenate((mu,np.reshape(newmu,(1,nd))))
            s = np.concatenate((s,np.reshape(news,(1,nd*nd))))
            k = k + 1
        # find the associated number for every components
        n = np.array([np.sum(c==j) for j in range(k)])
        # find unrepresented components
        badidx = np.argwhere(n==0)
        Nbad = len(badidx)

        # remove unrepresented components
        if Nbad>0:
            mu = np.delete(mu,badidx,axis=0)
            s = np.delete(s,badidx,axis=0)
            # if the unrepresented compont removed is in the middle, make the sequential component indicators change
            for cnt,i in enumerate(badidx):
                idx = np.argwhere(c>=(i-cnt))
                c[idx] = c[idx]-1
            k -= Nbad        # update component number

        # recompute n
        n = np.array([np.sum(c==j) for j in range(k)])

        # recompute pi
        pi = n.astype(float)/np.sum(n)

        pcnt = int(100.0*z/float(Nsamples))
        if pcnt>oldpcnt:
            print('{}: %--- {}% complete ----------------------%'.format(time.asctime(),pcnt))
            oldpcnt = pcnt

        # add sample
        S = Sample(mu,s,pi,lam,r,beta,w,alpha,k)
        newS = copy.deepcopy(S)
        Samp.addsample(newS)
        z += 1

    return Samp, Y



def IntegralApprox(y,lam,r,beta,w,G=1,size=100):
    """estimates the integral in Eq.17 of Rasmussen (2000)"""
    temp = np.zeros(len(y))
    inv_betaw = inv(beta*w)

    inv_r = inv(r)
    i = 0
    bad = 0
    while i < size:
        mu = mv_norm.rvs(mean=lam,cov=inv_r,size=1)
        s = drawWishart(float(beta),inv_betaw)
        try:
            temp += mv_norm.pdf(y,mean=np.squeeze(mu),cov=G*inv(s))
        except:
            bad += 1
            pass
        i += 1
    return temp/float(size)

def logpalpha(alpha,k=1,N=1):
    """The log of Eq.15 in Rasmussen (2000)"""
    return (k-1.5)*np.log(alpha) - 0.5/alpha + special.gammaln(alpha) - special.gammaln(N+alpha)

def logpalphaprime(alpha,k=1,N=1):
    """The derivative (wrt alpha) of the log of Eq.15 in Rasmussen (2000)"""
    return (k-1.5)/alpha + 0.5/(alpha*alpha) + special.psi(alpha) - special.psi(alpha+N)

def logpbeta(beta,k=1,s=1,w=1,nd=1,logdet_w=1,cumculative_sum_equation=1):
    """The log of the second part of Eq.9 in Rasmussen (2000)"""
    return -1.5*np.log(beta - nd + 1.0) \
        - 0.5*nd/(beta - nd + 1.0) \
        + 0.5*beta*k*nd*np.log(0.5*beta) \
        + 0.5*beta*k*logdet_w \
        + 0.5*beta*cumculative_sum_equation \
        - k * special.multigammaln(0.5*beta, nd)

def logpbetaprime(beta,k=1,s=1,w=1,nd=1,logdet_w=1,cumculative_sum_equation=1):
    """The derivative (wrt beta) of the log of Eq.9 in Rasmussen (2000)"""
    psi = 0.0
    for j in range(1,nd+1):
        psi += special.psi(0.5*beta + 0.5*(1.0 - j))
    return -1.5/(beta - nd + 1.0) \
        + 0.5*nd/(beta - nd + 1.0)**2 \
        + 0.5*k*nd*(1.0 + np.log(0.5*beta)) \
        + 0.5*k*logdet_w \
        + 0.5*cumculative_sum_equation \
        - 0.5*k*psi

# def drawGammaRas(a,theta,size=1):
#     """Returns Gamma distributed samples according to
#     the Rasmussen (2000) definition"""
#     return gamma.rvs(0.5*a,loc=0,scale=2.0*theta/a,size=size)

def drawGamma(a, theta, size=1):
    """
    Returns Gamma distributed samples
    """
    return gamma.rvs(a, loc=0, scale=theta, size=size)

def drawWishart(df, scale, size=1):
    """
    Returns Wishart distributed samples
    """
    return wishart.rvs(df=df, scale=scale, size=size)

def drawMVNormal(mean=0,cov=1,size=1):
    """
    Returns multivariate normally distributed samples
    """
    return mv_norm.rvs(mean=mean,cov=cov,size=size)

def drawIndicator(pvec):
    """Draws stochastic indicator values from multinomial distributions """
    res = np.zeros(pvec.shape[1])
    # loop over each data point
    for j in range(pvec.shape[1]):
        c = np.cumsum(pvec[:,j])        # the cumulative un-scaled probabilities
        R = np.random.uniform(0,c[-1],1)        # a random number
        r = (c-R)>0                     # truth table (less or greater than R)
        y = (i for i,v in enumerate(r) if v)    # find first instant of truth
        try:
            res[j] = y.__next__()           # record component index
        except:                 # if no solution (must have been all zeros)
            res[j] = np.random.randint(0,pvec.shape[0]) # pick uniformly
    return res


def drawAlpha(k, N, size=1):
    """
    Draw alpha from its distribution (Eq.15 Rasmussen 2000) using ARS
    Make it robust with an expanding range in case of failure
    """
    ars = ARS(logpalpha, logpalphaprime, xi=[0.1, 0.6], lb=0, ub=np.inf, k=k, N=N)
    # draw alpha but also pass random seed to ARS code
    return ars.draw(size)

def  drawBeta(k,s,w,size=1):
    """Draw beta from its distribution (Eq.9 Rasmussen 2000) using ARS
    Make it robust with an expanding range in case of failure"""
    nd = w.shape[0]
    
    # compute Determinant of w, det(w)
    logdet_w = slogdet(w)[1]
    # compute cumulative sum j from i to k, [ log(det(sj))- trace(w * sj)]
    cumculative_sum_equation = 0
    for sj in s:
        sj = np.reshape(sj,(nd,nd))
        cumculative_sum_equation += slogdet(sj)[1]
        cumculative_sum_equation -= np.trace(np.dot(w,sj))
    
    lb = nd
    ars = ARS(logpbeta, logpbetaprime, xi= [lb+0.1,lb+1000], lb=lb, ub = float("inf"),\
                k=k, s=s, w=w, nd=nd, logdet_w=logdet_w, cumculative_sum_equation=cumculative_sum_equation)
    return ars.draw(size)

def greedy(x):
    """computes the enclosed probability
    """
    s = x.shape
    x = np.reshape(x,(1,-1))
    x = np.squeeze(x/np.sum(x))
    idx = np.squeeze(np.argsort(x))
    test = x[idx]
    z = np.cumsum(x[idx])
    d = np.zeros(len(z))
    d[idx] = z
    return 1.0 - np.reshape(d,s)


def plotresult(Samp,Y,outfile,Ngrid=100,M=4,plottype='ellipse'):
    """Plots samples of ellipses drawn from the posterior"""
    nd = Samp.nd
    N = Samp.N
    lower = np.min(Y,axis=0)
    upper = np.max(Y,axis=0)
    lower = lower - 0.5*(upper-lower)
    upper = upper + 0.5*(upper-lower)
    xvec = np.zeros((nd,Ngrid))
    for i in range(nd):
        xvec[i,:] = np.linspace(lower[i],upper[i],Ngrid)
    label = ['$x_{}$'.format(i) for i in range(nd)]
    levels = [0.68, 0.95,0.999]
    alpha = [1.0, 0.5, 0.2]

    plt.figure(figsize = (nd,nd))
    gs1 = gridspec.GridSpec(nd, nd)
    gs1.update(left=0.15, right=0.85, top=0.85, bottom=0.15, wspace=0, hspace=0)

    # pick random samples to use
    randidx = np.random.randint(N/2,N,M)

    
    cnt = 0
    for i in range(nd):
        for j in range(nd):
            
            ij = np.unravel_index(cnt,[nd,nd])
            ax1 = plt.subplot(gs1[ij])
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])    

            # scatter plot the data in lower triangle plots
            if i>j:
                ax1.plot(Y[:,j],Y[:,i],'r.',alpha=0.5,markersize=0.5)
                ax1.set_xlim([lower[j],upper[j]])
                ax1.set_ylim([lower[i],upper[i]])
            elif i==j:  # otherwise on the diagonal plot histograms
                if nd>1:
                    ax1.set_xlim([lower[j],upper[j]])
                else:
                    plt.xlim([lower[j],upper[j]])
                    plt.ylim([lower[i],upper[i]])

            # if off the diagonal
            if i>=j:
                # loop over randomly selected samples
                for k in randidx:
                    samples = Samp[k]
                    s = np.reshape(samples.s,(samples.k,nd*nd))
                    m = np.reshape(samples.mu,(samples.k,nd))
                    p = np.reshape(np.array(np.squeeze(samples.pi)),(-1,1))

                    # loop over components in this sample
                    for b in range(samples.k):
                        tempC = inv(np.reshape(s[b,:],(nd,nd)))
                        ps = tempC[np.ix_([i,j],[i,j])] if i!=j else tempC[i,i]

                        # if we have a 2D covariance after projecting
                        if ps.size==4:
                            w,v = eig(ps)
                            e = Ellipse(xy=m[b,[j,i]],width=2.0*np.sqrt(6.0*w[1]), \
                                height=2*np.sqrt(6.0*w[0]), \
                                angle=(180.0/np.pi)*np.arctan2(v[0,1],v[0,0]), \
                                alpha=np.squeeze(p[b]))
                            e.set_facecolor('none')
                            e.set_edgecolor('b')
                            ax1.add_artist(e)
                        elif ps.size==1:
                            if nd>1:
                                ax1.plot(xvec[i,:],p[b]*norm.pdf(xvec[i,:],loc=m[b,i],scale=np.sqrt(np.squeeze(ps))),'b',alpha=p[b])
                            else:
                                plt.plot(xvec[i,:],p[b]*norm.pdf(xvec[i,:],loc=m[b,i],scale=np.sqrt(np.squeeze(ps))),'b',alpha=p[b])
                        else:
                            print('{}: ERROR strange number of elements in projected matrix'.format(time.asctime()))
                            exit(0)

            if j>i:
                ax1.axis('off') if nd>1 else plt.axis('off')
            if cnt>=nd*(nd-1):
                plt.xlabel(label[j],fontsize=12)
                ax1.xaxis.labelpad = -5
            if (cnt % nd == 0) and cnt>0:
                plt.ylabel(label[i],fontsize=12)
                ax1.yaxis.labelpad = -3
            cnt += 1

    plt.savefig(outfile,dpi=300)

