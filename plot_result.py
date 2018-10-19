import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from numpy.linalg import inv,eig
from scipy.stats import norm


def plot_result(Samp, Y, outfile, Ngrid=100, M=4):
    """Plots samples of ellipses drawn from the posterior"""
    D = Samp.D
    N = Samp.N
    lower = np.min(Y, axis=0)
    upper = np.max(Y, axis=0)
    lower = lower - 0.5*(upper - lower)
    upper = upper + 0.5*(upper - lower)
    xvec = np.zeros((D, Ngrid))
    for i in range(D):
        xvec[i, :] = np.linspace(lower[i], upper[i], Ngrid)
    label = ['$x_{}$'.format(i) for i in range(D)]
    levels = [0.68, 0.95, 0.999]
    alpha = [1.0, 0.5, 0.2]

    plt.figure(figsize=(D, D))
    gs1 = gridspec.GridSpec(D, D)
    gs1.update(left=0.15, right=0.85, top=0.85, bottom=0.15, wspace=0, hspace=0)

    # pick random samples to use
    randidx = np.random.randint(N / 2, N, M)

    cnt = 0
    for i in range(D):
        for j in range(D):
            ij = np.unravel_index(cnt, [D, D])
            ax1 = plt.subplot(gs1[ij])
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

            # scatter plot the data in lower triangle plots
            if i > j:
                ax1.plot(Y[:, j], Y[:, i], 'r.', alpha=0.5, markersize=0.5)
                ax1.set_xlim([lower[j], upper[j]])
                ax1.set_ylim([lower[i], upper[i]])
            elif i == j:  # otherwise on the diagonal plot histograms
                if D > 1:
                    ax1.set_xlim([lower[j], upper[j]])
                else:
                    plt.xlim([lower[j], upper[j]])
                    plt.ylim([lower[i], upper[i]])

            # if off the diagonal
            if i >= j:
                # loop over randomly selected samples
                for k in randidx:
                    samples = Samp[k]
                    s = np.reshape(samples.s, (samples.k, D*D))
                    m = np.reshape(samples.mu, (samples.k, D))
                    p = np.reshape(np.array(np.squeeze(samples.pi)), (-1, 1))

                    # loop over components in this sample
                    for b in range(samples.k):
                        tempC = inv(np.reshape(s[b, :], (D, D)))
                        ps = tempC[np.ix_([i, j], [i, j])] if i != j else tempC[i, i]

                        # if we have a 2D covariance after projecting
                        if ps.size == 4:
                            w, v = eig(ps)
                            e = Ellipse(xy=m[b, [j, i]], width=2.0 * np.sqrt(6.0*w[1]), \
                                        height=2*np.sqrt(6.0*w[0]), \
                                        angle=(180.0 / np.pi)*np.arctan2(v[0, 1], v[0, 0]), \
                                        alpha=np.squeeze(p[b]))
                            e.set_facecolor('none')
                            e.set_edgecolor('b')
                            ax1.add_artist(e)
                        elif ps.size == 1:
                            if D > 1:
                                ax1.plot(xvec[i, :],
                                         p[b]*norm.pdf(xvec[i, :], loc=m[b, i], scale=np.sqrt(np.squeeze(ps))), 'b',
                                         alpha=p[b])
                            else:
                                plt.plot(xvec[i, :],
                                         p[b]*norm.pdf(xvec[i, :], loc=m[b, i], scale=np.sqrt(np.squeeze(ps))), 'b',
                                         alpha=p[b])
                        else:
                            print('{}: ERROR strange number of elements in projected matrix'.format(time.asctime()))
                            exit(0)

            if j > i:
                ax1.axis('off') if D > 1 else plt.axis('off')
            if cnt >= D * (D - 1):
                plt.xlabel(label[j], fontsize=12)
                ax1.xaxis.labelpad = -5
            if (cnt % D == 0) and cnt > 0:
                plt.ylabel(label[i], fontsize=12)
                ax1.yaxis.labelpad = -3
            cnt += 1

    plt.savefig(outfile, dpi=300)

