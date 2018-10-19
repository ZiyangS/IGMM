import argparse
import time
import sys
from igmm import igmm_full_cov_sampler
from plot_result import plot_result
import pandas as pd

# the maximum positive integer for use in setting the ARS seed
maxsize = sys.maxsize

def parser():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(prog='main.py', description='Applies an N-Dimensional infinite Gaussian mixture model to data')


    # arguments for multivariate gaussian distribution covariance matrix type, full or diagonal
    parser.add_argument('-c', '--cov_type', type=str, default="full", help='covariance matrix type, input full or diagonal')
    # arguments for reading in a data file
    parser.add_argument('-i', '--inputfile', type=str, default=None, help='the input file name')
    # arguments for sampling number
    parser.add_argument('-n', '--Nsamples', type=int, default=2000, help='the number of gibbs samples to produce')
    # general analysis parameters
    parser.add_argument('-I', '--Nint', type=int, default=100, help='the number of samples used in approximating the tricky integral')
    parser.add_argument('-a', '--anneal', action='count', default=0, help='perform simulated annealing')

    # catch any input errors   
    args = parser.parse_args()
    if args.cov_type and args.cov_type not in ["full", "diagonal"]:
        print('{} : ERROR - the covariance matrix must be full or diagonal.'.format(time.asctime()))
        exit(1)
    if not args.inputfile:
        print('{} : ERROR - must specify an input file.'.format(time.asctime()))
        exit(1)
    if args.Nint < 1:
        print('{} : ERROR - the integration samples must be > 0. Exiting.'.format(time.asctime()))
        exit(1)
    if args.Nsamples < 1:
        print('{} : ERROR - the number of igmm samples must be > 0. Exiting.'.format(time.asctime()))
        exit(1)

    return parser.parse_args()


def readdata(inputfile):
    """
        reads in data from an input text file
        inputfile - the name of the input file
    """
    dataset_df = pd.read_csv(inputfile, header=None)
    dataset = dataset_df.values
    return dataset


# the main part of the code
def main():
    """Takes command line args and computes samples from the joint posterior
    using Gibbs sampling"""

    # record the start time
    t = time.time()

    # get the command line args
    args = parser()

    # read in data if required
    if args.inputfile:
        Y = readdata(args.inputfile)

    # call igmm Gibbs sampler
    Samp, Y = igmm_full_cov_sampler(Y, cov_type=args.cov_type, Nsamples=args.Nsamples, Nint=args.Nint,
                                    anneal=args.anneal)

    # print computation time
    print("{}: time to complete main analysis = {} sec".format(time.asctime(), time.time() - t))

    # plot chains, histograms, average maps, and overlayed ellipses
    print('{}: making output plots'.format(time.asctime()) )
    plot_result(Samp, Y, "graphs/" + args.cov_type + '_ellipses.png', M=4, Ngrid=100)

    print('{}: success'.format(time.asctime())  )


if __name__ == "__main__":
    exit(main())












