# IGMM
infinite Gaussian Mixture Model(Dirichlet Process Gaussian Mixture Model) by gibbs sampling. 

Multivaraite Gaussian Distribution with full covariance and diagonal covariance matrix. Implementing IGMM with full covariance will use gibbs sampling. If there is diagonal covariance matrix, it can become the product of lots of Univariate Guassian distribution. In this condition, I update algorithms and change the formulas about Sj which are eq 6 - 9 in (Rasmussen 2000).

Command line:
implementing by full covariance(if not assign args, default is full) : python main.py -i datasets/MVN_3components_full_cov.csv

implementing by full covariance : python main.py -c diagonal -i datasets/MVN_4components_diagonal_cov.csv

implementing by diagonal covariance : python main.py -c diagonal -i datasets/MVN_4components_diagonal_cov.csv


