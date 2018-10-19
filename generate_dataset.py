import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# numpy.random.multivariate normal simulation is not well compared with scipy's function
# from numpy.random import multivariate_normal
from scipy.stats import multivariate_normal as mv_norm


mean_component_1 = [0, 0]
cov_component_1 = [[1, 0], [0, 100]]  # diagonal covariance
# data = multivariate_normal(mean=mean, cov=cov, size=1000)
data_component_1 = mv_norm.rvs(mean=mean_component_1, cov=cov_component_1, size=1000)

mean_component_2 = [10, 5]
cov_component_2 = [[2.5 , 0], [0, 4]]  # diagonal covariance
data_component_2 = mv_norm.rvs(mean=mean_component_2, cov=cov_component_2, size=800)

mean_component_3 = [15, 13]
cov_component_3 = [[0.5 , 0], [0, 2]]  # diagonal covariance
data_component_3 = mv_norm.rvs(mean=mean_component_3, cov=cov_component_3, size=900)

mean_component_4 = [-10, -7]
cov_component_4 = [[4 , 0], [0, 1]]  # diagonal covariance
data_component_4 = mv_norm.rvs(mean=mean_component_4, cov=cov_component_4, size=650)


dataset = np.concatenate([data_component_1, data_component_2, data_component_3, data_component_4], axis=0)
plt.scatter(dataset[:,0], dataset[:,1])
# plt.show()

df = pd.DataFrame(data=dataset)
# to csv without index and column number
df.to_csv(path_or_buf="dataset/MVN_4components_digonal_cov.csv", index=False, header=False)