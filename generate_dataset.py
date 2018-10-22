import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# numpy.random.multivariate normal simulation is not well compared with scipy's function
# from numpy.random import multivariate_normal
from scipy.stats import multivariate_normal as mv_norm


mean_component_1 = [0, 5]
cov_component_1 = [[1, 0], [0, 1]]  # diagonal covariance
# data = multivariate_normal(mean=mean, cov=cov, size=1000)
data_component_1 = mv_norm.rvs(mean=mean_component_1, cov=cov_component_1, size=1000)

mean_component_2 = [0, -5]
cov_component_2 = [[1 , 0], [0, 1]]  # diagonal covariance
data_component_2 = mv_norm.rvs(mean=mean_component_2, cov=cov_component_2, size=1000)

mean_component_3 = [5, 0]
cov_component_3 = [[1 , 0], [0, 1]]  # diagonal covariance
data_component_3 = mv_norm.rvs(mean=mean_component_3, cov=cov_component_3, size=1000)

mean_component_4 = [-5, 0]
cov_component_4 = [[1 , 0], [0, 1]]  # diagonal covariance
data_component_4 = mv_norm.rvs(mean=mean_component_4, cov=cov_component_4, size=1000)


dataset = np.concatenate([data_component_1, data_component_2, data_component_3, data_component_4], axis=0)
print(np.cov(dataset.transpose()))
plt.scatter(dataset[:,0], dataset[:,1])
plt.show()

df = pd.DataFrame(data=dataset)
# to csv without index and column number
df.to_csv(path_or_buf="datasets/MVN_4components_diagonal_cov.csv", index=False, header=False)