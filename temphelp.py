import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

data = sm.datasets.get_rdataset("dietox", "geepack").data
md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"])
mdf = md.fit()
print(mdf.summary())

data1 = np.load("dabug_data_1.npy")
# =============================================================================
# my_data= pd.DataFrame(data = np.concatenate((data1[0:,0:], np.zeros((len(data1), 1))), axis = 1), 
#              
#              columns = ["X","Y", "Sudo"]) 
# =============================================================================
my_data = pd.DataFrame(data    = data1[0 :, 0 :],
                       columns = ["X", "Y"])
md = smf.ols("X ~ Y", my_data)
mdf = md.fit()
print(mdf.summary())
