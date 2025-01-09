import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv("/home/user/Desktop/MACE-code/experiment/plot/loss_distribution/loss_results.csv")
plt.figure()
plt.scatter(df['Number_of_Atoms'], np.sqrt(df['Energy_Loss']))
plt.savefig('energy_error.png', dpi=300)

plt.figure()
plt.scatter(df['Number_of_Atoms'], np.sqrt(df['Forces_Loss']))
plt.savefig('forces_error.png', dpi=300)
# plt.figure()
# plt.hist(np.sqrt(df['Energy_Loss'].values), bins=50)
# plt.savefig('energy_hist.png', dpi=300)

# plt.figure()
# plt.hist(np.sqrt(df['Forces_Loss'].values), bins=50)
# plt.savefig('forces_hist.png', dpi=300)
df['sqrt_eng'] = np.sqrt(df['Energy_Loss'])
df['sqrt_force'] = np.sqrt(df['Forces_Loss'])

print(df.describe())
subdf = df[df['Number_of_Atoms']!=77]
print(subdf.describe())