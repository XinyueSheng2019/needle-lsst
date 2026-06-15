import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filter_colours = {'u': '#6A5ACD', 'g': '#2ca02c', 'r': '#d62728', 'i': '#ff7f0e', 'z': '#8c564b', 'y': '#1b1b1b'}

path = '/Users/dylanmagill/Downloads/train_full_lightcurves.csv'

df = pd.read_csv(path)

object_ID = 'alae_hu_Mithrim'

u_mask = ((df['Filter'] == 'u') & (df['Object_ID'] == object_ID))
g_mask = ((df['Filter'] == 'g') & (df['Object_ID'] == object_ID))
r_mask = ((df['Filter'] == 'r') & (df['Object_ID'] == object_ID))
i_mask = ((df['Filter'] == 'i') & (df['Object_ID'] == object_ID))
z_mask = ((df['Filter'] == 'z') & (df['Object_ID'] == object_ID))
y_mask = ((df['Filter'] == 'y') & (df['Object_ID'] == object_ID))

u_flux = np.array(df.loc[u_mask, 'Flux']); u_time = np.array(df.loc[u_mask, 'Time (MJD)']); u_err = np.array(df.loc[u_mask, 'Flux_err'])
g_flux = np.array(df.loc[g_mask, 'Flux']); g_time = np.array(df.loc[g_mask, 'Time (MJD)']); g_err = np.array(df.loc[g_mask, 'Flux_err'])
r_flux = np.array(df.loc[r_mask, 'Flux']); r_time = np.array(df.loc[r_mask, 'Time (MJD)']); r_err = np.array(df.loc[r_mask, 'Flux_err'])
i_flux = np.array(df.loc[i_mask, 'Flux']); i_time = np.array(df.loc[i_mask, 'Time (MJD)']); i_err = np.array(df.loc[i_mask, 'Flux_err'])
z_flux = np.array(df.loc[z_mask, 'Flux']); z_time = np.array(df.loc[z_mask, 'Time (MJD)']); z_err = np.array(df.loc[z_mask, 'Flux_err'])
y_flux = np.array(df.loc[y_mask, 'Flux']); y_time = np.array(df.loc[y_mask, 'Time (MJD)']); y_err = np.array(df.loc[y_mask, 'Flux_err'])

print(len(u_flux) + len(g_flux) + len(r_flux) + len(i_flux) + len(z_flux) + len(y_flux))

plt.errorbar(u_time,u_flux, yerr= u_err,label='u', fmt = '.',color = filter_colours['u'])
plt.errorbar(g_time,g_flux,yerr=g_err, label='g', fmt = '.', color = filter_colours['g'])
plt.errorbar(r_time,r_flux,yerr = r_err, label='r', fmt = '.', color = filter_colours['r'])
plt.errorbar(i_time,i_flux,yerr = i_err, label='i', fmt = '.', color = filter_colours['i'])
plt.errorbar(z_time,z_flux,yerr = z_err, label='z', fmt = '.', color = filter_colours['z'])
plt.errorbar(y_time,y_flux,yerr = y_err, label='y', fmt = '.', color = filter_colours['y'])
plt.xlabel('Days - Relative to Peak')
plt.ylabel('Flux - μJy')
#plt.xlim(-750,750)
plt.legend()
#plt.savefig('')
#plt.close()
plt.show()