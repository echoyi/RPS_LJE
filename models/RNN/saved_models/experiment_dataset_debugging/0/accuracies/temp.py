import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

acc_ours = np.load('accuracies_ours.npy')
acc_rep = np.load('accuracies_rep.npy')
acc_random = np.load('accuracies_random.npy')

d = {
    'Ours': acc_ours,
    'Representer': acc_rep,
    'Random': acc_random
}

idx = pd.Index(data=np.linspace(0.05, 0.4, 8), name='Fraction of training data checked')
df = pd.DataFrame(data=d, index=idx)
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.lineplot(data=df, markers=True)
for l in ax.lines:
    xy = l.get_xydata()
    if len(xy) > 0:
        for data in xy:
            x = data[0]
            y = data[1]
            ax.annotate(f'{y:.3f}', xy=(x, y), xycoords=('data', 'data'),
                        ha='left', va='center', color=l.get_color())
plt.ylabel('test accuracy')
plt.tight_layout()
plt.savefig('accuracies_plot_temp.png')
