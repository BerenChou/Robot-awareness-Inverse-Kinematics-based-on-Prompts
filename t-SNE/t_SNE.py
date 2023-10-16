from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


tsne = TSNE(n_components=2, random_state=42)

train_jas = np.load('data_without_mutation/training/jas_npy.npy').reshape(60000, -1)  # (60000, 600)
val_jas = np.load('data_without_mutation/validation/jas_npy.npy').reshape(20000, -1)  # (20000, 600)

all_data = np.concatenate((train_jas, val_jas))  # (80000, 600)


# 执行t-SNE降维
embedded_data = tsne.fit_transform(all_data)

# 绘制第一组数据的图形
plt.scatter(embedded_data[:60000, 0], embedded_data[:60000, 1], s=0.5, label='Data 1')
# 绘制第二组数据的图形
plt.scatter(embedded_data[60000:, 0], embedded_data[60000:, 1], s=0.5, label='Data 2')

plt.legend()
plt.show()
