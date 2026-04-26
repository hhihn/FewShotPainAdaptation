import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = np.load("fixed_task_bank_embeddings.npz", allow_pickle=True)
X = data["embeddings"]
labels = data["labels"]
subjects = data["subjects"]

Z = PCA(n_components=2).fit_transform(X)

plt.figure(figsize=(6, 5))
plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="coolwarm", s=20)
plt.title("Embeddings colored by class")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.scatter(Z[:, 0], Z[:, 1], c=subjects, cmap="tab20", s=20)
plt.title("Embeddings colored by subject")
plt.tight_layout()
plt.show()
