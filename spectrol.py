# Import thư viện
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# Load dữ liệu Iris (lấy 2 thuộc tính đầu)
iris = load_iris()
X = iris.data[:, :2]

# Khởi tạo mô hình GMM
gmm = GaussianMixture(
    n_components=3,        # số cụm
    covariance_type='full',
    random_state=42
)

# Huấn luyện mô hình
gmm.fit(X)

# Dự đoán nhãn cụm
labels = gmm.predict(X)

# Lấy xác suất mỗi điểm thuộc các cụm
proba = gmm.predict_proba(X)

# In kết quả
print(f"Converged: {gmm.converged_}")
print(f"Iterations: {gmm.n_iter_}")
print(f"Log-likelihood: {gmm.lower_bound_:.2f}")

# Vẽ kết quả phân cụm
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=40)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Gaussian Mixture Model clustering (Iris)")
plt.show()
