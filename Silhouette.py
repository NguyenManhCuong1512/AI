# Import thư viện
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Tạo dữ liệu moon
X, y_true = make_moons(
    n_samples=300,
    noise=0.08,
    random_state=42
)

# Chuẩn hóa dữ liệu
X = StandardScaler().fit_transform(X)

# Spectral Clustering
spectral = SpectralClustering(
    n_clusters=2,
    affinity='rbf',
    gamma=1.0,
    random_state=42
)

# Dự đoán nhãn
labels = spectral.fit_predict(X)

# Đánh giá bằng Silhouette Score
score = silhouette_score(X, labels)
print(f"Silhouette: {score:.4f}")

# Vẽ kết quả phân cụm
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=40)
plt.title("Spectral Clustering on Make Moons")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
