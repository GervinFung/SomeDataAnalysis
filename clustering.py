import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def best_k_mean(transformed):
    from sklearn.metrics import silhouette_score
    k_mean, best_score = 0, 0
    for k in range(2, 100):
        k_mean_model = KMeans(n_clusters=k, random_state=0)
        lbl = k_mean_model.fit_predict(transformed)

        score = silhouette_score(transformed, lbl)
        if score > best_score:
            best_score = silhouette_score(transformed, lbl)
            k_mean = k
    return k_mean


def k_mean_clustering(transformed, k_mean):
    plt.figure(figsize=(10, 5))
    kmn = KMeans(n_clusters=k_mean, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmn.fit(transformed)

    labels = kmn.predict(transformed)
    plt.scatter(transformed[:, 0], transformed[:, 1], c=labels)
    centroids = kmn.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='d', s=100, c='red')
    plt.xlabel("Variables affecting Condition of Heart Disease")
    plt.ylabel("Heart Disease")
    plt.show()


def gaussian_mixture(transformed, k_mean):
    plt.figure(figsize=(10, 5))
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=k_mean, max_iter=300, n_init=10, random_state=0)
    gmm.fit(transformed)

    labels = gmm.predict(transformed)
    plt.scatter(transformed[:, 0], transformed[:, 1], c=labels)
    plt.xlabel("Variables affecting Condition of Heart Disease")
    plt.ylabel("Heart Disease")
    plt.show()


def all_clustering(x):
    from sklearn.preprocessing import MinMaxScaler

    scaled_data = MinMaxScaler().fit_transform(x)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca.fit(scaled_data)
    transformed = pca.transform(scaled_data)

    k_mean = best_k_mean(transformed)

    k_mean_clustering(transformed, k_mean)
    gaussian_mixture(transformed, k_mean)
