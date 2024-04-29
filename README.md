import pandas as pd
import numpy as np

# Load the MSA dataset
mcdonalds = pd.read_csv("mcdonalds.csv")  # Assuming the dataset is stored in a CSV file

# Display variable names
print(mcdonalds.columns.tolist())

# Display sample size
print(mcdonalds.shape)

# Display first three rows of the data
print(mcdonalds.head(3))

# Extracting columns 1 to 11 and converting "Yes" to 1 and "No" to 0
MD_x = mcdonalds.iloc[:, 0:11].apply(lambda x: (x == "Yes") + 0)

# Calculating column means and rounding to two decimal places
column_means = np.round(MD_x.mean(), 2)

print(column_means)

from sklearn.decomposition import PCA

# Perform PCA
pca = PCA()
MD_pca = pca.fit_transform(MD_x)

# Summarize PCA results
print("Proportion of Variance Explained:")
print(np.round(pca.explained_variance_ratio_, 2))
print("\n")

print("Summary Statistics:")
print(pd.DataFrame(MD_pca).describe())

# Print standard deviations of each principal component
print("Standard deviations (1, .., p={0}):".format(len(pca.explained_variance_)))
print(np.round(pca.explained_variance_ ** 0.5, 1))

# Print rotation matrix
print("Rotation (n x k) = (11 x 11):")
print(pd.DataFrame(pca.components_, columns=MD_x.columns))

import matplotlib.pyplot as plt

# Plot PCA results
plt.scatter(MD_pca[:, 0], MD_pca[:, 1], color='grey')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Plot')
plt.show()

# Assuming you want to plot the projection axes
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector[:2] * 3 * np.sqrt(length)
    plt.plot([0, v[0]], [0, v[1]], '-k', lw=3)
plt.axis('equal')
plt.title('Projection Axes')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

from sklearn.cluster import KMeans

# Set random seed
np.random.seed(1234)

# Initialize variables
best_model = None
best_score = float('inf')

# Perform KMeans clustering with 2 to 8 clusters
for n_clusters in range(2, 9):
    for _ in range(10):  # Number of repetitions
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(MD_x)
        score = kmeans.inertia_
        if score < best_score:
            best_score = score
            best_model = kmeans

# Relabel clusters
labels = best_model.labels_

print(labels)

# Plotting the number of segments against the within-cluster sum of squares
plt.plot(range(2, 9), [KMeans(n_clusters=i).fit(MD_x).inertia_ for i in range(2, 9)], marker='o')
plt.xlabel('Number of segments')
plt.ylabel('Within-cluster sum of squares')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

from sklearn.utils import resample

# Set random seed
np.random.seed(1234)

# Initialize variables
n_boot = 100
boot_results = []

# Bootstrap over 2 to 8 clusters
for n_clusters in range(2, 9):
    cluster_results = []
    for _ in range(n_boot):
        # Bootstrap resampling
        boot_samples = resample(MD_x)
        
        # Perform clustering on bootstrapped sample
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(boot_samples)
        
        # Store clustering results
        cluster_results.append(kmeans.labels_)
    
    # Store bootstrapping results
    boot_results.append(cluster_results)

# Convert to numpy array for further analysis if needed
MD_b28 = np.array(boot_results)

print(MD_b28.shape)  # Shape: (n_clusters, n_boot, n_samples)

from sklearn.metrics import adjusted_rand_score

# Calculate adjusted Rand index for each bootstrap iteration
adjusted_rand_indices = []
for i, n_clusters in enumerate(range(2, 9)):
    cluster_results = MD_b28[i]  # Bootstrap results for current number of clusters
    cluster_adjusted_rands = []
    for labels in cluster_results:
        # Calculate adjusted Rand index for each bootstrapped sample
        true_labels = np.random.choice(labels, len(labels), replace=True)  # Randomize true labels
        adjusted_rand = adjusted_rand_score(true_labels, labels)
        cluster_adjusted_rands.append(adjusted_rand)
    adjusted_rand_indices.append(cluster_adjusted_rands)

# Plot adjusted Rand index against the number of segments
plt.plot(range(2, 9), np.mean(adjusted_rand_indices, axis=1), marker='o')
plt.xlabel('Number of segments')
plt.ylabel('Adjusted Rand index')
plt.title('Bootstrapping Adjusted Rand Index for Optimal Number of Clusters')
plt.show()

# Extract cluster labels for cluster "4"
cluster_4_labels = labels[MD_kmeans == 4]

# Plot histogram
plt.hist(cluster_4_labels, bins=np.arange(3)-0.5, rwidth=0.8)
plt.xlabel('Cluster Assignment')
plt.ylabel('Frequency')
plt.title('Histogram of Cluster Assignments for Cluster 4')
plt.xlim(0, 1)
plt.xticks([0, 1])
plt.show()

# Extract cluster assignments for cluster "4"
cluster_4_labels = labels[MD_kmeans == 4]

# Assign to MD.k4
MD_k4 = cluster_4_labels

from sklearn.metrics import silhouette_score, silhouette_samples

# Compute silhouette widths for each sample in cluster "4"
silhouette_widths = silhouette_samples(MD_x, MD_k4)

# Mean silhouette width for cluster "4"
mean_silhouette_width = silhouette_score(MD_x, MD_k4)

print("Mean silhouette width for cluster 4:", mean_silhouette_width)

# Plot segment stability
plt.plot(range(len(MD_r4)), MD_r4, marker='o')
plt.xlabel('Segment number')
plt.ylabel('Segment stability')
plt.title('Segment Stability for Cluster 4')
plt.ylim(0, 1)
plt.show()

from mixmod import FLXMCmvbinary
from mixmod.mixmodcluster import MixModCluster
import numpy as np

# Set random seed
np.random.seed(1234)

# Create a MixModCluster object
mm = MixModCluster()

# Perform model-based clustering with FLXMCmvbinary model
MD_m28 = mm.fit(MD_x.values, k=range(2, 9), nrep=10, model=FLXMCmvbinary())

# Display the result
print(MD_m28)

# Plot the value of information criteria (AIC, BIC, ICL)
MD_m28.plot_information_criteria()
plt.ylabel('Value of Information Criteria (AIC, BIC, ICL)')
plt.show()

from collections import Counter

# Get cluster assignments for MD.m4
mixture_clusters = MD_m4.labels_

# Count occurrences of each cluster combination
cluster_comparison = Counter(zip(MD_k4, mixture_clusters))

# Display comparison table
print("Comparison of KMeans and Mixture Model Clusters:")
print("KMeans  Mixture")
for kmeans_cluster, mixture_cluster in sorted(cluster_comparison.keys()):
    print(f"{kmeans_cluster:7} {mixture_cluster}")
    print(cluster_comparison[(kmeans_cluster, mixture_cluster)])


from flexmix import flexmix

# Fit flexmix model with FLXMCmvbinary model
MD_m4a = flexmix(MD_x.values, k=2, cluster=MD_k4, model=FLXMCmvbinary())

# Get cluster assignments for MD_m4a
mixture_clusters_flexmix = MD_m4a['cluster']

# Count occurrences of each cluster combination
cluster_comparison_flexmix = Counter(zip(MD_k4, mixture_clusters_flexmix))

# Display comparison table
print("Comparison of KMeans and Flexmix Model Clusters:")
print("KMeans  Flexmix")
for kmeans_cluster, flexmix_cluster in sorted(cluster_comparison_flexmix.keys()):
    print(f"{kmeans_cluster:7} {flexmix_cluster}")
    print(cluster_comparison_flexmix[(kmeans_cluster, flexmix_cluster)])

# Compute log-likelihood for MD_m4a
log_likelihood_m4a = MD_m4a.log_likelihood_

# Compute log-likelihood for MD_m4
log_likelihood_m4 = MD_m4.log_likelihood_

print("Log-likelihood for MD_m4a:", log_likelihood_m4a)
print("Log-likelihood for MD_m4:", log_likelihood_m4)

# Reverse the levels of the "Like" variable
mcdonalds['Like.n'] = 6 - pd.to_numeric(mcdonalds['Like'])

# Display the frequency table of the new variable "Like.n"
like_n_counts = mcdonalds['Like.n'].value_counts().sort_index()
print(like_n_counts)

# Define the formula string
formula_str = "Like.n ~ " + "+".join(mcdonalds.columns[:11])

# Convert the formula string to a formula object
formula_obj = pd.formula(formula_str)

# Display the formula object
print(formula_obj)

from mixmod import FLXMRmv
from mixmod.mixmodcluster import MixModCluster

# Set random seed
np.random.seed(1234)

# Create a MixModCluster object
mm = MixModCluster()

# Define the formula string
formula_str = "Like.n ~ " + "+".join(mcdonalds.columns[:11])

# Perform model-based regression with FLXMRmv model
MD_reg2 = mm.fit(data=mcdonalds, formula=formula_str, k=2, nrep=10, model=FLXMRmv())

# Display the result
print(MD_reg2)

# Refit the model
MD_ref2 = MD_reg2.refit()

# Summary of the refitted model
print(MD_ref2.summary())

# Plot the refitted model with significance
MD_ref2.plot(significance=True)
plt.show()

from scipy.cluster.hierarchy import linkage, dendrogram

# Compute the distance matrix
distance_matrix = np.transpose(MD_x).corr()

# Perform hierarchical clustering
MD_vclust = linkage(distance_matrix, method='average')

# Plot the dendrogram
dendrogram(MD_vclust)
plt.show()


# Reverse the order of hierarchical clustering
hclust_order = np.flip(MD_vclust['leaves'])

# Create the bar chart
plt.bar(range(len(MD_k4)), MD_k4[hclust_order], color='skyblue')

# Show plot
plt.show()

# Plot cluster assignments projected onto the first two principal components
plt.scatter(MD_pca[:, 0], MD_pca[:, 1], c=MD_k4, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Cluster Assignments Projected onto PCA Components')

# Show the projection axes
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector[:2] * 3 * np.sqrt(length)
    plt.plot([0, v[0]], [0, v[1]], '-k', lw=3)

plt.axis('equal')
plt.show()


from statsmodels.graphics.mosaicplot import mosaic

# Create a DataFrame with cluster assignments and the "Like" variable
data = pd.DataFrame({'k4': MD_k4, 'Like': mcdonalds['Like']})

# Create a mosaic plot
mosaic(data, ['k4', 'Like'], title='', axes_label=True)

# Show the plot
plt.xlabel('Segment number')
plt.show()

# Create a mosaic plot
mosaic(data, ['k4', 'Gender'], title='', axes_label=True)

# Show the plot
plt.xlabel('Segment number')
plt.show()


# Create a list to hold the age values for each cluster
age_by_cluster = [mcdonalds['Age'][k4 == cluster] for cluster in range(1, 5)]  # Assuming there are 4 clusters

# Create the box plot
plt.boxplot(age_by_cluster, labels=range(1, 5), showmeans=True, notch=True, patch_artist=True, varwidth=True)

# Add labels and title
plt.xlabel('Cluster (k4)')
plt.ylabel('Age')
plt.title('Box Plot of Age by Cluster')

# Show the plot
plt.show()

import pyR

# Initialize the R interpreter
r = pyR.open()

# Load the partykit library
r("library('partykit')")

# Convert the pandas DataFrame to an R dataframe
r.assign("mcdonalds", mcdonalds)

# Fit the conditional inference tree
r("tree <- ctree(factor(k4 == 3) ~ Like.n + Age + VisitFrequency + Gender, data = mcdonalds)")

# Plot the tree
r("plot(tree)")


# Compute the mean of VisitFrequency for each cluster
visit = mcdonalds.groupby(k4)['VisitFrequency'].mean()

print(visit)

# Compute the mean of "Like.n" for each cluster
like = mcdonalds.groupby(k4)['Like.n'].mean()
print(like)

# Convert Gender to numeric before computing the proportion of females
mcdonalds['Gender_numeric'] = (mcdonalds['Gender'] == 'Female').astype(int)

# Compute the proportion of females for each cluster
female = mcdonalds.groupby(k4)['Gender_numeric'].mean()
print(female)


# Create the scatter plot
plt.scatter(visit, like, s=10 * female, alpha=0.5)

# Add text labels for clusters
for i, txt in enumerate(range(1, 5)):
    plt.annotate(txt, (visit[i], like[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Set axis labels and limits
plt.xlabel('Visit Frequency')
plt.ylabel('Likeability')
plt.xlim(2, 4.5)
plt.ylim(-3, 3)

# Show the plot
plt.show()




    
# McDonald-s-code-translation
