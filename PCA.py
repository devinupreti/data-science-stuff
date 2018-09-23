
####################################################
# PCA functions for reduction and plotting
# Author : Devin Upreti 
####################################################


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


####################################################
# To 2 dimensional data
# show_graph is a boolean, true if you wanna plot the data
# data_normalized should not contain target (y value)
# can be obtained by data_normalized = StandardScaler().fit_transform(x)
def pca2d(data_normalized, original_data, show_graph):
    pca = PCA(2)
    components = pca.fit_transform(data_normalized)
    data_components = pd.DataFrame(data = components, columns = ['PC1', 'PC2'])
    data_labeled = pd.concat([data_components, original_data[['label']]], axis = 1)
    
    if show_graph:
        # Plotting the 2 Dimensional Data
        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(1,1,1) 
        class_colors = [ (0,'r'),(1,'k') ]
        classes = [0, 1]

        for element in class_colors:
            plant, color = element
            indexes = data_labeled['label'] == plant
            ax.scatter(data_labeled.loc[indexes, 'PC1'] , data_labeled.loc[indexes, 'PC2'], c = color)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('2d')
        ax.legend(classes)
        plt.show()
        #print(pca.explained_variance_ratio_)
    return data_labeled

####################################################
# To 3 dimensional data
# show_graph is a boolean, true if you wanna plot the data
def pca3d(data_normalized, original_data, show_graph):
    pca = PCA(3)
    components = pca.fit_transform(data_normalized)
    data_components = pd.DataFrame(data = components, columns = ['PC1', 'PC2','PC3'])
    data_labeled = pd.concat([data_components, original_data[['label']]], axis = 1)
    
    if show_graph:
        from mpl_toolkits.mplot3d import Axes3D
        # Plotting the 2 Dimensional Data
        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(111, projection='3d')
        class_colors = [(0,'r'),(1,'k')]
        classes = [0, 1]

        for element in class_colors:
            plant, color = element
            indexes = data_labeled['label'] == plant
            ax.scatter(data_labeled.loc[indexes, 'PC1'] , data_labeled.loc[indexes, 'PC2'],data_labeled.loc[indexes, 'PC3'], c = color)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('3d')
        ax.legend(classes)
        plt.show()
        #print(pca.explained_variance_ratio_)
        #print(pca.explained_variance_ratio_.cumsum())
    return data_labeled
