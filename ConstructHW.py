import KNN
import Kmeans

def constructHW_knn(X,K_neigs,is_probH):

    H = KNN.construct_H_with_KNN(X,K_neigs,is_probH)

    G = KNN._generate_G_from_H(H)

    return G

def constructHW_kmean(X,clusters):

    H = Kmeans.construct_H_with_Kmeans(X,clusters)

    G = Kmeans._generate_G_from_H(H)

    return G
