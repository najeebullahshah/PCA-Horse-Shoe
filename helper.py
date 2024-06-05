# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:42:27 2024

@author: Najeebullah Shah
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from prinpy.local import CLPCG
from prinpy.glob import NLPCA
from scipy.spatial.distance import euclidean
import os

class Helper():
    """
    
    """
    
    def __init__(self, imgType = '.eps'):
        
        self.imgType = imgType
        
        directory = "./figures"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        dX_human = pd.read_csv('./datasets/embryonic_data_490genes.txt',  sep='\t', index_col=0)
        dY_human = pd.read_csv('./datasets/labels.txt')
        dX_human = dX_human.transpose()
        transformer = FunctionTransformer(np.log1p, validate=True)
        self.X_human = transformer.transform(dX_human)
        self.y_human = dY_human.values.ravel()
        
        
        dX_liver = pd.read_csv('./datasets/dataEryth2100x2500.csv', index_col=0)
        dY_liver = pd.read_csv('./datasets/dataY_2100x2500.csv', index_col=0)
        self.X_liver = dX_liver.values
        self.y_liver = dY_liver.values.ravel()
        
        dX_reduced_mouse = pd.read_csv("./datasets/deng_horse_shoe.csv", sep=",",index_col=0)
        
        self.X_reduced_human = PCA(n_components = 4).fit_transform(self.X_human)
        self.X_reduced_liver = PCA(n_components = 2).fit_transform(self.X_liver)
        self.X_reduced_mouse = dX_reduced_mouse.values
        
        self.simulatedDatasetA = self.generateSimulatedDatasetA()
        self.X_reduced_DSA = PCA(n_components = 2).fit_transform(self.simulatedDatasetA.T)
        
        self.simulatedDatasetBlock = self.generateBlockSimulatedDataset()
        
        self.simulatedDatasetB = self.generatedSimulateDatasetB()
        self.X_reduced_DSB =  PCA(n_components = 2).fit_transform(self.simulatedDatasetBMatrix)
        
        self.simulatedDatasetANoSat1 = self.generatedSimulatedDatasetANoSaturation1()
        self.X_reduced_DSA_noSat1 = PCA(n_components = 3).fit_transform(self.simulatedDatasetANoSat1.T)
        
        self.simulatedDatasetANoSat2 = self.generatedSimulatedDatasetANoSaturation2()
        self.X_reduced_DSA_noSat2 = PCA(n_components = 3).fit_transform(self.simulatedDatasetANoSat2.T)
        
        
    def generateSimulatedDatasetA(self):
        n = 15  #otus 
        band = 5 #band size
        band2 = 3
        gap1 = 7
        p = n - band + 1 #samples
        min_gene_count = 0
        max_gene_count = 100
        y = [1./band]*band +  [0]*(n-band)
        table = self.shift(y,p-1)
        table = np.column_stack(table)
        u, k, v = np.linalg.svd(table.T)
        return table
    
    def generateBlockSimulatedDataset(self):
        n_blocks = 4
        block_size = 25
        min_count = 0
        max_count = 10
        np.random.seed(1)
        block_sub_matrix = np.block([np.zeros((block_size, block_size))]*n_blocks)
        block_sub_matrix[:, 0:block_size] = np.random.randint(min_count, max_count, size = (block_size, block_size))
        block_matrix = block_sub_matrix
        for j in range(block_size, n_blocks*block_size, block_size):
            block_sub_matrix = np.block([np.zeros((block_size, block_size))]*n_blocks)
            block_sub_matrix[:,j:(j+block_size)] = np.random.randint(min_count, max_count, size = (block_size,block_size))
            block_matrix = np.vstack((block_matrix, block_sub_matrix))
        block_df = pd.DataFrame(block_matrix, columns = ['V' + str(i) for i in range(block_matrix.shape[1])], 
                                index = ['S' + str(i) for i in range(block_matrix.shape[0])])
        self.simulatedBlockDatasetMatrix = block_matrix
        return block_df
    
    def generatedSimulateDatasetB(self):
        n_blocks = 3
        block_size = 150
        min_count = 0
        max_count = 100

        np.random.seed(1)

        #Initialization
        block_sub_matrix1 = np.block([np.zeros((block_size, block_size))]*n_blocks)
        block_sub_matrix1[:, 0:block_size] = np.random.randint(min_count, max_count, size = (block_size, block_size))
        block_sub_matrix1[:, block_size:(2*block_size)] = np.tril(np.random.randint(min_count, max_count, 
                                                                                   size = (block_size, block_size)))

        block_sub_matrix2 = np.block([np.zeros((block_size, block_size))]*n_blocks)
        block_sub_matrix2[:, block_size:(2*block_size)] = np.random.randint(min_count, max_count, 
                                                                            size = (block_size, block_size))
        block_sub_matrix2[:, 0:block_size] = np.triu(np.random.randint(min_count, max_count, 
                                                                                   size = (block_size, block_size)))
        block_sub_matrix2[:, (2*block_size):(3*block_size)] = np.tril(np.random.randint(min_count, max_count, 
                                                                                        size = (block_size, block_size)))


        block_sub_matrix3 = np.block([np.zeros((block_size, block_size))]*n_blocks)
        block_sub_matrix3[:, (2*block_size):(3*block_size)] = np.random.randint(min_count, max_count, 
                                                                                size = (block_size, block_size))
        block_sub_matrix3[:, block_size:(2*block_size)] = np.triu(np.random.randint(min_count, max_count, 
                                                                                   size = (block_size, block_size)))


        block_matrix = np.vstack((block_sub_matrix1, block_sub_matrix2, block_sub_matrix3))

            
        block_df = pd.DataFrame(block_matrix, columns = ['V' + str(i) for i in range(block_matrix.shape[1])], 
                                index = ['S' + str(i) for i in range(block_matrix.shape[0])])
        block_df
        
        
        X = block_matrix
        X = np.log10(X + 1)
        X = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
        
        self.simulatedDatasetBMatrix = X
        return block_df
    
    def generatedSimulatedDatasetANoSaturation1(self):
        n = 19  #otus 
        band = 10 #band size
        band2 = 3
        gap1 = 7
        p = n - band + 1 #samples
        min_gene_count = 0
        max_gene_count = 100
        y = [1./band]*band +  [0]*(n-band)
        table = self.shift(y,p-1)
        table = np.column_stack(table)
        u, k, v = np.linalg.svd(table.T)
        return table
        
    def generatedSimulatedDatasetANoSaturation2(self):
        n = 39  #otus 
        band = 20 #band size
        band2 = 3
        gap1 = 7
        p = n - band + 1 #samples
        min_gene_count = 0
        max_gene_count = 100
        y = [1./band]*band +  [0]*(n-band)
        table = self.shift(y,p-1)
        table = np.column_stack(table)
        u, k, v = np.linalg.svd(table.T)
        return table
        
    def normsquare(self,x):
        return np.sum([p**2 for p in x])

    def distanceCal(self, x1,x2):
        return np.sqrt(self.normsquare(x1-x2))

    def shift1(self, l):
        newlist = [0] * len(l)
        for i in range(1,len(l)):
            newlist[i] = l[i-1]
        return newlist

    def shift(self, l,n):
        sl = l
        table = [l]
        if n == 0:
            return table
        else:
            for k in range(n):
                sl = self.shift1(sl)
                table.append(sl)
            return table

    def join(self, point1,point2):
        x0, y0 = point1
        x1, y1 = point2
        line_ans = lambda x: x* (y1-y0) / (x1-x0) + (y1 - y0) 
        
    def generateSubplotsFigure1(self):
        # Set the figure size (width, height) in inches
        fig_width, fig_height = 8, 6  # Example size, adjust as needed

        # Set global font size
        font_size = 12  # Example font size, adjust as needed
        plt.rcParams.update({'font.size': font_size})

        # Create the first figure
        fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
        ax1.scatter(self.X_reduced_human[:,0], self.X_reduced_human[:,1], s = 15)
        ax1.set_xlabel('PCA1')
        ax1.set_ylabel('PCA2')
        fig1.savefig('./figures/unlabelled_human_pca'+self.imgType, dpi=600)
        plt.close(fig1)

        # Create the second figure with the same settings
        fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
        ax2.scatter(self.X_reduced_mouse[:,0],self. X_reduced_mouse[:,1], s = 15)
        ax2.set_xlabel('PCA1')
        ax2.set_ylabel('PCA2')
        #fig2.savefig('figure2.png', dpi=600)
        fig2.savefig('./figures/unlabelled_mouse_pca'+self.imgType, dpi=600)
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(fig_width, fig_height))
        ax3.scatter(self.X_reduced_liver[:,0], self.X_reduced_liver[:,1], s = 15)
        ax3.set_xlabel('PCA1')
        ax3.set_ylabel('PCA2')
        #fig3.savefig('figure3.png', dpi=600)
        fig3.savefig('./figures/unlabelled_liver_pca'+self.imgType, dpi=600)
        plt.close(fig3)
        
    def generateSubplotsFigure2(self):
        plt.imshow(self.simulatedDatasetA, cmap=plt.cm.Reds)
        plt.xlabel('Cells',fontsize=15)
        plt.ylabel('Genes',fontsize=15, rotation=90)
        fig = plt.savefig('./figures/data_matrix_dist_band_datasetA'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)

        plt.scatter(self.X_reduced_DSA[:, 0], self.X_reduced_DSA[:, 1], s = 50)
        plt.xlabel("PCA1", fontsize = 10); plt.ylabel("PCA2", fontsize = 10)
        fig = plt.savefig('./figures/pca_simulated_datasetA'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        
        cl = CLPCG()
        fig, ax = plt.subplots()
        cl.fit(self.X_reduced_DSA[:, 0], self.X_reduced_DSA[:, 1], e_max = .075)
        fig, ax = plt.subplots()
        ax.scatter(self.X_reduced_DSA[:,0], self.X_reduced_DSA[:,1], s = 50)
        cl.plot(ax)
        pts = cl.fit_points   # fitted points with PC that spline is passed through
        ax.scatter(pts[:,0], pts[:,1], s = 80, c = 'green')
        plt.xlabel("PCA1", fontsize = 10); plt.ylabel("PCA2", fontsize = 10)
        fig = plt.savefig('./figures/best_fit_pc_with_datasetA'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        p = 11
        endpoint_distance = [self.distanceCal(self.simulatedDatasetA[:,0],self.simulatedDatasetA[:,x]) for x in range(p)]
        plt.scatter(range(p),endpoint_distance, s = 50)
        plt.scatter(0,endpoint_distance[0], s = 80, c = 'black')
        for i, txt in enumerate(range(p)):  #POINTS = SAMPLES
            if(i==0):
                plt.annotate(txt, (i+.3,endpoint_distance[i]-0.01),fontsize=10)
            elif(i>=10):
                plt.annotate(txt, (i-0.1,endpoint_distance[i]-0.03),fontsize=10)
            else:
                plt.annotate(txt, (i+.22,endpoint_distance[i]-0.03),fontsize=10)
        plt.ylabel('Distance to Sample "0"',fontsize=12)
        plt.xlabel('Ordered Sample Indexes',fontsize=12)
        plt.plot(range(p),[endpoint_distance[p-1] for x in range(p)], c='orange', linestyle='--')
        fig = plt.savefig('./figures/saturation_dist_simulated_datasetA'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        plt.plot([x/p for x in range(p)], self.X_reduced_DSA[:,0])
        plt.ylabel('PCA1 Values',fontsize=12)
        plt.xlabel('Normalized Ordered Sample Indexes',fontsize=12)
        fig = plt.savefig('./figures/simulated_dataset_a_eigenfunction1'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        plt.plot([x/p for x in range(p)], self.X_reduced_DSA[:,1])
        plt.ylabel('PCA2 Values',fontsize=12)
        plt.xlabel('Normalized Ordered Sample Indexes',fontsize=12)
        fig = plt.savefig('./figures/simulated_dataset_a_eigenfunction2'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)

        plt.plot(self.X_reduced_DSA[:,0], self.X_reduced_DSA[:,1])
        plt.ylabel('PCA1',fontsize=12)
        plt.xlabel('PCA2',fontsize=12)
        fig = plt.savefig('./figures/simulated_dataset_a_horse_shoe'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
    def generateBlockDatasetsFigures(self):
        plt.imshow(self.simulatedDatasetBlock, cmap=plt.cm.Reds)
        plt.xlabel('Cells',fontsize=12)
        plt.ylabel('Genes',fontsize=12, rotation=90)
        fig = plt.savefig('./figures/block_matrix_four_cells'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        np.random.seed(1)
        X = self.simulatedBlockDatasetMatrix
        X = np.log10(X + 1)
        X = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
        X_reduced = PCA(n_components = 2).fit_transform(X)
        plt.scatter(X_reduced[0:25, 0], X_reduced[0:25, 1], label='Cell A', s = 50, c = 'b')
        plt.scatter(X_reduced[25:50, 0], X_reduced[25:50, 1],  label='Cell B', s = 50, c = 'r')
        plt.scatter(X_reduced[50:75, 0], X_reduced[50:75, 1], label='Cell C', s = 50, c = 'g')
        plt.scatter(X_reduced[75:100, 0], X_reduced[75:100, 1], label='Cell D', s = 50, c = 'pink')
        plt.xlabel("PCA1", fontsize = 10); plt.ylabel("PCA2", fontsize = 10)
        plt.legend()
        fig = plt.savefig('./figures/pca_four_cells'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
    def generateSupplementaryFigure3(self):
        plt.imshow(self.simulatedDatasetB, cmap=plt.cm.Reds)
        plt.xlabel('Cells',fontsize=12)
        plt.ylabel('Genes',fontsize=12, rotation=90)
        fig = plt.savefig('./figures/data_matrix_simulated_datasetB'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        plt.scatter(self.X_reduced_DSB[:, 0], self.X_reduced_DSB[:, 1], s = 5)
        plt.xlabel("PCA1", fontsize = 12); plt.ylabel("PCA2", fontsize = 12)
        fig = plt.savefig('./figures/pca2d_simulated_datasetB'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        cl = CLPCG()
        fig, ax = plt.subplots()
        cl.fit(self.X_reduced_DSB[:, 0], self.X_reduced_DSB[:, 1], e_max = 1.5)
        fig, ax = plt.subplots()
        ax.scatter(self.X_reduced_DSB[:,0], self.X_reduced_DSB[:,1], s = 5)
        cl.plot(ax)
        pts = cl.fit_points   # fitted points with PC that spline is passed through
        ax.scatter(pts[:,0], pts[:,1], s = 80, c = 'green')
        plt.xlabel("PCA1", fontsize = 12); plt.ylabel("PCA2", fontsize = 12)
        fig = plt.savefig('./figures/best_fit_pc_with_datasetB'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        tot_sam = 450
        endpoint_distance = [self.distanceCal(self.simulatedDatasetBMatrix[:,0],self.simulatedDatasetBMatrix[:,x]) for x in range(tot_sam)]
        sort_indx = np.argsort(np.array(endpoint_distance))
        endpoint_sorted_distances = [endpoint_distance[x] for x in sort_indx]
        plt.scatter(range(tot_sam),endpoint_sorted_distances, s = 5)
        plt.scatter(0,endpoint_sorted_distances[0], s = 20, c = 'black')
        plt.annotate("0", (0+.8,endpoint_sorted_distances[0]+0.3),fontsize=10)
        plt.ylabel('Distance to Sample "0"',fontsize=12)
        plt.xlabel('Ordered Sample Indexes',fontsize=12)
        plt.plot(range(tot_sam),[39.0 for x in range(tot_sam)], c='orange', linestyle='--')
        fig = plt.savefig('./figures/saturation_dist_simulated_datasetB'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        plt.plot([x/tot_sam for x in range(tot_sam)], self.X_reduced_DSB[:,0])
        plt.ylabel('PCA1 Values',fontsize=12)
        plt.xlabel('Normalized Ordered Sample Indexes',fontsize=12)
        fig = plt.savefig('./figures/simulated_dataset_b_eigenfunction1'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)

        plt.plot([x/tot_sam for x in range(tot_sam)], self.X_reduced_DSB[:,1])
        plt.ylabel('PCA2 Values',fontsize=12)
        plt.xlabel('Normalized Ordered Sample Indexes',fontsize=12)
        fig = plt.savefig('./figures/simulated_dataset_b_eigenfunction2'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)

        plt.plot(self.X_reduced_DSB[:,0], self.X_reduced_DSB[:,1])
        plt.ylabel('PCA1',fontsize=12)
        plt.xlabel('PCA2',fontsize=12)
        fig = plt.savefig('./figures/simulated_dataset_b_horse_shoe'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
    def generateSubplotsFigure3(self):
        
        dX_all_sorted_human = pd.read_csv('./datasets/sorted_human490_indexes.csv', index_col=0) 
        indexes = dX_all_sorted_human.index
        X_all_sorted_human = dX_all_sorted_human.values
        plt.imshow(X_all_sorted_human.T, cmap=plt.cm.Reds)
        plt.xlabel('Cells',fontsize=7)
        plt.ylabel('Genes',fontsize=7, rotation=90)
        plt.tick_params(axis='both', which='major', labelsize=6)
        fig = plt.savefig('./figures/data_matrix_sorted_human'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        curve_pts = np.genfromtxt('./datasets/curve_pts_1.csv', delimiter=',')
        proj = np.genfromtxt('./datasets/proj_1.csv', delimiter=',')
        data_new = np.genfromtxt('./datasets/data_new_1.csv', delimiter=',')
        plt.scatter(data_new[:,0], data_new[:,1], s = 5)
        plt.plot(curve_pts[:,0], 
                 curve_pts[:,1], 
                 color = 'black',
                 linewidth = '2.5')
        plt.xlabel('PCA1',fontsize=12)
        plt.ylabel('PCA2',fontsize=12, rotation=90)
        fig = plt.savefig('./figures/best_fit_nlpca_pc_with_human'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        endpoint_distance = [self.distanceCal(X_all_sorted_human[0,:],row) for row in X_all_sorted_human]
        sort_indx = np.argsort(np.array(endpoint_distance))
        y_all_sorted = np.array([int(ind[1]) for ind in dX_all_sorted_human.index])
        plt.scatter(range(1529), endpoint_distance, 
                         c=y_all_sorted, cmap=plt.cm.Set1, alpha=0.5, s = 5)
        plt.scatter(0,endpoint_distance[0], s = 50, c = 'black')
        plt.annotate("0", (15.0,endpoint_distance[0]),fontsize=10)
        plt.ylabel('Distance to Sample "0"',fontsize=15)
        plt.xlabel('Niche Sorted Samples Indexes',fontsize=15)
        plt.plot(range(1529),[90.0 for x in range(1529)], c='black', linestyle='--')
        fig = plt.savefig('./figures/saturation_dist_human'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        firstItem = self.X_reduced_human[1520,:]
        distances = np.zeros(1529)
        indx = 0
        for row in self.X_reduced_human:
            distances[indx] = euclidean(firstItem, row)
            indx = indx+1
        distances_sorted = distances[np.argsort(distances)]
        X_reduced_human_sorted = self.X_reduced_human[np.argsort(distances),:]
        plt.plot([x/1529 for x in range(1529)], X_reduced_human_sorted[:,0])
        plt.xlabel('Niche Sorted Samples Indexes',fontsize=15)
        plt.ylabel('PCA1 values',fontsize=15)
        fig = plt.savefig('./figures/human_dataset_eigenfunction1'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)

        plt.plot([x/1529 for x in range(1529)], X_reduced_human_sorted[:,1])
        plt.xlabel('Niche Sorted Samples Indexes',fontsize=15)
        plt.ylabel('PCA2 values',fontsize=15)
        fig = plt.savefig('./figures/human_dataset_eigenfunction2'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
    def generateSubplotsFigure4(self):
        dX_all_sorted_mouse = pd.read_csv('./datasets/data_deng_sorted.csv', index_col=0) 
        self.X_all_sorted_mouse = dX_all_sorted_mouse.values
        plt.imshow(self.X_all_sorted_mouse.T, cmap=plt.cm.Reds)
        plt.xlabel('Cells',fontsize=9)
        plt.ylabel('Genes',fontsize=9, rotation=90)
        plt.tick_params(axis='both', which='major', labelsize=4)
        fig = plt.savefig('./figures/data_matrix_sorted_mouse'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        
        # create solver
        pca = NLPCA()
        data_new_mouse = pca.preprocess( [self.X_reduced_mouse[:,0],self.X_reduced_mouse[:,1]] )
        pca.fit(data_new_mouse, epochs = 300, nodes = 15, lr = .01, verbose = 0)
        proj_mouse, curve_pts_mouse = pca.project(data_new_mouse)
        plt.scatter(data_new_mouse[:,0], data_new_mouse[:,1], s = 5)
        plt.plot(curve_pts_mouse[:,0], 
                 curve_pts_mouse[:,1], 
                 color = 'black',
                 linewidth = '2.5')
        plt.xlabel('PCA1',fontsize=15)
        plt.ylabel('PCA2',fontsize=15, rotation=90)
        fig = plt.savefig('./figures/best_fit_nlpca_pc_with_mouse'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        
        indexes = pd.read_csv('./datasets/data_deng_sorted.csv')["Unnamed: 0"].values
        dataY = pd.read_csv('./datasets/data_deng_y.csv', index_col=0).values.ravel()
        endpoint_distance = [self.distanceCal(self.X_all_sorted_mouse[0,:],row) for row in self.X_all_sorted_mouse]
        y_all_sorted = [dataY[int(ind)] for ind in indexes]
        plt.scatter(range(262), endpoint_distance, 
                         c=y_all_sorted, cmap=plt.cm.Set1, alpha=0.5, s = 5)
        plt.scatter(0,endpoint_distance[0], s = 50, c = 'black')
        plt.annotate("0", (15.0,endpoint_distance[0]),fontsize=10)
        plt.ylabel('Distance to Sample "0"',fontsize=15)
        plt.xlabel('Niche Sorted Samples Indexes',fontsize=15)
        plt.plot(range(262),[45.0 for x in range(262)], c='black', linestyle='--')
        plt.rcParams['axes.grid'] = False
        fig = plt.savefig('./figures/saturation_dist_mouse'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        firstItem = self.X_reduced_mouse[14,:]
        distances = np.zeros(262)
        indx = 0
        for row in self.X_reduced_mouse:
            distances[indx] = euclidean(firstItem, row)
            indx = indx+1
        distances_sorted = distances[np.argsort(distances)]
        X_reduced_mouse_sorted = self.X_reduced_mouse[np.argsort(distances),:]
        plt.plot([x/262 for x in range(262)], X_reduced_mouse_sorted[:,0])
        plt.xlabel('Niche Sorted Samples Indexes',fontsize=15)
        plt.ylabel('PCA1 values',fontsize=15)
        fig = plt.savefig('./figures/mouse_dataset_eigenfunction1'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)

        plt.plot([x/262 for x in range(262)], X_reduced_mouse_sorted[:,1])
        plt.xlabel('Niche Sorted Samples Indexes',fontsize=15)
        plt.ylabel('PCA2 values',fontsize=15)
        fig = plt.savefig('./figures/mouse_dataset_eigenfunction2'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        
    def generateSubplotsFigure5(self):
        dX_all_sorted_liver = pd.read_csv('./datasets/sorted_liver_tumor_2100x2500.csv', index_col=0) 
        X_all_sorted_liver_brighter = dX_all_sorted_liver.values
        X_all_sorted_liver_brighter = X_all_sorted_liver_brighter + 3.0
        X_all_sorted_liver_brighter[X_all_sorted_liver_brighter <= 3.0] = 0.0
        plt.imshow(X_all_sorted_liver_brighter.T, cmap=plt.cm.Reds)
        plt.xlabel('Cells',fontsize=15)
        plt.ylabel('Genes',fontsize=15, rotation=90)
        fig = plt.savefig('./figures/data_matrix_sorted_liver'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        # create solver
        pca = NLPCA()
        data_new_liver = pca.preprocess( [self.X_reduced_liver[:,0],self.X_reduced_liver[:,1]] )
        pca.fit(data_new_liver, epochs = 300, nodes = 15, lr = .01, verbose = 0)
        proj_liver, curve_pts_liver = pca.project(data_new_liver)
        plt.scatter(data_new_liver[:,0], data_new_liver[:,1], s = 5)
        plt.plot(curve_pts_liver[:,0], 
                 curve_pts_liver[:,1], 
                 color = 'black',
                 linewidth = '2.5')
        plt.xlabel('PCA1',fontsize=15)
        plt.ylabel('PCA2',fontsize=15, rotation=90)
        fig = plt.savefig('./figures/best_fit_nlpca_pc_with_liver'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        indexes = pd.read_csv('./datasets/sorted_liver_tumor_2100x2500.csv')["Unnamed: 0"].values
        dataY = pd.read_csv('./datasets/dataY_2100x2500.csv', index_col=0).values.ravel()
        X_all_sorted_liver = dX_all_sorted_liver.values
        endpoint_distance = [self.distanceCal(X_all_sorted_liver[0,:],row) for row in X_all_sorted_liver]
        y_all_sorted = [dataY[int(ind)] for ind in indexes]
        plt.scatter(range(2500), endpoint_distance, 
                         c=y_all_sorted, cmap=plt.cm.Set1, alpha=0.5, s = 5)
        plt.scatter(0,endpoint_distance[0], s = 50, c = 'black')
        plt.annotate("0", (15.0,endpoint_distance[0]),fontsize=10)
        plt.ylabel('Distance to Sample "0"',fontsize=15)
        plt.xlabel('Niche Sorted Samples Indexes',fontsize=15)
        plt.plot(range(2500),[14.0 for x in range(2500)], c='black', linestyle='--')
        plt.rcParams['axes.grid'] = False
        fig = plt.savefig('./figures/saturation_dist_liver'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        firstItem = self.X_reduced_liver[446,:]
        distances = np.zeros(2500)
        indx = 0
        for row in self.X_reduced_liver:
            distances[indx] = euclidean(firstItem, row)
            indx = indx+1
        distances_sorted = distances[np.argsort(distances)]
        X_reduced_liver_sorted = self.X_reduced_liver[np.argsort(distances),:]
        plt.plot([x/2500 for x in range(2500)], X_reduced_liver_sorted[:,0])
        plt.xlabel('Niche Sorted Samples Indexes',fontsize=15)
        plt.ylabel('PCA1 values',fontsize=15)
        fig = plt.savefig('./figures/liver_dataset_eigenfunction1'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)

        plt.plot([x/2500 for x in range(2500)], X_reduced_liver_sorted[:,1])
        plt.xlabel('Niche Sorted Samples Indexes',fontsize=15)
        plt.ylabel('PCA2 values',fontsize=15)
        fig = plt.savefig('./figures/liver_dataset_eigenfunction2'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
    def generateSupplementaryFigure4(self):
        i = 0
        distDictionary = {}
        for sampleI in self.simulatedDatasetA.T:
            j = 0
            for sampleJ in self.simulatedDatasetA.T:
                distnce = euclidean(sampleI, sampleJ)
                #print(np.round(distnce,2))
                j = j+ 1
            i = i + 1
        i = 0
        ySample0 = np.zeros(11)
        sample0 = self.simulatedDatasetA.T[0,:]
        for sampleI in self.simulatedDatasetA.T:
            ySample0[i] = euclidean(sampleI, sample0)
            i = i + 1
        plt.plot([x/11 for x in range(11)], ySample0)
        plt.xlabel('Normalized Samples Indexes',fontsize=10)
        plt.ylabel('Euclidean Distance to Sample "0"',fontsize=10, rotation=90)
        fig = plt.savefig('./figures/dataset_a_exp_distances'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
            
        y = [1-np.exp(-(x)/(x+1)) for x in range(10)]
        plt.plot([x/10 for x in range(10)], y)
        plt.xlabel('Normalized "n"',fontsize=10)
        plt.ylabel('Values of Exponential Function',fontsize=10, rotation=90)
        fig = plt.savefig('./figures/exp_distances'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
    def generateSubplotsFigure6(self):
        im = plt.imshow(self.simulatedDatasetANoSat1, cmap=plt.cm.Reds)
        plt.xlabel('Cells',fontsize=15)
        plt.ylabel('Genes',fontsize=15, rotation=90)
        fig = plt.savefig('./figures/data_matrix_dist_band_datasetA_noSat'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        plt.scatter(self.X_reduced_DSA_noSat1[:, 0], self.X_reduced_DSA_noSat1[:, 1], s = 50)
        plt.scatter(self.X_reduced_DSA_noSat1[0, 0], self.X_reduced_DSA_noSat1[0, 1], s = 80, c = 'black')
        plt.xlabel("PCA1", fontsize = 15); plt.ylabel("PCA2", fontsize = 15)
        p = self.simulatedDatasetANoSat1.T.shape[0]
        for i, txt in enumerate(range(p)):
            plt.annotate(txt, (self.X_reduced_DSA_noSat1[i, 0]+0.006,self.X_reduced_DSA_noSat1[i, 1]), fontsize=12)
        fig = plt.savefig('./figures/pca_dist_simulated_datasetA_noSat'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        endpoint_distance = [self.distanceCal(self.simulatedDatasetANoSat1[:,0],self.simulatedDatasetANoSat1[:,x]) for x in range(p)]
        plt.scatter(range(p),endpoint_distance, s = 50)
        plt.scatter(0,endpoint_distance[0], s = 80, c = 'black')
        for i, txt in enumerate(range(p)):  #POINTS = SAMPLES
            if(i==0):
                plt.annotate(txt, (i+.3,endpoint_distance[i]-0.01),fontsize=10)
            elif(i>=10):
                plt.annotate(txt, (i-0.1,endpoint_distance[i]-0.03),fontsize=10)
            else:
                plt.annotate(txt, (i+.22,endpoint_distance[i]-0.03),fontsize=10)
        plt.ylabel('Distance to Sample "0"',fontsize=15)
        plt.xlabel('Ordered Sample Indexes',fontsize=15)
        plt.plot(range(p),[endpoint_distance[p-1] for x in range(p)], c='orange', linestyle='--')
        fig = plt.savefig('./figures/saturation_dist_simulated_datasetA_noSat'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
    def generateSupplementaryFigure5(self):
        cl = CLPCG()
        cl.fit(self.X_reduced_human[:,0], self.X_reduced_human[:,1], e_max = 5.9)
        fig, ax = plt.subplots()
        cl.plot(ax)     # .plot will display the fit curve.
        plt.xlabel('Response Curve Variable 1',fontsize=12)
        plt.ylabel('Response Curve Variable 2',fontsize=12, rotation=90)
        fig = plt.savefig('./figures/best_fit_clpcg_pc_human'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)


        fig, ax = plt.subplots()
        ax.scatter(self.X_reduced_human[:,0], self.X_reduced_human[:,1], s = 5)
        cl.plot(ax)     # .plot will display the fit curve.                # you can optionally pass in a matplotlib ax
        pts = cl.fit_points   # fitted points with PC that spline is passed through
        ax.scatter(pts[:,0], pts[:,1], s = 80, c = 'green')
        cl.plot(ax)
        plt.xlabel('PCA1',fontsize=12)
        plt.ylabel('PCA2',fontsize=12)
        fig = plt.savefig('./figures/best_fit_clpcg_pc_with_human'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
    def generateSupplementaryFigure6(self):
        im = plt.imshow(self.simulatedDatasetANoSat2, cmap=plt.cm.Reds)
        plt.xlabel('Cells',fontsize=15)
        plt.ylabel('Genes',fontsize=15, rotation=90)
        fig = plt.savefig('./figures/data_matrix_dist_band_datasetA_noSat20B'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)

        plt.scatter(self.X_reduced_DSA_noSat2[:, 0], self.X_reduced_DSA_noSat2[:, 1], s = 50)
        plt.scatter(self.X_reduced_DSA_noSat2[0, 0], self.X_reduced_DSA_noSat2[0, 1], s = 80, c = 'black')
        plt.xlabel("PCA1", fontsize = 15); plt.ylabel("PCA2", fontsize = 15)
        p = 20
        for i, txt in enumerate(range(p)):
            plt.annotate(txt, (self.X_reduced_DSA_noSat2[i, 0]+0.006,self.X_reduced_DSA_noSat2[i, 1]), fontsize=12)
        fig = plt.savefig('./figures/pca_dist_simulated_datasetA_noSat20B'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)

        endpoint_distance = [self.distanceCal(self.simulatedDatasetANoSat2[:,0],self.simulatedDatasetANoSat2[:,x]) for x in range(p)]
        plt.scatter(range(p),endpoint_distance, s = 50)
        plt.scatter(0,endpoint_distance[0], s = 80, c = 'black')
        for i, txt in enumerate(range(p)):  #POINTS = SAMPLES
            if(i==0):
                plt.annotate(txt, (i+.3,endpoint_distance[i]-0.01),fontsize=10)
            else:
                plt.annotate(txt, (i+.1,endpoint_distance[i]-0.02),fontsize=10)
        plt.ylabel('Distance to Sample "0"',fontsize=15)
        plt.xlabel('Ordered Sample Indexes',fontsize=15)
        plt.plot(range(p),[endpoint_distance[p-1] for x in range(p)], c='orange', linestyle='--')
        fig = plt.savefig('./figures/saturation_dist_simulated_datasetA_noSat20B'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
    def generateSupplementaryFigure7(self):
        y1 = [-np.sin(3.67*((x/50)-1/2)) for x in range(50)]
        plt.plot([x/50 for x in range(50)], y1)
        plt.ylabel('Eigen vector 1 values',fontsize=15)
        plt.xlabel('Normalized Sample Indexes',fontsize=15)
        fig = plt.savefig('./figures/model_eigenfunction1'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)


        y2 = [np.cos(6.39*((x/50)-1/2)) for x in range(50)]
        plt.plot([x/50 for x in range(50)], y2)
        plt.ylabel('Eigen vector 2 values',fontsize=15)
        plt.xlabel('Normalized Sample Indexes',fontsize=15)
        fig = plt.savefig('./figures/model_eigenfunction2'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)


        plt.plot([np.sqrt(0.07)*mm for mm in y1], [np.sqrt(0.02)*mm for mm in y2])
        plt.xlabel('Eigen vector 1',fontsize=15)
        plt.ylabel('Eigen vector 2',fontsize=15)
        fig = plt.savefig('./figures/model_horse_shoe'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
    def generateSupplementaryFigure8(self):
        plt.plot([x/10 for x in range(10)], self.X_reduced_DSA_noSat1[:,0])
        plt.xlabel('Normalized Sample Indexes',fontsize=15)
        plt.ylabel('PCA1 values',fontsize=15)
        fig = plt.savefig('./figures/noSatA_eigenfunction1'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)

        plt.plot([x/10 for x in range(10)], self.X_reduced_DSA_noSat1[:,1])
        plt.xlabel('Normalized Sample Indexes',fontsize=15)
        plt.ylabel('PCA2 values',fontsize=15)
        fig = plt.savefig('./figures/noSatA_eigenfunction2'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)

        plt.plot(self.X_reduced_DSA_noSat1[:,0], self.X_reduced_DSA_noSat1[:,1])
        plt.xlabel('PCA1',fontsize=15)
        plt.ylabel('PCA2',fontsize=15)
        fig = plt.savefig('./figures/noSatA_horse_shoe'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        plt.plot([x/20 for x in range(20)], self.X_reduced_DSA_noSat2[:,0])
        plt.xlabel('Normalized Sample Indexes',fontsize=15)
        plt.ylabel('PCA1 values',fontsize=15)
        fig = plt.savefig('./figures/noSatB_eigenfunction1'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)

        plt.plot([x/20 for x in range(20)], self.X_reduced_DSA_noSat2[:,1])
        plt.xlabel('Normalized Sample Indexes',fontsize=15)
        plt.ylabel('PCA2 values',fontsize=15)
        fig = plt.savefig('./figures/noSatB_eigenfunction2'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)

        plt.plot(self.X_reduced_DSA_noSat2[:,0], self.X_reduced_DSA_noSat2[:,1])
        plt.xlabel('PCA1',fontsize=15)
        plt.ylabel('PCA2',fontsize=15)
        fig = plt.savefig('./figures/noSatB_horse_shoe'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
    def generateSupplementaryFigure9(self):
        plt.scatter(self.X_reduced_human[:,0], self.X_reduced_human[:,1], s = 5)
        plt.xlabel('PCA1',fontsize=15)
        plt.ylabel('PCA2',fontsize=15)
        fig = plt.savefig('./figures/pca12'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)


        plt.scatter(self.X_reduced_human[:,0], self.X_reduced_human[:,2], s = 5)
        plt.xlabel('PCA1',fontsize=15)
        plt.ylabel('PCA3',fontsize=15)
        fig = plt.savefig('./figures/pca13'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)

        plt.scatter(self.X_reduced_human[:,0], self.X_reduced_human[:,3], s = 5)
        plt.xlabel('PCA1',fontsize=15)
        plt.ylabel('PCA4',fontsize=15)
        fig = plt.savefig('./figures/pca14'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)


        plt.scatter(self.X_reduced_human[:,1], self.X_reduced_human[:,2], s = 5)
        plt.xlabel('PCA2',fontsize=15)
        plt.ylabel('PCA3',fontsize=15)
        fig = plt.savefig('./figures/pca23'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)


        plt.scatter(self.X_reduced_human[:,1], self.X_reduced_human[:,3], s = 5)
        plt.xlabel('PCA2',fontsize=15)
        plt.ylabel('PCA4',fontsize=15)
        fig = plt.savefig('./figures/pca24'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)

        plt.scatter(self.X_reduced_human[:,2], self.X_reduced_human[:,3], s = 5)
        plt.xlabel('PCA3',fontsize=15)
        plt.ylabel('PCA4',fontsize=15)
        fig = plt.savefig('./figures/pca34'+self.imgType, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
    def randOfMatrices(self):
        dX_all_sorted_mouse = pd.read_csv('./datasets/data_deng_sorted.csv', index_col=0) 
        X_all_sorted_mouse = dX_all_sorted_mouse.values
        rank = np.linalg.matrix_rank(self.X_human)
        print("rank of human embryonic dataset:", rank)
        rank = np.linalg.matrix_rank(self.X_liver)
        print("rank of liver haematopoiesis:", rank)
        rank = np.linalg.matrix_rank(X_all_sorted_mouse)
        print(X_all_sorted_mouse.shape)
        print("rank of mouse embryonic:", rank)
        
    def rateZeroEntries(self):
        dX_all_sorted_mouse = pd.read_csv('./datasets/data_deng_sorted.csv', index_col=0) 
        X_all_sorted_mouse = dX_all_sorted_mouse.values
        rate_of_zero_entries = np.mean(self.X_human == 0)
        print("rate of zero entries in human embryonic dataset:", rate_of_zero_entries)
        
        rate_of_zero_entries = np.mean(self.X_liver == 0)
        print("rate of zero entries in liver haematopoiesis dataset:", rate_of_zero_entries)
        
        rate_of_zero_entries = np.mean(X_all_sorted_mouse == 0)
        print("rate of zero entries in mouse embryonic dataset:", rate_of_zero_entries)
            
            
        
        
            