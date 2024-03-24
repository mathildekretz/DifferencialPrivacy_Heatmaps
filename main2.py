import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance

import matplotlib.pyplot as plt
import seaborn as sns

from User2 import UserProbabilities, PreProcessing


class CellParameters():
    def __init__(self, px, py, i) -> None:
        self.root_point = np.array([px, py])    #root point of the grid cell
        self.edge_length = 1/(2**i)         #lenght of grid cell
        self.i = i                          #level of grid cell

class AggregatedCell(CellParameters):
    def __init__(self, px, py, i, yi) -> None:
        super().__init__(px, py, i)
        self.yi = yi #value of the aggregated distribution on this cell


class GridParameters():
    def __init__(self, l):
        self.l = l
        self.delta = 2**l
        self.G = None 
        self.Gp = None

        self.construct_G()
    
    def construct_G(self):
        self.G = np.arange(0, 1, 1/ self.delta)
        self.Gp = [CellParameters(x, y, self.delta) for x in self.G for y in self.G]


class PyramidParameters():
    def __init__(self, i, G) -> None:
        self.i = i
        self.G = G
        self.delta = len(G)
        self.C = None
        self.coordinates = None
        self.P = None

        self.construct_C()
        self.construct_P()

    def construct_C(self):
        self.coordinates = np.arange(0, 1, 1/ (2**self.i))
        self.C = [CellParameters(x, y, self.i) for x in self.coordinates for y in self.coordinates]
    
    def construct_P(self):
        self.P = np.zeros((2**(2*self.i), self.delta**2))
        for i,x in enumerate(self.G): #
            ci = np.searchsorted(self.coordinates, x, 'right') -1
            for j,y in enumerate(self.G):
                cj = np.searchsorted(self.coordinates, y, 'right') -1
                self.P[ci*(2**self.i)+cj, i*(self.delta)+j] = 1 


def construct_pyramid_transform(G, l):
    P = PyramidParameters(0, G).P.transpose()
    for i in range(1, l+1):
        Pi = (1/2**i) * PyramidParameters(i, G).P.transpose()
        P = np.concatenate((P, Pi), axis=1)
    return P.transpose()

def construct_y_hat(l, y):
    # print("y", [elm.i for elm in y], [elm.yi for elm in y])
    usi_size = int((1-4**(l+1))/(1-4))
    y_hat = np.zeros((usi_size, 1))
    for cell in y:
        prev_usi = int((1-4**(cell.i))/(1-4)) if cell.i > 0 else 0
        # print("i, px, py, yi, prev", cell.i, cell.root_point[0], cell.root_point[1], cell.yi, prev_usi)
        # print("coord : ", int(prev_usi + cell.root_point[0]*(2**(2*cell.i)) + cell.root_point[1]*(2**cell.i)))
        y_hat[int(prev_usi + cell.root_point[0]*(2**(2*cell.i)) + cell.root_point[1]*(2**cell.i))] = cell.yi
    return y_hat


def get_children(C, i):
    children = []
    for cell in C:
        x_child = [cell.root_point[0], cell.root_point[0] + 1/(2**(i+1))]
        y_child = [cell.root_point[1], cell.root_point[1] + 1/(2**(i+1))]
        child = [CellParameters(x, y, i +1) for x in x_child for y in y_child]
        children += child
    return children


class SparseEMDAgg():
    def __init__(self, user_probabilities, G, l, eps, w) -> None:
        self.user_probabilities = user_probabilities
        self.l = l
        self.G = G
        self.w = w
        self.eps = eps

        self.a_hat = self.dp_sparse_emd_agg()

    def dp_sparse_emd_agg(self):
        """
        inputs: p array of n probability distribution
                eps array of l privay parameters
                w integer 
        
        output: """
        p = self.user_probabilities 

        s = np.sum(p, axis=0).reshape(-1)
        y_list = []
        print("construction of y' ")
        for i in range(self.l +1):
            pyramidi = PyramidParameters(i, self.G)
            mui = np.random.laplace(0, 1/self.eps, 2**(2*i))
            yi = (pyramidi.P @ s + mui) *(1/(2**i))
            yi = yi.reshape((2**i, 2**i))
            y_list.append(yi)
            #print("i, yi: ", i, yi)
        s_hat = np.clip(self.reconstruct(y_list, self.w, s), 0, None)
        a_hat = s_hat / np.sum(s_hat)
        return a_hat.reshape((2**self.l, 2**self.l))

    def reconstruct(self, y, w, s):
        """
        inputs: y array of noisy measurements
                w integer
        outputs: """

        pyramidi = PyramidParameters(0, self.G)
        Si = pyramidi.C
        Si_list = [Si]
        Yi_list = [AggregatedCell(0,0,0,y[0][0,0])] 

        print("construction of Si")
        for i in range(1, self.l + 1):
            Ti = get_children(Si, i)
            coordinate_nb = min(w, len(Ti))
            sorted_yi = np.argsort(y[i], axis=None)
            max_val = sorted_yi[-coordinate_nb:]
            coordinates = np.unravel_index(max_val, y[i].shape)

            Si = [CellParameters(px, py, i) for px,py in zip(coordinates[0]/(2**i), coordinates[1]/(2**i))]
            Yi = [AggregatedCell(px / (2**i), py / (2**i), i, y[i][px, py]) for px, py in zip(coordinates[0], coordinates[1])]
            Si_list.append(Si)
            Yi_list += Yi
            #print("i, coor nb, yi, max val, indices 2D:", i, coordinate_nb, y[i], max_val, coordinates)
            # print("new si:", [elm.root_point for elm in Si], s[elm.yi for elm in Yi])
        P = construct_pyramid_transform(self.G, self.l)
        y_hat = construct_y_hat(self.l, Yi_list)
        # print(y_hat)

        print("EMD solution")
        def l1_norm(y, P, s):
            z = y - P @ s
            return la.norm(z, ord=1)
        
        obj_function = lambda s: l1_norm(y_hat, P, s)
        s_0 = s
        s_hat = minimize(obj_function, s_0, method='BFGS')
        # print(s_hat)

        return s_hat.x
    

def main():
    l = 4
    main_grid = GridParameters(l)
    user_proba = PreProcessing('COVID-19_Cases_US.csv', l)

    s = np.sum(user_proba.users_probabilities, axis=0)
    a = s / np.sum(s)

    print(user_proba.users_probabilities.shape)

    plt.figure(figsize=(8, 6))
    sns.heatmap(a, cmap='viridis', fmt='.2f', linecolor='white')
    plt.ylabel('latitude')
    plt.xlabel('longitude')
    plt.title('Non private Heatmap')
    plt.savefig('3nonprivate_l'+str(l)+'.png')
    plt.clf()
    
    eps_array = np.array([0.1, 0.5, 1, 2, 5, 10])
    emd_array = np.zeros_like(eps_array)
    sim_array = np.zeros_like(eps_array)
    kldiv_array = np.zeros_like(eps_array)

    for i, eps in enumerate(eps_array) :
        algo = SparseEMDAgg(user_proba.users_probabilities, main_grid.G, l, eps=eps, w=20)
        private_agg = algo.a_hat

        plt.figure(figsize=(8, 6))
        sns.heatmap(private_agg, cmap='viridis', fmt='.2f', linecolor='white')
        plt.ylabel('latitude')
        plt.xlabel('longitude')
        plt.title(r'Private Heatmap, $\varepsilon =$' + str(eps))
        plt.savefig('3private_eps'+str(eps)+'_l'+str(l)+'.png')    

        sim_array[i] = np.sum(np.sqrt(a * private_agg))
        epsilon=1e-12
        kldiv_array[i] = np.sum(a * np.log((a + epsilon) / (private_agg + epsilon)))
        emd_array[i] = wasserstein_distance(a.flatten(), private_agg.flatten())
    
    plt.clf()
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    axes[0].plot(eps_array, sim_array, marker='+', color='deepskyblue')
    axes[0].set_xlabel(r'$\varepsilon$')
    axes[0].set_title('Similarity')

    axes[1].plot(eps_array, kldiv_array, marker='+', color='mediumorchid')
    axes[1].set_xlabel(r'$\varepsilon$')
    axes[1].set_title('KL Divergence')

    axes[2].plot(eps_array, emd_array, marker='+', color='yellowgreen')
    axes[2].set_xlabel(r'$\varepsilon$')
    axes[2].set_title('EMD')

    plt.tight_layout()
    plt.savefig('3metrics'+'_l'+str(l)+'.png')

if __name__ == "__main__":
    main()