import pandas as pd
import numpy as np
from itertools import combinations
import func.corr_functions as corr


class network:
    """Defines input as network
    :parameter pd.DataFrame that contains the adjacency matrix of the network, np.ndarray timecourse matrix
    TODO use absolute values or set negativ values to zero
    """
    def __init__(self, Adjacency_Matrix, tc=[]):
        assert isinstance(Adjacency_Matrix, (pd.DataFrame, np.ndarray)), "Input must be numpy.ndarray or panda.DataFrame"
        self.adj_mat=pd.DataFrame(Adjacency_Matrix)
        self.nodes = list(self.adj_mat.index)
        self.number_nodes = len(self.nodes)

        if tc:
            assert isinstance(tc, np.ndarray), "Timecourse must be np.ndarray"
            self.time_course=pd.DataFrame(tc, index=self.nodes)
            self.cov_mat=corr.covariance_mat(tc)
        else:
            self.time_course=None

    def degree(self, node="all"):
        """
        Calculate the degree of each node in the network.
        :return n dimensional pd.Series with degrees of all nodes in the network
        """
        diag = np.diagonal(np.array(self.adj_mat))
        if not np.all(np.isclose(diag, 0)):             # Check if the diagonal is close to zero
            raise Exception('The diagonal elements of adjacency matrix must be close to zero')

        degree = self.adj_mat.sum(axis=1)               # Calculate degree and return as pd.Series
        return degree

    def shortestpath(self):
        """
        Calculate the shortest path between all nodes in the network using Dijstrak Algorithm:
        https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
        :return Dictionary of two nxn dimensional pd.DataFrames with shortest path / shortest distance between all pairs of nodes in the network
        """
        adj_mat = self.adj_mat
        if not np.all(adj_mat>=0): raise ValueError('Adjancency matrix elements must be positiv.')
        shortestdist_df=pd.DataFrame(np.zeros(adj_mat.shape), columns=self.nodes, index=self.nodes)                     # Initialize Path matrix and distance matrix
        shortestpath_df=pd.DataFrame(np.empty(adj_mat.shape, dtype=str), columns=self.nodes, index=self.nodes)

        for n in range(self.number_nodes):
            node_set=pd.DataFrame({'Distance': np.full((self.number_nodes), np.inf),
                                   'Previous': ['']*(self.number_nodes), 'Path': ['']*(self.number_nodes)}, index=self.nodes)
            node_set.loc[self.nodes[n], 'Distance'] = 0
            unvisited_nodes=self.nodes.copy()
            while unvisited_nodes != []:
                current=node_set.loc[unvisited_nodes,'Distance'].idxmin()    # Select node with minimal Distance of the unvisited nodes
                unvisited_nodes.remove(current)
                for k in self.nodes:
                    dist=node_set.loc[current, 'Distance'] + adj_mat.loc[current, k]
                    if node_set.loc[k,'Distance'] > dist:
                        node_set.loc[k,'Distance'] = dist
                        node_set.loc[k,'Previous'] = current
            shortestdist_df.loc[:,n]=node_set.loc[:,'Distance']
            shortestdist_df.loc[n, :]=node_set.loc[:,'Distance']

            # Create Dataframe with string values for the shortest path between each pair of nodes
            for k in self.nodes:
                path=str(k)
                current=k
                while node_set.loc[current, 'Previous'] != '':
                    current=node_set.loc[current, 'Previous']
                    path=str(current)+', '+path
                node_set.loc[k,'Path']=path
            shortestpath_df.loc[:,n]=node_set.loc[:,'Path']
            shortestpath_df.loc[n,:]=node_set.loc[:,'Path']
        return {'Path_Length': shortestdist_df, 'Path': shortestpath_df}

    def num_triangles(self, normalize=True):
        """
        Calculate sum of triangles edge weights around each node in network
        :return: n dimensional pd.Series
        """
        triangles = pd.Series(np.zeros(self.number_nodes), index=self.nodes)
        all_combinations = combinations(self.nodes, 3)        # Create list of all possible triangles
        adj_mat = self.adj_mat
        if not np.all(adj_mat>=0): raise ValueError('Adjancency matrix elements must be positiv.')

        if normalize:                                   # Normalizes the weights by the maximum weight.
            max_weight = np.max(adj_mat.to_numpy())
            adj_mat /= max_weight

        m_dict={}
        for combi in all_combinations:
            n1_n2 = adj_mat.loc[combi[0],combi[1]]        # Get path length between pairs in triangle combination
            n1_n3 = adj_mat.loc[combi[0],combi[2]]
            n2_n3 = adj_mat.loc[combi[1],combi[2]]
            m_dict[combi] = (n1_n2*n1_n3*n2_n3)**(1/3)       # Calculate the triangle sum of the combination and save it in dictionary
        for node in self.nodes:
            triangles.loc[node] = 0.5 * np.sum([m_dict[s] for s in m_dict if node in s])    # Sum all of the triangles that contain the node
        return triangles

    def char_path(self, node_by_node=False, shortestpath=None):
        """
        Calculate the characteristic path length of the network
        :return: Dictionary with average node distance np.array and characteristic path length np.float object
        """
        if shortestpath is not None:                                         # Check if shortest_pathlength is defined
            if not isinstance(shortestpath, (np.ndarray, pd.DataFrame)):
                raise ValueError('Shortest Pathlength must be numpy.ndarray or pd.DataFrame')
            sum_shrtpath_df = np.sum(np.asarray(shortestpath), axis=-1)                         # Sums Shortest Path Dataframe along axis -1
        else:
            sum_shrtpath_df = np.sum(np.asarray(self.shortestpath()['Path_Length']), axis=-1)   # Sums Shortest Path Dataframe along axis -1
        avg_shrtpath_node = sum_shrtpath_df / (self.number_nodes-1)                             # Divide each element in sum array by n-1 regions
        char_pathlength = np.sum(avg_shrtpath_node) / self.number_nodes

        if node_by_node:
            return avg_shrtpath_node                                                            # Return average shortest path node by node
        else:
            return char_pathlength

    def glob_efficiency(self, shortestpath=None):
        """
        Calculate the global efficiency of the network
        :return: np.float object
        """
        if shortestpath is not None:
            if not isinstance(shortestpath, (np.ndarray, pd.DataFrame)):     # Check if input is np.ndarray or pd.DataFrame
                raise ValueError('Shortest Pathlength must be numpy.ndarray or pd.DataFrame')
            shortestpath = np.asarray(shortestpath)
            np.fill_diagonal(shortestpath, 1.)             # Sets diagonal to 1
        else:
            shortestpath = np.asarray(self.shortestpath()['Path_Length'])
            np.fill_diagonal(shortestpath, 1.)      # Set diagonal to 1

        inv_shrtpath=1/shortestpath                                                 # Computes shortest path and takes inverse
        np.fill_diagonal(inv_shrtpath, 0)                                           # Set Diagonal from 1 -> 0
        sum_invpath_df = np.sum(inv_shrtpath, axis=-1)                              # Sums Shortest Path Dataframe along axis 1
        avg_invpath = sum_invpath_df / (self.number_nodes-1)                        # Divide each element in sum array by n-1 regions
        glob_efficiency= np.sum(avg_invpath) / self.number_nodes                    # Calculate sum of the sum array and take the average
        return glob_efficiency

    def clust_coeff(self, node_by_node=False, normalize=False):
        """
        Calculate the cluster coefficient of the network
        :return: Network cluster coefficient np.float object or ndim np.array of node by node cluster coefficients
        """
        if not isinstance(normalize, bool): raise ValueError('Normalize must be boolean (True/False).')

        triangles = np.array(self.num_triangles(normalize=normalize)) * 2
        degrees = np.array(self.degree())
        excl_nodes = np.where(degrees < 2); triangles[excl_nodes] = 0; degrees[excl_nodes] = 2     # Sets traingle sum to zero where degree is below 2
        degrees *= (degrees-1)
        node_clust = triangles / degrees
        net_clust = (1/self.number_nodes) * np.sum(node_clust)

        if node_by_node:
            return pd.Series(node_clust, index=self.nodes)
        else:
            return net_clust

    def transitivity(self):
        """
        Calculate the transitivity of the network
        :return: np.float
        """

        sum_triangles = np.sum(np.asarray(self.num_triangles())*2)     # Multiply sum of triangles with 2 and sum the array
        degrees = np.asarray(self.degree())
        degrees *= (degrees-1)
        sum_degrees = np.sum(degrees)
        transitivity = sum_triangles / sum_degrees

        return transitivity

    def closeness_centrality(self, char_path=None):
        """
        Calculate the closeness centrality of each node in network.
        Optionally takes in the shortest average pathlength of each node, which saves computation time.
        :param: n dimensional array or pd.Series that contains the average shortest path for each node
        :return: ndimensional pd.Series
        """
        if char_path is not None:               # If char_path is defined takes this as input
            if not isinstance(char_path, (pd.Series, np.ndarray)): raise ValueError('Characteristic Path must be pd.Series or np.ndarray')
            node_avg_distance = np.asarray(char_path)
        else:
            node_avg_distance = np.asarray(self.char_path(node_by_node=True))  # Compute average shortest path node by node
        if not np.all(node_avg_distance): raise Exception('Node by node shortest average path must not contain zeros.')

        close_cent = 1 / node_avg_distance                      # Inverts the average shortest path
        close_cent = pd.Series(close_cent, index=self.nodes)    # Converts to pd.Series
        return close_cent

    def betweenness_centrality(self):
        """
        Calculate the betweenness centrality of each node in network
        :return: ndimensional pd.Series
        """
        betw_centrality=pd.Series(np.zeros(self.number_nodes), index=self.nodes)
        shortest_paths=self.shortestpath()['Path']

        for n in self.nodes:
            counter = 0
            mat=shortest_paths.drop(n, axis=0); mat=mat.drop(n, axis=1)  # Drops the nth column and the nth row.
            substr='-'+str(n)+'-'

            for c in mat.columns:
                for e in mat.loc[:c,c]:
                    if e.find(substr) != -1:
                        counter += 1
            betw_centrality.loc[n]=counter/((self.number_nodes-1)*(self.number_nodes-2))

        return betw_centrality

    def small_worldness(self, nrandnet=10, niter=10, seed=None, hqs=False, tc=[]):
        """
        Computes small worldness (sigma) of network
        :param: seed: float or integer which sets the seed for random network generation
                niter: int of number of iterations that should be done during network generation
                hqs: boolean value defines if hqs is used for random network generation
                tc: timecourse as np.ndarray for hqs algorithm
        :return:
        """
        import func.random_reference as randomnet

        if hqs:
            tc = np.array(tc)
            if not tc.any() and self.time_course.any():
                tc = self.time_course
            elif not tc.any(): raise ValueError("Timecourse not specified")

            random_net = randomnet.hqs_rand(tc)
            random_clust_coeff = random_net.clust_coeff()
            random_char_path = random_net.char_path()['characteristic_path']

        else:
            if nrandnet < 1: raise ValueError("Minimum one iteration")
            random_clust_coeff = []
            random_char_path = []
            for i in range(nrandnet):
                random_net=randomnet.rewired_rand(self.adj_mat, niter, seed)
                print(f'{i+1} random network generated.')
                random_clust_coeff.append(random_net.clust_coeff())
                random_char_path.append(random_net.char_path()['characteristic_path'])
                print(f'Random Char Path: {random_char_path}')
                print(f'Random clust coeff: {random_clust_coeff}')

            random_clust_coeff=np.mean(random_clust_coeff)
            random_char_path=np.mean(random_char_path)

        sig_num=(self.clust_coeff()/random_clust_coeff)
        sig_den=(self.char_path()['characteristic_path']/random_char_path)
        sigma=sig_num/sig_den

        return sigma


    def modularity(self):
        #TODO find algorithm to find modules in network
        pass

