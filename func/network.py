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
        return self.adj_mat.sum(axis=1)-1

    def shortestpath(self):
        """
        Calculate the shortest path between all nodes in the network using Dijstrak Algorithm:
        https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
        :return Dictionary of two nxn dimensional pd.DataFrames with shortest path / shortest distance between all pairs of nodes in the network
        """
        inv_adj_mat=self.adj_mat.abs()                                                                                  # Inverts adjacency matrix
        shortestdist_df=pd.DataFrame(np.zeros(inv_adj_mat.shape), columns=self.nodes, index=self.nodes)                 # Initialize Path matrix and distance matrix
        shortestpath_df=pd.DataFrame(np.empty(inv_adj_mat.shape, dtype=str), columns=self.nodes, index=self.nodes)

        for n in range(len(self.nodes)):
            node_set=pd.DataFrame({'Distance': np.full((len(self.nodes)), np.inf),
                                   'Previous': ['']*(len(self.nodes)), 'Path': ['']*(len(self.nodes))}, index=self.nodes)
            node_set.loc[self.nodes[n], 'Distance'] = 0
            unvisited_nodes=self.nodes.copy()
            while unvisited_nodes != []:
                current=node_set.loc[unvisited_nodes,'Distance'].idxmin()    # Select node with minimal Distance of the unvisited nodes
                unvisited_nodes.remove(current)
                for k in self.nodes:
                    dist=node_set.loc[current, 'Distance'] + inv_adj_mat.loc[current, k]
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
        return {'Distance': shortestdist_df, 'Path': shortestpath_df}

    def num_triangles(self):
        """
        Calculate sum of triangles edge weights around each node in network
        :return: n dimensional pd.Series
        """
        triangles=pd.Series(np.zeros(len(self.nodes)), index=self.nodes)
        all_combinations=combinations(self.nodes, 3)        # Create list of all possible triangles
        abs_adj_mat = self.adj_mat.abs()
        sum_dict={}
        for combi in all_combinations:
            n1_n2=abs_adj_mat.loc[combi[0],combi[1]]        # Get path length between pairs in triangle combination
            n1_n3=abs_adj_mat.loc[combi[0],combi[2]]
            n2_n3=abs_adj_mat.loc[combi[1],combi[2]]
            sum_dict[combi]=(n1_n2+n1_n3+n2_n3)**(1/3)       # Calculate the triangle sum of the combination and save it in dictionary
        for node in self.nodes:
            triangles[node]=0.5*np.sum([sum_dict[s] for s in sum_dict if node in s])    # Sum all of the triangles that contain the node
        return triangles

    def char_path(self, shortest_pathlength=None):
        """
        Calculate the characteristic path length of the network
        :return: Dictionary with average node distance np.array and characteristic path length np.float object
        """
        if shortest_pathlength is not None:                                         # Check if shortest_pathlength is defined
            if not isinstance(shortest_pathlength, (np.ndarray, pd.DataFrame)):
                raise ValueError('Shortest Pathlength must be numpy.ndarray or pd.DataFrame')
            sum_shrtpath_df = np.sum(np.asarray(shortest_pathlength), axis=-1)                # Sums Shortest Path Dataframe along axis -1
        else:
            sum_shrtpath_df = np.sum(self.shortestpath()['Distance'], axis=-1)              # Sums Shortest Path Dataframe along axis -1
        avg_shrtpath_node = sum_shrtpath_df / (len(self.nodes)-1)                       # Divide each element in sum array by n-1 regions
        char_pathlength = np.sum(avg_shrtpath_node) / len(self.nodes)                     #
        return {'node_avg_dist':avg_shrtpath_node, 'characteristic_path': char_pathlength}    # Calculate sum of the sum array and take the average

    def glob_efficiency(self):
        """
        Calculate the global efficiency of the network
        :return: np.float object
        """
        inv_shrtpath=self.shortestpath()['Distance'].pow(-1)        # Takes the inverse of each element of the Dataframe
        np.fill_diagonal(inv_shrtpath.to_numpy(), 0)                # Set Diagonal from inf -> 0
        sum_invpath_df=inv_shrtpath.sum(axis=1)                     # Sums Shortest Path Dataframe along axis 1
        avg_invpath=np.divide(sum_invpath_df, len(self.nodes)-1)    # Divide each element in sum array by n-1 regions
        glob_efficiency= np.sum(avg_invpath) / len(self.nodes)      # Calculate sum of the sum array and take the average
        return glob_efficiency

    def clust_coeff(self):
        """
        Calculate the cluster coefficient of the network
        :return: Dictionary of network cluster coefficient np.float object and ndim np.array of node cluster coefficients
        """
        triangles=np.multiply(np.array(self.num_triangles()), 2)
        degrees=np.array(self.degree())
        excl_nodes=np.where(degrees < 2); triangles[excl_nodes]=0
        degrees=np.multiply(degrees, degrees-1)
        node_clust=np.divide(triangles,degrees)
        net_clust=(1/len(self.nodes))*np.sum(node_clust)
        return {'node_cluster':pd.Series(node_clust, index=self.nodes), 'net_cluster':net_clust}

    def transitivity(self):
        """
        Calculate the transitivity of the network
        :return: np.float
        """
        triangles=np.sum(np.multiply(np.asarray(self.num_triangles()),2))     # Multiply sum of triangles with 2 and sum the array
        degrees=np.array(self.degree())
        degrees=np.sum(np.multiply(degrees, degrees-1))
        return np.divide(triangles, degrees)

    def closeness_centrality(self):
        """
        Calculate the closeness centrality of each node in network
        :return: ndimensional pd.Series
        """
        node_avg_distance=self.char_path()['node_avg_dist']
        return pd.Series(np.power(node_avg_distance, -1), index=self.nodes)

    def betweenness_centrality(self):
        """
        Calculate the betweenness centrality of each node in network
        :return: ndimensional pd.Series
        """
        betw_centrality=pd.Series(np.zeros(len(self.nodes)), index=self.nodes)
        shortest_paths=self.shortestpath()['Path']

        for n in self.nodes:
            counter = 0
            mat=shortest_paths.drop(n, axis=0); mat=mat.drop(n, axis=1)  # Drops the nth column and the nth row.
            substr='-'+str(n)+'-'

            for c in mat.columns:
                for e in mat.loc[:c,c]:
                    if e.find(substr) != -1:
                        counter += 1
            betw_centrality.loc[n]=counter/((len(self.nodes)-1)*(len(self.nodes)-2))

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
            random_clust_coeff = random_net.clust_coeff()['net_cluster']
            random_char_path = random_net.char_path()['characteristic_path']

        else:
            if nrandnet < 1: raise ValueError("Minimum one iteration")
            random_clust_coeff = []
            random_char_path = []
            for i in range(nrandnet):
                random_net=randomnet.rewired_rand(self.adj_mat, niter, seed)
                print(f'{i+1} random network generated.')
                random_clust_coeff.append(random_net.clust_coeff()['net_cluster'])
                random_char_path.append(random_net.char_path()['characteristic_path'])
                print(f'Random Char Path: {random_char_path}')
                print(f'Random clust coeff: {random_clust_coeff}')

            random_clust_coeff=np.mean(random_clust_coeff)
            random_char_path=np.mean(random_char_path)

        sig_num=(self.clust_coeff()['net_cluster']/random_clust_coeff)
        sig_den=(self.char_path()['characteristic_path']/random_char_path)
        sigma=sig_num/sig_den

        return sigma


    def modularity(self):
        #TODO find algorithm to find modules in network
        return

