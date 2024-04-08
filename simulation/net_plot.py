import networkx as nx
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np


class NetPlot:
    """
    NetPlot class provides methods for plotting network graphs and analyzing various network properties.

    Attributes:
        - sim: An instance of the simulation class.
        - gdf_nodes: GeoDataFrame containing the nodes of the graph
        - G: Base network graph
        - GN: Null network graph
        - RGG: Random Geometric Graph
        - title: Title of the graph
        - save: Flag to determine whether to save the graph
        - filename: Name of the file to save the graph
        - color: Color of the title in the graph
        - folder: Folder to save the graph
        - ext: File extension of the saved graph
        - pos_mercator: Dictionary containing the Mercator coordinates of the nodes
        - perc: Percentage of nodes removed in log plot
        - type: Type of network (base or null)
        - alpha: Transparency of the edges in the graph
        - lw: Line width of the edges in the graph

    Methods:
        - plot_network(title='Change Me'): Plots the base network graph
        - plot_null_network(title='Change Me'): Plots the null network graph
        - add_sim(sim): Adds a simulation instance to the class
        - build_geo_graph(): Builds the GeoDataFrame for the nodes
        - model(): Returns the appropriate network graph based on the type attribute
        - plot_null_graph(): Plots the null graph
        - plot_graph(): Plots the base graph with additional features
        - log_plot(sim, perc_removed): Plots the log plot for the specified simulation
        - compare_plot(title): Compares the clustering coefficient and shortest path for the base and null graph

    This class provides various methods for analyzing and visualizing network graphs.
    """
    def __init__(self):
        self.sim = None
        self.gdf_nodes = None
        self.G = None
        self.GN = None
        self.RGG = None
        self.title = None
        self.save = False
        self.filename = None
        self.color = 'black'
        self.folder = 'sim_output/'
        self.ext = '.png'
        self.pos_mercator = None
        self.perc = None
        self.type = 'base'
        self.alpha = 0.5
        self.lw = 0.2

    def plot_network(self, title='Change Me'):
        self.title = title
        self.build_geo_graph()
        self.plot_graph()

    def plot_null_network(self, title='Change Me'):
        self.title = title
        self.build_geo_graph()
        self.plot_null_graph()

    def add_sim(self, sim):
        self.sim = sim
        self.G = sim.G
        self.GN = sim.GN

    def build_geo_graph(self):
        """
        Builds a geographic graph using the given model.
        :return: None
        """
        # Instantiate the model to generate a graph
        g = self.model()

        # Fetch the node attributes 'pos'
        pos = nx.get_node_attributes(g, 'pos')

        # Set 'weight' attribute of each edge in the graph as 1: intent was to plug in weights here eventually but we ran out of time
        for (u, v) in g.edges():
            distance = 1
            g[u][v]['weight'] = distance

        # Create a GeoDataFrame with index as graph nodes and the pos as the geometry
        gdf_nodes = gpd.GeoDataFrame(
            index=g.nodes(),
            geometry=gpd.points_from_xy([pos[node][1] for node in g.nodes()],
                                        [pos[node][0] for node in g.nodes()],
                                        crs='EPSG:4326')
        )

        # Convert the CRS of gdf_nodes to epsg=3857 and assign to attribute
        self.gdf_nodes = gdf_nodes.to_crs(epsg=3857)

    def model(self):
        if self.type == 'base':
            return self.G
        else:
            return self.GN

    def plot_null_graph(self):
        # Generate a graph model
        g = self.model()

        # Calculate betweenness centrality for each node in the RGG graph
        bet_cen = nx.betweenness_centrality(self.RGG)

        # Calculate closeness centrality for each node in the RGG graph
        clos_cen = nx.closeness_centrality(self.RGG)

        # Find the maximum betweenness centrality value
        btw_max = max(bet_cen.values())

        # To avoid division by zero set the maximum betweenness centrality value to a really small number if it's zero
        if btw_max <= 0:
            btw_max = 0.00001

        # Normalize betweenness centrality values
        normalized_bet_cen = {
            node: value / btw_max for node, value in bet_cen.items()
        }

        # Create a list of normalized betweenness centrality values corresponding to each node in the RGG graph
        node_color = [normalized_bet_cen.get(node) for node in self.RGG.nodes()]

        # Create a list of node sizes corresponding to closeness centrality values
        sizes = [(v ** 3) * 150000 for v in clos_cen.values()]

        # Get Mercator coordinates for each node in the graph
        pos_mercator = {
            node: (point.x, point.y) for node, point in zip(self.gdf_nodes.index, self.gdf_nodes.geometry)
        }

        # Debug message
        print(len(sizes), len(node_color), len(g.nodes()), len(self.RGG.nodes()), len(pos_mercator))

        # Create a figure
        fig, ax = plt.subplots(figsize=(20, 30))

        # Draw nodes
        nx.draw_networkx_nodes(g, pos=pos_mercator, ax=ax, node_size=sizes, node_color=node_color, cmap=plt.cm.coolwarm)

        # Draw edges
        nx.draw_networkx_edges(self.RGG, pos=pos_mercator, ax=ax, alpha=self.alpha, width=self.lw)

        # Set graph title
        ax.set_title(self.title)

        # Show axes
        plt.axis('on')

        # Save or display the graph
        if self.save:
            plt.savefig(self.folder + 'null-model/' + 'null_' + self.filename + self.ext)
        else:
            plt.show()
        plt.close()

    def plot_graph(self):
        # Create an instance of the model
        g = self.model()

        # Use GeoDataFrame instance attribute of the class
        gdf_nodes = self.gdf_nodes

        # Create a plot area
        fig, ax = plt.subplots(figsize=(20, 30))

        # Plot the nodes to the chart
        gdf_nodes.plot(ax=ax, markersize=20, color='green')

        # Calculate the betweenness centrality of the graph
        bet_cen = nx.betweenness_centrality(g)
        # Calculate the closeness centrality of the graph
        clos_cen = nx.closeness_centrality(g)

        # Find maximal centrality value
        btw_max = max(bet_cen.values())

        # Check for case when maximum degree of centrality is 0
        if btw_max <= 0:
            btw_max = 0.00001

        # Normalise betweenness centrality
        normalized_bet_cen = {
            node: value / btw_max for node, value in bet_cen.items()
        }

        # Define node colors and sizes
        node_color = [normalized_bet_cen.get(node) for node in g.nodes()]
        sizes = [(v ** 3) * 150000 for v in clos_cen.values()]

        # Add a basemap to the plot
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Voyager)

        # Set numerical limits on plot axes
        ax.set_xlim(
            gdf_nodes.total_bounds[[0, 2]].min() - 20000, gdf_nodes.total_bounds[[0, 2]].max() + 20000
        )
        ax.set_ylim(
            gdf_nodes.total_bounds[[1, 3]].min() - 20000, gdf_nodes.total_bounds[[1, 3]].max() + 20000
        )

        # Create nodes coordinates dictionary
        pos_mercator = {
            node: (point.x, point.y) for node, point in zip(gdf_nodes.index, gdf_nodes.geometry)
        }
        self.pos_mercator = pos_mercator

        # Map planning area names to nodes
        labels = nx.get_node_attributes(g, 'Planning Area')

        # Plot the graph
        nx.draw(g, pos=pos_mercator, ax=ax, node_size=sizes, node_color=node_color,
                cmap=plt.cm.coolwarm, alpha=1, with_labels=True, labels=labels)

        # Set title and make it bold
        plt.title(self.title, fontsize=23, fontweight='bold', color=self.color)

        # Don't show axes
        plt.axis('off')

        # Save or show the graph
        if self.save:
            plt.savefig(self.folder + 'base-model/' + 'base_' + self.filename + self.ext)
        else:
            plt.show()

        # Clear plot to save memory
        plt.close()

    def log_plot(self, sim):
        ### THIS CODE WAS INTEGRATED AND UPDATED FROM THE COURSE MATERIALS ###
        # Instantiate the model to generate a graph
        g = self.model()

        # Calculate the degree for all nodes in the graph
        degrees = [g.degree(node) for node in g]

        # Calculate minimum degree and add a small value to prevent division by zero error
        kmin = min(degrees) + 0.001
        kmax = max(degrees)

        kavg = np.average(degrees)

        # Get 30 logarithmically spaced bins between kmin and kmax
        bin_edges = np.logspace(np.log10(kmin), np.log10(kmax), num=30)
        # Histogram the degrees into these bins
        density, _ = np.histogram(degrees, bins=bin_edges, density=True)

        plt.figure(figsize=(4, 5))

        log_be = np.log10(bin_edges)
        x = 10**((log_be[1:] + log_be[:-1]) / 2)

        plt.loglog(x, density, marker='o', linestyle='none')
        plt.xlabel(r"degree $k$", fontsize=16)
        plt.ylabel(r"$P(k)$", fontsize=16)
        plt.title(f'Sim: {sim}, <k>=: {np.round(kavg, 2)}')

        ax = plt.gca()
        #
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # Check save flag and either save the figure or display in the notebook
        if self.save:
            plt.savefig(self.folder + 'log-plot/' + 'log_' + self.filename + self.ext)
        else:
            plt.show()

        plt.close()  # Close the figure to free up memory

    def compare_plot(self, title):
        # Adapted from a file by Raffi. His file will contain attributions.

        # Calculate average clustering coefficient and average shortest path length.
        # We're taking the mean of all the clustering coefficients for all the nodes in the Alberta Power Grid.
        C = np.mean(list(nx.clustering(self.GN).values()))
        D = nx.average_shortest_path_length(self.GN, weight=None)

        # Print Average Clustering Coefficient and Average Shortest Path for Alberta Power Grid.
        print("Average Clustering Coefficient (Alberta Power Grid): ", C)
        print("Average Shortest Path (Alberta Power Grid): ", D)

        # Repeat the same for a Random Geometric Graph.
        c = np.round(np.mean(list(nx.clustering(self.RGG).values())), 4)
        d = np.round(nx.average_shortest_path_length(self.RGG, weight=None), 4)

        # Print Average Clustering Coefficient and Average Shortest Path for Random Geometric Graph.
        print("Average Clustering Coefficient (Random Geometric Graph): ", c)
        print("Average Shortest Path (Random Geometric Graph): ", d)

        # Define clustering values for both graphs.
        c_values = [c, C]

        # Set up plot for clustering.
        fig = plt.figure(figsize=(6, 4))
        plt.boxplot(c_values)

        # Scatter plot clustering values.
        plt.scatter([2], [C], color='r', marker='+', s=150)

        plt.xticks([1, 2], labels=['Random Geometric Graph', 'Alberta Power Grid'])
        plt.ylabel('Clustering')

        plt.ylim(0, 2)
        plt.yticks(np.arange(3))

        plt.xlim(0.5, 2.5)

        plt.title(f'Clustering Coefficient Avg_Real({C}) Avg_Rand({c})', fontweight='bold', color=self.color)

        # Conditionally save or show the plot.
        if self.save:
            plt.savefig(self.folder + 'null-clust/' + 'clust_' + self.filename + self.ext)
        else:
            plt.show()

        # Define path values for both graphs.
        d_values = [d, D]

        # Set up plot for paths.
        fig = plt.figure(figsize=(6, 4))
        plt.boxplot(d_values)

        # Scatter plot path values.
        plt.scatter([2], [D], color='r', marker='+', s=150)

        # Set up labels for x-axis and y-axis.
        plt.xticks([1, 2], labels=['Random Geometric Graph', 'Alberta Power Grid'])
        plt.ylabel('Average Shortest Path')

        # Define y-axis limit and ticks.
        plt.ylim(0, 10)
        plt.yticks(np.arange(11))

        # Set axis limits.
        plt.xlim(0.5, 2.5)

        # Add a title to the plot.
        plt.title(f'Shortest Path Avg_Real({D}) Avg_Rand({d})')

        # Conditionally save or show the plot.
        if self.save:
            plt.savefig(self.folder + 'null-short-path/' + 'sp_' + self.filename + self.ext)
        else:
            plt.show()

        plt.close()

    def null_degree_dist(self, sim, k_avg):
        ### CODE ADAPTED FROM COURSE MATERIALS ###
        # Plotting degree distribution function (referred to null model exercise from class)\
        degrees = [self.RGG.degree(n) for n in self.RGG.nodes()]
        kmin = min(degrees)
        kmax = max(degrees)

        if kmin > 0:
            bin_edges = np.logspace(np.log10(kmin), np.log10(kmax) + 1, num=30)
        else:
            bin_edges = np.logspace(0, np.log10(kmax) + 1, num=20)
        density, _ = np.histogram(degrees, bins=bin_edges, density=True)

        fig = plt.figure(figsize=(6, 4))
        log_be = np.log10(bin_edges)
        x = 10 ** ((log_be[1:] + log_be[:-1]) / 2)

        plt.loglog(x, density, marker='o', linestyle='none')
        plt.xlabel(r"degree $k$", fontsize=16)
        plt.ylabel(r"$P(k)$", fontsize=16)
        plt.title(f'Sim: {sim}, <k>=: {np.round(k_avg, 2)}')
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        if self.save:
            plt.savefig(self.folder + 'null-degree/' + 'degree_' + self.filename + self.ext)
        else:
            plt.show()

        plt.close()
