import networkx as nx
import geopandas as gpd


class NullModel:
    """
    Class representing a Null Model.

    Attributes:
        G (nx.Graph): A networkx graph representing the base model.
        base_model: The base model used to create the null model.
        rgg (nx.Graph): A random geometric graph representing the null model.
        pos_mercator: A dictionary containing the positions of the nodes in the base model.
        alpha (float): Transparency parameter for plotting.
        lw (float): Line width parameter for plotting.
        radius (int): Radius parameter for building the random geometric graph.
        folder (str): Folder path for saving plots.
        ext (str): File extension for saving plots.
        save (bool): Whether to save the plots or show them.
        filename (str): File name for saving plots.
        lg_conn_comp: Size of the largest connected component in the null model.
        degrees (list): List of degrees of nodes in the null model.
        k_avg (float): Average degree of nodes in the null model.
        k2_avg (float): Average squared degree of nodes in the null model.
        fc (float): Critical threshold of the null model.
        btwn (dict): Dictionary of betweenness centrality values for nodes in the null model.
        close (dict): Dictionary of closeness centrality values for nodes in the null model.
        color (str): Color parameter for plot title.
        title (str): Title of the plot.
        gdf_nodes: GeoDataFrame containing the nodes of the base model as points.

    Methods:
        build(): Builds the null model by creating the random geometric graph and calculating graph statistics.
        build_geo_graph(): Builds a GeoDataFrame with the nodes of the base model.
        calc_graph_stats(): Calculates various graph statistics for the null model.
    """
    def __init__(self, model):
        self.G = None
        self.base_model = model
        self.rgg = None
        self.pos_mercator = None
        self.alpha = 0.8
        self.lw = 0.2
        self.radius = 77147
        self.folder = 'sim_output/'
        self.ext = '.png'
        self.save = False
        self.filename = None
        self.lg_conn_comp = None
        self.degrees = None
        self.k_avg = None
        self.k2_avg = None
        self.fc = None
        self.btwn = None
        self.close = None
        self.color = 'black'
        self.title = 'Change Me'
        self.gdf_nodes = None

    def build(self):
        self.G = self.base_model.G
        self.rgg = nx.random_geometric_graph(self.G.nodes(), radius=self.radius, pos=self.pos_mercator)
        self.build_geo_graph()
        self.calc_graph_stats()

    def build_geo_graph(self):

        pos = nx.get_node_attributes(self.G, 'pos')

        for (u, v) in self.G.edges():
            distance = 1
            self.G[u][v]['weight'] = distance

        gdf_nodes = gpd.GeoDataFrame(
            index=self.G.nodes(),
            geometry=gpd.points_from_xy([pos[node][1] for node in self.G.nodes()],
                                        [pos[node][0] for node in self.G.nodes()],
                                        crs='EPSG:4326')
        )

        self.gdf_nodes = gdf_nodes.to_crs(epsg=3857)

    def calc_graph_stats(self):
        self.lg_conn_comp = len(max(nx.connected_components(self.rgg), key=len))
        self.degrees = [degree for node, degree in self.rgg.degree()]
        n_deg = len(self.degrees)
        self.k_avg = sum(self.degrees) / n_deg
        self.k2_avg = sum(degree ** 2 for degree in self.degrees) / n_deg
        self.fc = 1 - 1 / (self.k2_avg / self.k_avg - 1)

        self.btwn = nx.betweenness_centrality(self.rgg)
        self.close = nx.closeness_centrality(self.rgg)
