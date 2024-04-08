import networkx as nx
import numpy as np
from storage.storage import Storage


class RbModel:
    """
    This class represents an RbModel object.

    Attributes:
        G (networkx.Graph): The graph object representing the model.
        db (int): The database object.
        nodes (pandas.DataFrame): The node data.
        edges (pandas.DataFrame): The edge data.
        locations (dict): A dictionary mapping facility codes to locations.
        gdf_nodes: The graph data frame for nodes.
        num_stat_runs (int): The number of statistical runs.
        lg_conn_comp (int): The size of the largest connected component.
        degrees (list): A list of degrees for each node.
        k_avg (float): The average degree of the graph.
        k2_avg (float): The average squared degree of the graph.
        fc (float): The critical threshold.
        btwn (dict): A dictionary of betweenness centrality values.
        close (dict): A dictionary of closeness centrality values.

    Methods:
        __init__(self, g): Initializes the RbModel object.
        __getstate__(self): Returns the state of the object for pickling.
        build(self): Builds the model by calling other methods.
        get_data(self): Retrieves data from the database.
        add_nodes(self): Adds nodes to the graph.
        add_edges(self): Adds edges to the graph.
        check_node(self, code): Checks if a specific facility code exists in the graph.
        get_title(self): Returns the title for the model.
        stats(self): Prints various statistics about the graph.
        calc_graph_stats(self): Calculates various graph statistics.
        raff_patch(self): Adds patches to fix data issues.
    """
    def __init__(self, g):
        self.G = g
        self.db = 0
        self.nodes = None
        self.edges = None
        self.locations = {}
        self.gdf_nodes = None
        self.num_stat_runs = 100
        self.lg_conn_comp = None
        self.degrees = None
        self.k_avg = None
        self.k2_avg = None
        self.fc = None
        self.btwn = None
        self.close = None

    # This method is for pickling the object
    def __getstate__(self):
        # Copy the object's state
        state = self.__dict__.copy()
        # Removes the 'db' attribute because it can't be pickled
        del state['db']
        # Returns the modified state
        return state

    # This method builds the model by calling other methods
    def build(self):
        # Retrieves data from the database
        self.get_data()
        # Adds nodes to the graph
        self.add_nodes()
        # Adds edges to the graph
        self.add_edges()
        # Calculates various graph statistics
        self.calc_graph_stats()

    # This method retrieves data from the database
    def get_data(self):
        # Creates a new Storage object
        self.db = Storage()
        # Sets the database file path
        self.db.db = '../network_data/grid.db'
        # Exports the 'nodes_loc' table from the database as dataframe
        self.nodes = self.db.export_table('nodes_loc')
        # Exports the 'edges' table from the database
        self.edges = self.db.export_table('edges')

    def add_nodes(self):
        node_data, edge_data = self.nodes.to_dict('records'), self.edges.to_dict('records')

        for location in node_data:
            self.locations[location['Facility Code']] = [
                location['Latitude (generated)'],
                location['Longitude (generated)']
            ]
            self.G.add_node(
                location['Facility Code'],
                name=location['Name'],
                node_id=location['node_id'],
                tfo=location['TFO'],
                planning_area=location['Planning Area'],
                bus_number=location['Bus Number'],
                voltage_kv=location['Voltage (kV)'],
                capability_mw=location['Capability (MW)'],
                substation=location['Substation'],
                pos=(location['Latitude (generated)'], location['Longitude (generated)'])
            )

    def add_edges(self):
        """
        This method processes edge data to create pairs of source-target nodes.
        It adds an edge between these pairs (source, target) only when both nodes exist in the graph.
        """
        # Preprocess edge data to create source-target pairs
        edge_pairs = self.edges.groupby('Line Name').apply(
            lambda df: df.iloc[0]['Facility Code'] if len(df) == 1 else list(df['Facility Code'])
        )

        # Construct (source, target, edge_attributes) list to add to graph
        edges_to_add = [(pair[0], pair[1], {'edge_id': idx}) for idx, pair in enumerate(edge_pairs) if
                        isinstance(pair, list) and pair[0] is not None and pair[1] is not None]

        # Iterate over (source, target, edge_attributes) list
        for source, target, edge_attr in edges_to_add:
            # Add edge if both source and target nodes exist in the graph
            if self.check_node(source) and self.check_node(target):
                self.G.add_edge(source, target, **edge_attr)

        # Run the raff_patch method to fix data issues
        self.raff_patch()

    def check_node(self, code):
        """
        This method checks if a specific node (identified through its facility code) exists in the graph.
        If the node exists, it returns True, otherwise it returns False.
        """
        found = False

        for node in self.G.nodes():
            # If node's facility_code matches the specific facility code, update found flag to True
            if node == code:
                found = True
                return found

        # If node not found in the graph, return found flag (False)
        if not found:
            return found

    def stats(self):
        print('STATS')
        clust_coeff = np.mean(list(nx.clustering(self.G).values()))
        number_of_components = nx.number_connected_components(self.G)
        largest_cc_size = len(max(nx.connected_components(self.G), key=len))
        avg_deg = np.average([self.G.degree(n) for n in self.G.nodes()])

        avg_path_lengths = []

        for i in nx.connected_components(self.G):
            subgraph = self.G.subgraph(i)
            avg_path_length = nx.average_shortest_path_length(subgraph)
            avg_path_lengths.append(avg_path_length)

        print('BEFORE ATTACK')

        print("Number of connected components:", number_of_components)
        print("Size of the largest connected component:", largest_cc_size)
        print(f'clustering coefficient: {clust_coeff}, shortest path per component: {avg_path_lengths}')

    def calc_graph_stats(self):
        self.lg_conn_comp = len(max(nx.connected_components(self.G), key=len))
        self.degrees = [degree for node, degree in self.G.degree()]
        n_deg = len(self.degrees)
        self.k_avg = sum(self.degrees) / n_deg
        self.k2_avg = sum(degree ** 2 for degree in self.degrees) / n_deg
        self.fc = 1 - 1 / (self.k2_avg / self.k_avg - 1)

        self.btwn = nx.betweenness_centrality(self.G)
        self.close = nx.closeness_centrality(self.G)

    def raff_patch(self):
        # Issues with data being patched up
        node_list = [('139S', '83S'),
                     ('134S', '135S'),
                     ('528S', '138S'),
                     ('981S', '804S'),
                     ('604S', '15S'),
                     ('205S', '103S'),
                     ('142S', '678S'),
                     ('528S', '15S'),
                     ('447S', '158S'),
                     ('363S', '959S'),
                     ('1019S', '649S'),
                     ('771S', '892S'),
                     ('114S', '770S'),
                     ('708S', '710S'),
                     ('873S', '765S'),
                     ('134S', '336S'),
                     ('112S', '354S'),
                     ('227S', '121S'),
                     ('313S', '225S'),
                     ('893S', '893S'),
                     ('313S', '67S'),
                     ('656S', '377S'),
                     ('893S', '164S'),
                     ('771S', '774S'),
                     ('528S', '240S'),
                     ('759S', '615S'),
                     ('512S', '65S'),
                     ('914S', '163S'),
                     ('257S', '158S'),
                     ('873S', '756S'),
                     ('918S', '757S'),
                     ('656S', '329S'),
                     ('918S', '252S'),
                     ('662S', '649S'),
                     ('880S', '716S'),
                     ('254S', '315S'),
                     ('205S', '312S'),
                     ('719S', '523S'),
                     ('492S', '370S'),
                     ('764S', '757S'),
                     ('112S', '103S'),
                     ('67S', '315S'),
                     ('1045S', '801S'),
                     ('708S', '709S'),
                     ('662S', '562S'),
                     ('956S', '777S'),
                     ('114S', '948S'),
                     ('981S', '803S'),
                     ('594S', '377S'),
                     ('267S', '899S'),
                     ('142S', '237S'),
                     ('594S', '656S'),
                     ('759S', '775S'),
                     ('383S', '415S'),
                     ('363S', '932S'),
                     ('267S', '880S'),
                     ('914S', '893S'),
                     ('370S', '315S'),
                     ('435S', '504S'),
                     ('594S', '329S'),
                     ('526S', '252S'),
                     ('914S', '164S'),
                     ('819S', '777S'),
                     ('572S', '963S'),
                     ('447S', '324S'),
                     ('914S', '767S'),
                     ('757S', '252S'),
                     ('139S', '254S'),
                     ('492S', '15S'),
                     ('296S', '656S'),
                     ('112S', '312S'),
                     ('656S', '213S'),
                     ('956S', '709S'),
                     ('764S', '223S'),
                     ('773S', '771S'),
                     ('428S', '179S'),
                     ('1019S', '1024S'),
                     ('594S', '296S'),
                     ('765S', '764S'),
                     ('385S', '229S'),
                     ('379S', '415S'),
                     ('773S', '775S'),
                     ('765S', '843S'),
                     ('772S', '776S'),
                     ('383S', '103S'),
                     ('428S', '151S'),
                     ('991S', '554S'),
                     ('526S', '757S'),
                     ('991S', '356S'),
                     ('142S', '65S'),
                     ('225S', '226S'),
                     ('719S', '368S'),
                     ('528S', '418S'),
                     ('918S', '710S'),
                     ('594S', '213S'),
                     ('1019S', '251S'),
                     ('1045S', '972S'),
                     ('512S', '392S'),
                     ('765S', '757S'),
                     ('267S', '221S'),
                     ('572S', '802S'),
                     ('112S', '806S'),
                     ('764S', '137S'),
                     ('435S', '324S'),
                     ('893S', '163S'),
                     ('772S', '2037S'),
                     ('385S', '225S'),
                     ('873S', '769S'),
                     ('772S', '767S'),
                     ('322S', '103S'),
                     ('383S', '59S'),
                     ('329S', '213S'),
                     ('528S', '674S'),
                     ('604S', '928S')]

        node_list = list(set(node_list))

        for i in node_list:
            self.G.add_edge(i[0], i[1])

        self.G.remove_edges_from([(u, v) for u, v in self.G.edges() if u == v])
