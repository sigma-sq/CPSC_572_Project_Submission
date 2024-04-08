import random
import networkx as nx
import numpy as np


class RbSim:
    def __init__(self, model, null_model):
        self.model = model
        self.null_model = null_model
        self.G = model.G
        self.GN = null_model.G
        self.RGG = null_model.rgg
        self.level = 5
        self.type = None
        self.total = 0
        self.title = None
        self.clust_coeff = []
        self.n_comp = []
        self.lg_con_comp = []
        self.short_path_per_comp = []
        self.avg_deg = []
        self.nplt = None

    def run_sim_random(self, nruns):
        crit, null_crit = self.critical()
        self.null_model.save = True

        mdl = {
            'null': [self.null_model, self.null_model.G, self.null_model.rgg, null_crit],
            'base': [self.model, self.model.G, None, crit]
        }

        for key, val in mdl.items():
            self.nplt.color = 'black'
            _avg_deg = np.round(val[0].k_avg, 2)
            _crit = val[3]
            _n = len(val[1].nodes())
            _fc = np.round(val[0].fc, 2)
            _lcc = val[0].lg_conn_comp
            _g = val[1]
            _rgg = val[2]
            _m = val[0]

            self.nplt.type = key

            for run in range(nruns):
                i = 0
                print(f'run: {run}')
                # Checks to see if we are sub-critical or it continues
                while _avg_deg > 1.2 and _lcc > 1:
                    _m.calc_graph_stats()

                    _avg_deg = np.round(_m.k_avg, 4)
                    perc_removed = np.round(self.total / _n, 2) * 100
                    print(f'{key} {i}: Fc: {np.round(_fc, 2)}, lg_con_comp: {_m.lg_conn_comp}, <k>={_avg_deg}')
                    self.nplt.filename = f'sim_{i}'

                    i += 1

                    self.random_attack(_g, _rgg)
                    # self.stats(val[0].G, self.nplt.save)
                    # avg_deg = self.avg_deg[-1]

                    _m.calc_graph_stats()
                    # lg_conn_comp = self.lg_con_comp[-1]

                    if key != 'null':
                        if _m.lg_conn_comp <= _n / 5:
                            self.nplt.color = 'red'

                        self.nplt.plot_network(
                            title=f'{key}: Rand_Sim {i}: Lg con comp({_m.lg_conn_comp}), Fc: {_fc} <k>={_avg_deg}'
                        )
                        self.nplt.log_plot(i, perc_removed)
                    else:
                        if _m.lg_conn_comp <= _n / 5:
                            self.nplt.color = 'red'

                        self.nplt.plot_null_network(
                            title=f'{key}: Rand_Sim {i}: Lg con comp({_m.lg_conn_comp}), Fc: {_fc} <k>={_avg_deg}'
                        )
                        self.nplt.null_degree_dist(i, _avg_deg)
                        # self.nplt.compare_plot(
                        #     title=f'{key}: Rand_Sim {i}: Lg con comp({_m.lg_conn_comp}), Crit Thres: {_fc}, % removed({perc_removed})'
                        # )

            self.total = 0

        self.null_model.save = False
        print(f'random attack simulation complete {nruns} runs')

    def run_sim_target_btwn(self):

        crit, null_crit = self.critical()
        self.null_model.save = True

        mdl = {
            'null': [self.null_model, self.null_model.G, self.null_model.rgg, null_crit],
            'base': [self.model, self.model.G, None, crit]
        }
        # Checks to see if we are sub-critical or it continues
        # i = 0
        # n = len(self.G.nodes())
        # fc = self.model.fc
        # lg_conn_comp = self.model.lg_conn_comp
        # btwn = nx.betweenness_centrality(self.G)
        # while self.model.k_avg > 1 and max((max(btwn.values())) > 1):
        #     perc_removed = np.round(self.total / n, 2)
        #     print(f'Crit_Thres: {fc}, % removed({perc_removed}) lg_con_comp: {lg_conn_comp}')
        #     self.nplt.filename = f'sim_{i}'
        #     i += 1
        #     print(self.nplt.filename)
        #
        #     self.targeted_attack_btwn(btwn)
        #     self.stats(self.nplt.save)
        #
        #     lg_conn_comp = self.lg_con_comp[-1]
        #
        #     if lg_conn_comp <= n / 5:
        #         self.nplt.color = 'red'
        #
        #     self.nplt.plot_network(
        #         title=f'Target_Btwn_Sim: Lg con comp({lg_conn_comp}), Crit Thres: {fc}, % removed({perc_removed})'
        #     )
        #
        #     self.nplt.log_plot(i, perc_removed)
        #
        # print(f'Targeted Betweenness attack simulation complete')
        # self.nplt.color = 'black'

        for key, val in mdl.items():
            self.nplt.color = 'black'
            _avg_deg = np.round(val[0].k_avg, 2)
            _crit = val[3]
            _n = len(val[1].nodes())
            _fc = np.round(val[0].fc, 2)
            _lcc = val[0].lg_conn_comp
            _g = val[1]
            _rgg = val[2]
            _m = val[0]
            if key == 'null':
                _btwn = nx.betweenness_centrality(_rgg)
            else:
                _btwn = nx.betweenness_centrality(_g)

            self.nplt.type = key
            # max((max(_btwn.values())) > 1)
            for run in range(1):
                i = 0
                print(f'run: {run}')
                # Checks to see if we are sub-critical or it continues
                while _avg_deg > 2 and _lcc > 1:
                    _m.calc_graph_stats()
                    if key == 'null':
                        _btwn = nx.betweenness_centrality(_rgg)
                    else:
                        _btwn = nx.betweenness_centrality(_g)
                    _avg_deg = np.round(_m.k_avg, 4)
                    perc_removed = np.round(self.total / _n, 2) * 100
                    print(
                        f'{key} {i}: Crit_Thres: {np.round(_fc, 2)}, % removed({perc_removed}) lg_con_comp: {_m.lg_conn_comp}, k_avg:{_avg_deg}')
                    self.nplt.filename = f'sim_{i}'

                    i += 1
                    self.targeted_attack_btwn(_btwn, _g, _rgg)

                    # self.random_attack(_g, _rgg)
                    # self.stats(val[0].G, self.nplt.save)
                    # avg_deg = self.avg_deg[-1]

                    _m.calc_graph_stats()
                    # lg_conn_comp = self.lg_con_comp[-1]

                    if key != 'null':
                        if _m.lg_conn_comp <= _n / 5:
                            self.nplt.color = 'red'

                        self.nplt.plot_network(
                            title=f'{key}: Targeted_Sim {i}: Lg con comp({_m.lg_conn_comp}), Fc: {_fc} <k>={_avg_deg}'
                        )
                        self.nplt.log_plot(i, perc_removed)
                    else:
                        if _m.lg_conn_comp <= _n / 5:
                            self.nplt.color = 'red'

                        self.nplt.plot_null_network(
                            title=f'{key}: Targeted_Sim {i}: Lg con comp({_m.lg_conn_comp}), Fc: {_fc} <k>={_avg_deg}'
                        )
                        self.nplt.null_degree_dist(i, _avg_deg)
                        # self.nplt.compare_plot(
                        #     title=f'{key}: Rand_Sim {i}: Lg con comp({_m.lg_conn_comp}), Crit Thres: {_fc}, % removed({perc_removed})'
                        # )

            self.total = 0
        self.null_model.save = False

    def random_attack(self, g, rgg):
        print(f'attacking: {self.nplt.type}', g)
        n_to_remove = random.sample(list(g.nodes()), self.level)

        if self.nplt.type != 'null':
            # simulate a random node attack
            # n_to_remove = random.sample(list(g.nodes()), self.level)
            g.remove_nodes_from(n_to_remove)
        else:
            print('remove g')
            g.remove_nodes_from(n_to_remove)
            print('remove rgg')
            rgg.remove_nodes_from(n_to_remove)

        self.type = "RANDOM"
        self.total = self.level + self.total
        # self.title = f"AFTER {self.type} ATTACK OF LEVEL: {self.level} TOTAL: {self.total}"

    def targeted_attack_btwn(self, btwn, g, rgg):
        # Simulate targeted nodel attack on high betweeness nodes
        # Calculate betweenness centrality for all nodes
        self.type = "TARGETED BETWEENESS"
        self.total = self.level + self.total
        # self.title = f"AFTER {self.type} ATTACK OF LEVEL: {self.level} TOTAL: {self.total}"

        print(f'attacking: {self.nplt.type}', g)
        # n_to_remove = random.sample(list(g.nodes()), self.level)

        # Sort nodes by betweenness in descending order
        sorted_nodes = sorted(btwn, key=btwn.get, reverse=True)

        # Remove the nodes with the highest betweenness centrality
        if self.nplt.type != 'null':
            for i in range(self.level):
                removed_node = sorted_nodes.pop(0)
                g.remove_node(removed_node)
        else:
            for i in range(self.level):
                removed_node = sorted_nodes.pop(0)
                g.remove_node(removed_node)
                rgg.remove_node(removed_node)

    def stats(self, g, save=False):

        clust_coeff = np.mean(list(nx.clustering(g).values()))
        number_of_components = nx.number_connected_components(g)
        largest_cc_size = len(max(nx.connected_components(g), key=len))
        avg_deg = np.average([g.degree(n) for n in g.nodes()])

        avg_path_lengths = []

        for i in nx.connected_components(g):
            subgraph = g.subgraph(i)
            avg_path_length = nx.average_shortest_path_length(subgraph)
            avg_path_lengths.append(avg_path_length)

        if not save:
            print('STATS')
            print(f'AFTER {self.type} ATTACK OF LEVEL: {self.level} TOTAL: {self.total}')

            print("Number of connected components:", number_of_components)
            print("Size of the largest connected component:", largest_cc_size)
            print(f'clustering coefficient: {clust_coeff}, shortest path per component: {avg_path_lengths}')
        else:
            self.clust_coeff.append(clust_coeff)
            self.lg_con_comp.append(largest_cc_size)
            self.n_comp.append(number_of_components)
            self.short_path_per_comp.append(avg_path_lengths)
            self.avg_deg.append(avg_deg)

    def critical(self):
        avg_deg = self.model.k_avg  # np.average([self.G.degree(n) for n in self.G.nodes()])
        avg_null_deg = self.null_model.k_avg
        crit = None
        null_crit = None
        if avg_deg >= np.log(len(self.G.nodes)):
            crit = ('Connected', 1)
        elif avg_deg > 1:
            crit = ("Supercritical", 2)
        elif avg_deg == 1:
            crit = ('Critical', 3)
        elif avg_deg < 1:
            crit = ('Subcritical', 4)

        if avg_null_deg >= np.log(len(self.G.nodes)):
            crit = ('Connected', 1)
        elif avg_null_deg > 1:
            crit = ("Supercritical", 2)
        elif avg_null_deg == 1:
            crit = ('Critical', 3)
        elif avg_null_deg < 1:
            crit = ('Subcritical', 4)

        return crit, null_crit
