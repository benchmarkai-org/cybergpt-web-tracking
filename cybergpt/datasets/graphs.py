import networkx as nx
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from cybergpt.datasets.base import BaseNodeDataset
from cybergpt.datasets.sequences import SequenceDataset


class GraphDataset(BaseNodeDataset):
    SCALAR_METRICS = [
        "ambient_size",
        "ambient_density",
        "ambient_avg_clustering",
        "ambient_avg_degree",
        "ambient_scc",
        "ambient_graph_entropy",
        "ambient_modularity",
        "ambient_avg_branching",
    ]
    ARRAY_METRICS = [
        "sizes",
        "densities",
        "scc_counts",
        "graph_entropies",
        "modularities",
        "branching_factors",
        "graphs_per_edge_distribution",
    ]

    def __init__(self, edge_lists: list[list[tuple[str, str]]]):
        super().__init__()
        self.edge_lists = edge_lists

        for edge_list in edge_lists:
            for v1, v2 in edge_list:
                self.vocab.add(v1)
                self.vocab.add(v2)

    @classmethod
    def from_sequence_dataset(cls, sequence_dataset: SequenceDataset) -> "GraphDataset":
        edge_lists = [
            [(v1, v2) for v1, v2 in zip(sequence, sequence[1:])]
            for sequence in sequence_dataset.sequences
        ]
        return cls(edge_lists)

    def _calculate_graph_entropy(self, G: nx.DiGraph) -> float:
        """Calculate the graph entropy using the eigenvalues of the adjacency matrix."""
        if len(G) <= 1:
            return 0
        try:
            eigenvalues = np.linalg.eigvals(nx.adjacency_matrix(G).todense())
            nonzero_eigs = np.abs(eigenvalues[eigenvalues != 0])
            if len(nonzero_eigs) > 0:
                normalized = nonzero_eigs / np.sum(nonzero_eigs)
                return -np.sum(normalized * np.log2(normalized))
            return 0
        except:
            return 0

    def _calculate_modularity(self, G: nx.DiGraph) -> float:
        """Calculate the modularity of the graph using the Louvain method."""
        try:
            G_undirected = G.to_undirected()
            if len(G_undirected) <= 1:
                return 0
            communities = nx.community.louvain_communities(G_undirected)
            modularity = nx.community.modularity(G_undirected, communities)
            return modularity
        except:
            return 0

    def calculate_metrics(self) -> dict:
        sizes, densities, scc_counts = [], [], []
        graph_entropies, modularities, branching_factors = [], [], []

        ambient_G = nx.DiGraph()
        edge_repetitions = defaultdict(int)
        ambient_out_edges = defaultdict(set)

        for edge_list in tqdm(self.edge_lists, desc="Calculating graph metrics"):
            out_edges = defaultdict(set)
            G = nx.DiGraph()

            for v1, v2 in edge_list:
                G.add_edge(v1, v2)
                out_edges[v1].add(v2)
                edge_repetitions[v1 + "->" + v2] += 1
                ambient_G.add_edge(v1, v2)
                ambient_out_edges[v1].add(v2)

            sizes.append(len(G))
            if len(G) > 1:
                densities.append(nx.density(G))
                scc_counts.append(len(list(nx.strongly_connected_components(G))))

            if len(G) > 0:
                # graph_entropies.append(self._calculate_graph_entropy(G))
                modularities.append(self._calculate_modularity(G))
                branching_factors.append(
                    np.mean([len(targets) for targets in out_edges.values()])
                )

        graphs_per_edge_distribution = np.array(
            [edge_repetitions[edge] for edge in edge_repetitions]
        ) / len(self.edge_lists)

        return {
            "ambient_size": len(ambient_G),
            "ambient_density": nx.density(ambient_G),
            "ambient_avg_clustering": nx.average_clustering(ambient_G),
            "ambient_avg_degree": np.mean([d for n, d in ambient_G.degree()]),
            "ambient_scc": len(list(nx.strongly_connected_components(ambient_G))),
            # "ambient_graph_entropy": self._calculate_graph_entropy(ambient_G),
            "ambient_modularity": self._calculate_modularity(ambient_G),
            "ambient_avg_branching": np.mean(
                [len(targets) for targets in ambient_out_edges.values()]
            ),
            "sizes": sizes,
            "densities": densities,
            "scc_counts": scc_counts,
            # "graph_entropies": graph_entropies,
            "modularities": modularities,
            "branching_factors": branching_factors,
            "graphs_per_edge_distribution": graphs_per_edge_distribution,
        }

    def summary_report(self, metrics: dict):
        """Generate a summary report for the graph dataset."""
        print(f"Graph Dataset Summary Report")
        print("------------------------------")
        print(f"Total graphs: {len(self.edge_lists)}")
        print("")
        print(f"Average graph size: {np.mean(metrics['sizes'])}")
        print(f"Average graph density: {np.mean(metrics['densities'])}")
        # print(f"Average graph entropy: {np.mean(metrics['graph_entropies'])}")
        print(
            f"Average strongly connected components: {np.mean(metrics['scc_counts'])}"
        )
        print(f"Average modularity: {np.mean(metrics['modularities'])}")
        print(f"Average branching factor: {np.mean(metrics['branching_factors'])}")
        print(
            f"Average fraction of graphs per edge: {np.mean(metrics['graphs_per_edge_distribution'])}"
        )
        print("")
        print(f"Ambient size: {metrics['ambient_size']}")
        print(f"Ambient density: {metrics['ambient_density']}")
        print(f"Ambient avg clustering: {metrics['ambient_avg_clustering']}")
        print(f"Ambient avg degree: {metrics['ambient_avg_degree']}")
        print(f"Ambient strongly connected components: {metrics['ambient_scc']}")
        # print(f"Ambient graph entropy: {metrics['ambient_graph_entropy']}")
        print(f"Ambient modularity: {metrics['ambient_modularity']}")
        print(f"Ambient avg branching: {metrics['ambient_avg_branching']}")
        print("")
