import sys
import unittest
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

matplotlib.use("Agg")

from nqs.graph import Chain1D, Graph, SquareLattice


class GraphHierarchyTests(unittest.TestCase):
    def test_children_inherit_from_base_graph(self) -> None:
        self.assertTrue(issubclass(Chain1D, Graph))
        self.assertTrue(issubclass(SquareLattice, Graph))

    def test_chain_1d_open_boundary_neighbors(self) -> None:
        graph = Chain1D(4, pbc=False)
        self.assertEqual(graph.shape, (4,))
        self.assertEqual(list(graph.iter_neighbor_pairs(1)), [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(graph.get_neighbors(1, 1), (0, 2))

    def test_chain_1d_periodic_neighbors(self) -> None:
        graph = Chain1D(4, pbc=True)
        self.assertEqual(len(list(graph.iter_edges("nn", 1))), 4)
        self.assertEqual(graph.get_neighbors(0, 1), (1, 3))
        self.assertEqual(graph.get_neighbors(0, 2), (2,))


class SquareLatticeTests(unittest.TestCase):
    def test_row_major_mapping(self) -> None:
        graph = SquareLattice(3, 2, pbc=False)
        self.assertEqual(graph.coord_to_index((0, 0)), 0)
        self.assertEqual(graph.coord_to_index((2, 1)), 5)
        self.assertEqual(graph.index_to_coord(4), (1, 1))

    def test_column_major_mapping(self) -> None:
        graph = SquareLattice(3, 2, pbc=False, ordering="column_major")
        self.assertEqual(graph.coord_to_index((0, 0)), 0)
        self.assertEqual(graph.coord_to_index((2, 1)), 5)
        self.assertEqual(graph.coord_to_index((1, 0)), 2)
        self.assertEqual(graph.index_to_coord(4), (2, 0))

    def test_open_boundary_edge_counts(self) -> None:
        graph = SquareLattice(2, 2, pbc=False)
        self.assertEqual(len(list(graph.iter_edges("nn", 1))), 4)
        self.assertEqual(len(list(graph.iter_edges("diag", 2))), 2)

    def test_periodic_boundary_edge_counts(self) -> None:
        graph = SquareLattice(3, 3, pbc=True)
        self.assertEqual(len(list(graph.iter_edges("nn", 1))), 18)
        self.assertEqual(len(list(graph.iter_edges("diag", 2))), 18)

    def test_edges_are_normalized_and_unique(self) -> None:
        graph = SquareLattice(3, 3, pbc=True)
        edges = list(graph.iter_edges("nn", 1)) + list(graph.iter_edges("diag", 2))
        self.assertTrue(all(edge.i < edge.j for edge in edges))
        self.assertEqual(len(edges), len(set(edges)))

    def test_neighbors_and_adjacency(self) -> None:
        graph = SquareLattice(3, 3, pbc=False)
        self.assertEqual(graph.get_neighbors(4, 1), (1, 3, 5, 7))
        self.assertEqual(graph.adjacency(2)[4], (0, 2, 6, 8))

    def test_to_networkx_keeps_neighbor_metadata(self) -> None:
        graph = SquareLattice(2, 2, pbc=False)
        nx_graph = graph.to_networkx({1: "blue", 2: "red"})

        self.assertEqual(nx_graph.number_of_nodes(), 4)
        self.assertEqual(nx_graph.number_of_edges(), 6)
        nn_orders = sorted(
            metadata["neighbor_order"]
            for _, _, metadata in nx_graph.edges(data=True)
            if metadata["color"] == "blue"
        )
        diag_orders = sorted(
            metadata["neighbor_order"]
            for _, _, metadata in nx_graph.edges(data=True)
            if metadata["color"] == "red"
        )
        self.assertEqual(nn_orders, [1, 1, 1, 1])
        self.assertEqual(diag_orders, [2, 2])

    def test_draw_returns_a_matplotlib_figure(self) -> None:
        graph = SquareLattice(4, 4, pbc=True)
        figure, axis = graph.draw(edge_specs={1: "blue", 2: "red"}, title="J1-J2")

        self.assertEqual(axis.get_title(), "J1-J2")
        self.assertGreater(len(axis.collections), 0)
        figure.clf()


if __name__ == "__main__":
    unittest.main()
