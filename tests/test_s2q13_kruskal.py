# This will work if ran from the root folder.
import sys 
sys.path.append("delivery_network")

from graph import Graph, graph_from_file
import unittest   # The test framework

class Test_Kruskal(unittest.TestCase):
    def test_network0(self):
        g = graph_from_file("input/network.00.in")
        mst = g.kruskal()
        self.assertEqual(mst.nb_nodes, g.nb_nodes)
        self.assertEqual(mst.nb_edges, g.nb_edges)
    
    def test_network1(self):
        g = graph_from_file("input/network.01.in")
        mst = g.kruskal()
        self.assertEqual(mst.nb_nodes, g.nb_nodes)
        self.assertEqual(mst.nb_edges, g.nb_edges)


if __name__ == '__main__':
    unittest.main()
