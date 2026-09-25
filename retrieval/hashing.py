"""Shared topology hashing for converted databases and retrieval queries."""
import networkx as nx

HASH_NETWORKX_VERSION = '3.4.2'


def get_hash(file, key='diffuse_tree', ignore_handles=True, dag=False):
    # NetworkX changed unlabeled WL hashes in 3.5. Keep released indexes compatible.
    if nx.__version__ != HASH_NETWORKX_VERSION:
        raise RuntimeError('PWM hash indexes require networkx==3.4.2; see the optional retrieval dependencies in requirements.txt and docs/setup.md.')
    graph = nx.DiGraph() if dag else nx.Graph()
    for node in file[key]:
        if ignore_handles and 'handle' in node['name'].lower():
            continue
        graph.add_node(node['id'])
        if node['parent'] != -1:
            graph.add_edge(node['id'], node['parent'])
    return nx.weisfeiler_lehman_graph_hash(graph)
