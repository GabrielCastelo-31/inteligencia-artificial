# pip install geopandas shapely rtree networkx
import networkx as nx
import matplotlib.pyplot as plt
import json

GROUP_COLORS = {0: 'red',
                1: 'green',
                2: 'blue',
                3: 'purple',
                4: 'orange'}
# Json com os vizinhos de cada UF
uf_json_path = '5_CSPs/uf_neighbors.json'
json_data = None
with open(uf_json_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)


# Constrói o grafo
G = nx.Graph()
for uf in json_data:
    G.add_node(uf, group=None)
    for neighbor in json_data[uf]:
        G.add_edge(uf, neighbor)


def backtracking_coloring(graph: nx.Graph, colors: list, node_index=0) -> bool:
    # nodes = list(graph.nodes)
    # if node_index == len(nodes):
    #     return True  # Todas as UFs foram coloridas com sucesso

    # node = nodes[node_index]
    # for color in colors:
    #     # Verifica se algum vizinho tem a mesma cor
    #     if all(graph.nodes[neighbor]['group'] != color for neighbor in graph.neighbors(node)):
    #         graph.nodes[node]['group'] = color
    #         if backtracking_coloring(graph, colors, node_index + 1):
    #             return True
    #         graph.nodes[node]['group'] = None  # Backtrack

    return False


# Executa o algoritmo de coloração antes de desenhar
color_list = list(GROUP_COLORS.keys())
success = backtracking_coloring(G, color_list)
if not success:
    print("Não foi possível colorir o grafo com as cores fornecidas.")

# Gera lista de cores para cada nó
node_colors = [GROUP_COLORS[G.nodes[node]['group']] if G.nodes[node]
               ['group'] is not None else 'gray' for node in G.nodes]

pos = nx.spring_layout(G, k=1, seed=10)  # layout espaçado e reprodutível
nx.draw(G, pos, with_labels=True, node_color=node_colors,
        font_weight='bold', font_color='white', font_size=10)
# plt.tight_layout()
# plt.savefig("grafo_coloring.png", dpi=200)
print("Imagem do grafo salva em grafo_coloring.png")
plt.show()
