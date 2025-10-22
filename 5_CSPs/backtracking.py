import networkx as nx
import matplotlib.pyplot as plt
import json
####### INSTALE AS DEPENDENCIAS PRESENTES NO ARQUIVO requirements.txt ########

# Cores a serem usadas na coloração do grafo
GROUP_COLORS = {0: 'red',
                1: 'green',
                2: 'blue',
                3: 'purple'}

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
        # Adiciona aresta entre UFs vizinhas. Caso o no ainda não exista, ele é criado automaticamente.
        G.add_edge(uf, neighbor)


def backtracking_coloring(graph: nx.Graph, colors: list, node_index=0) -> bool:
    """ Realiza a coloração do grafo usando backtracking.
    Retorna True se a coloração for bem-sucedida, False caso contrário."""
    nodes = list(graph.nodes)
    # for node in nodes:
    #     print(f"Tentando colorir UF {node}...")
    if node_index == len(nodes):
        return True  # Todas as UFs foram coloridas com sucesso

    node = nodes[node_index]  # Pega o nó atual a ser colorido
    # Tenta cada cor disponível
    for color in colors:
        # Verifica se algum vizinho tem a mesma cor que a cor atual
        if all(graph.nodes[neighbor]['group'] != color for neighbor in graph.neighbors(node)):
            # Se não houver conflito, atribui a cor
            graph.nodes[node]['group'] = color
            if backtracking_coloring(graph, colors, node_index + 1):
                return True  # Continua para o próximo nó
            # Se não funcionar, remove a cor (backtrack)
            graph.nodes[node]['group'] = None
    return False  # Nenhuma cor válida encontrada, retorna False


# Executa o algoritmo de backtracking para colorir o grafo
color_list = list(GROUP_COLORS.keys())
success = backtracking_coloring(G, color_list)
if not success:
    print("Não foi possível colorir o grafo com as cores fornecidas.")
else:
    # imprime as cores atribuídas a cada nó e seus vizinhos
    for node in G.nodes:
        print(
            f"UF: {node}, Cor: {GROUP_COLORS[G.nodes[node]['group']]}, Vizinhos e as cores deles: " +
            ", ".join(
                [f"{neighbor}({GROUP_COLORS[G.nodes[neighbor]['group']]})" for neighbor in G.neighbors(node)])
        )

# Gera lista de cores para visualização do grafo
node_colors = [GROUP_COLORS[G.nodes[node]['group']] if G.nodes[node]
               ['group'] is not None else 'gray' for node in G.nodes]

# Desenha o grafo
pos = nx.spring_layout(G, k=5, method='energy')
nx.draw(G, pos, with_labels=True, node_color=node_colors,
        font_weight='bold', font_color='white', font_size=10)
plt.show()

# PROFESSOR: Caso não seja possível visualizar o grafo na tela, comente a linha plt.show() acima
# e descomente as linhas abaixo para salvar a imagem em um arquivo png
# plt.savefig("5_CSPs/grafo_coloring.png", dpi=200)
# print("Imagem do grafo salva em grafo_coloring.png")
