import random
import pygame
from collections import deque

# Configurações visuais
# Edite CELL_SIZE e GRID_ROWS/COLS para ajustar o tamanho do labirinto
# Tamanhos padrão: 30x30 células de 20px
CELL_SIZE = 20  # diminua o tamanho para labirintos maiores
MARGIN = 2
GRID_ROWS = 30
GRID_COLS = 30
FOOTER_HEIGHT = 70  # espaço para texto
FOOTER_WIDTH = 10
FPS = 60  # taxa de frames
# passos de busca por frame (aumente para acelerar em grades grandes como 500x100)
ANIMATION_SPEED = 5
DENSITY = 0.3  # densidade de paredes ao preencher aleatoriamente o labirinto com paredes

# Cores RGB
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
GREEN = (0, 200, 0)       # início
RED = (200, 0, 0)         # fim
YELLOW = (255, 255, 0)    # visitadas
BLUE = (50, 50, 255)      # caminho final
LIGHT_BLUE = (150, 200, 255)  # nó atualmente explorado (frontier)


class MazeGUIAnimated:
    """Interface gráfica para criar labirintos e animar BFS/DFS.
    Permite definir início, fim, paredes e escolher entre BFS e DFS.
    Mostra o progresso da busca em tempo real.
    """

    def __init__(self, rows, cols):
        """Inicializa a janela e variáveis.
        rows: número de linhas do grid
        cols: número de colunas do grid
        0 = célula livre
        1 = parede
        start = (r, c) célula de início
        goal = (r, c) célula de fim
        mode = "wall", "erase", "start", "goal" (modo de edição)
        algorithm = "bfs" ou "dfs"
        path = lista de células do caminho encontrado (se houver)
        visited = conjunto de células visitadas
        frontier = lista de células na fronteira (a explorar)
        came_from = dicionário para reconstruir o caminho
        """

        # inicialização do pygame e janela
        pygame.init()
        self.rows = rows
        self.cols = cols
        width = cols * CELL_SIZE + (cols + 1) * MARGIN + FOOTER_WIDTH
        height = rows * CELL_SIZE + (rows + 1) * \
            MARGIN + FOOTER_HEIGHT  # espaço para texto
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Maze Solver Animated (BFS / DFS)")

        # estrutura do labirinto
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.start = None
        self.goal = None

        # Para a animação
        self.visited = set()
        self.came_from = {}
        self.frontier = deque()

        self.path = None

        # Algoritmo selecionado: "bfs" ou "dfs". Padrão é BFS.
        self.algorithm = "bfs"

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

    def draw(self):
        self.screen.fill(BLACK)
        # desenha células
        for r in range(self.rows):
            for c in range(self.cols):
                color = WHITE
                if self.grid[r][c] == 1:
                    color = BLACK
                cell_rect = pygame.Rect(
                    MARGIN + c * (CELL_SIZE + MARGIN),
                    MARGIN + r * (CELL_SIZE + MARGIN),
                    CELL_SIZE,
                    CELL_SIZE
                )
                pygame.draw.rect(self.screen, color, cell_rect)

        # desenhar visitadas
        for (r, c) in self.visited:
            cell_rect = pygame.Rect(
                MARGIN + c * (CELL_SIZE + MARGIN),
                MARGIN + r * (CELL_SIZE + MARGIN),
                CELL_SIZE,
                CELL_SIZE
            )
            pygame.draw.rect(self.screen, YELLOW, cell_rect)

        # desenhar fronteira atual
        for (r, c) in self.frontier:
            cell_rect = pygame.Rect(
                MARGIN + c * (CELL_SIZE + MARGIN),
                MARGIN + r * (CELL_SIZE + MARGIN),
                CELL_SIZE,
                CELL_SIZE
            )
            pygame.draw.rect(self.screen, LIGHT_BLUE, cell_rect)

        # desenhar caminho (se encontrado) – sobreposto após visitadas/fronteira
        if self.path:
            for (r, c) in self.path:
                cell_rect = pygame.Rect(
                    MARGIN + c * (CELL_SIZE + MARGIN),
                    MARGIN + r * (CELL_SIZE + MARGIN),
                    CELL_SIZE,
                    CELL_SIZE
                )
                pygame.draw.rect(self.screen, BLUE, cell_rect)

        # desenhar início e fim
        if self.start:
            (sr, sc) = self.start
            cell_rect = pygame.Rect(
                MARGIN + sc * (CELL_SIZE + MARGIN),
                MARGIN + sr * (CELL_SIZE + MARGIN),
                CELL_SIZE,
                CELL_SIZE
            )
            pygame.draw.rect(self.screen, GREEN, cell_rect)
        if self.goal:
            (gr, gc) = self.goal
            cell_rect = pygame.Rect(
                MARGIN + gc * (CELL_SIZE + MARGIN),
                MARGIN + gr * (CELL_SIZE + MARGIN),
                CELL_SIZE,
                CELL_SIZE
            )
            pygame.draw.rect(self.screen, RED, cell_rect)

        # desenha texto de estado / instrução em múltiplas linhas
        instructions = [
            f"Modo: {self.mode} | Algoritmo: {self.algorithm.upper()} | Teclas: s=start, g=goal, w=wall, e=erase, f=random walls",
            "b=BFS, d=DFS, space=executar, r=reset"
        ]

        y_text = self.rows * (CELL_SIZE + MARGIN) + 2
        for line in instructions:
            text_surface = self.font.render(line, True, WHITE)
            self.screen.blit(text_surface, (10, y_text))
            y_text += 20  # espaçamento entre linhas

        pygame.display.flip()

    def pos_from_mouse(self, pos):
        """Converte posição do mouse para coordenadas da célula."""
        x, y = pos
        for r in range(self.rows):
            for c in range(self.cols):
                cell_rect = pygame.Rect(
                    MARGIN + c * (CELL_SIZE + MARGIN),
                    MARGIN + r * (CELL_SIZE + MARGIN),
                    CELL_SIZE,
                    CELL_SIZE
                )
                if cell_rect.collidepoint(x, y):
                    return (r, c)
        return None

    def fill_random_walls(self, density=DENSITY):
        """Preenche o grid com paredes aleatórias baseado na densidade."""
        for r in range(self.rows):
            for c in range(self.cols):
                if random.random() < density:
                    self.grid[r][c] = 1

    def start_search(self):
        """Inicializa estruturas e começa a animação."""
        if self.start is None or self.goal is None:
            return
        sr, sc = self.start
        gr, gc = self.goal
        if self.grid[sr][sc] == 1 or self.grid[gr][gc] == 1:
            return

        self.visited = set()
        self.came_from = {}
        self.frontier.clear()
        self.path = None

        # Inicialização para BFS / DFS
        self.frontier.append((sr, sc))
        self.came_from[(sr, sc)] = None
        self.visited.add((sr, sc))

    def step(self):
        """Executa um passo da busca (um nó expandido)."""
        if not self.frontier:
            # acabou sem encontrar
            return False  # sinal de término sem solução

        # retirar próximo nó dependendo do algoritmo
        if self.algorithm == "bfs":
            r, c = self.frontier.popleft()
        else:  # dfs
            r, c = self.frontier.pop()

        # se chegamos ao objetivo
        if (r, c) == self.goal:
            # reconstruir caminho
            path = []
            cur = (r, c)
            while cur is not None:
                path.append(cur)
                cur = self.came_from[cur]
            path.reverse()
            self.path = path
            return False  # sinal de término com sucesso

        # expandir vizinhos
        # iterar sobre os 4 vizinhos (cima, baixo, esquerda, direita)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # nr e nc são as novas coordenadas
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr][nc] == 0 and (nr, nc) not in self.came_from:
                    self.frontier.append((nr, nc))
                    self.came_from[(nr, nc)] = (r, c)
                    self.visited.add((nr, nc))

        return True  # sinal de continuar

    def run(self):
        running = True
        self.mode = "wall"  # modos: wall, erase, start, goal
        animating = False

        while running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        # definir início
                        self.mode = "start"
                    elif event.key == pygame.K_g:
                        # definir fim
                        self.mode = "goal"
                    elif event.key == pygame.K_w:
                        # desenhar paredes
                        self.mode = "wall"
                    elif event.key == pygame.K_e:
                        # apagar paredes
                        self.mode = "erase"
                    elif event.key == pygame.K_f:
                        # preencher paredes aleatórias
                        self.fill_random_walls()
                    elif event.key == pygame.K_b:
                        # escolher BFS
                        self.algorithm = "bfs"
                    elif event.key == pygame.K_d:
                        # escolher DFS
                        self.algorithm = "dfs"
                    elif event.key == pygame.K_r:
                        # resetar tudo
                        self.grid = [[0 for _ in range(self.cols)]
                                     for _ in range(self.rows)]
                        self.start = None
                        self.goal = None
                        self.visited = set()
                        self.came_from = {}
                        self.frontier.clear()
                        self.path = None
                        animating = False
                    elif event.key == pygame.K_SPACE:
                        # inicia animação e executa busca
                        self.start_search()
                        animating = True
                elif event.type == pygame.MOUSEBUTTONDOWN and not animating:
                    pos = pygame.mouse.get_pos()
                    cell = self.pos_from_mouse(pos)
                    if cell is not None:
                        r, c = cell
                        if self.mode == "wall":
                            self.grid[r][c] = 1
                        elif self.mode == "erase":
                            self.grid[r][c] = 0
                        elif self.mode == "start":
                            self.start = (r, c)
                        elif self.mode == "goal":
                            self.goal = (r, c)

            # se estamos animando, dar um passo por frame
            if animating:
                for _ in range(ANIMATION_SPEED):  # quantidade de passos por frame
                    cont = self.step()
                    if not cont:
                        animating = False
                        break
            self.draw()  # desenha a tela

        pygame.quit()


# executar caso este arquivo seja o principal
if __name__ == "__main__":
    gui = MazeGUIAnimated(GRID_ROWS, GRID_COLS)
    gui.run()
