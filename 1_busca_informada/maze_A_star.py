import pygame
from collections import deque
import random
import heapq

# -----------------------------
# Configurações gerais
# -----------------------------
CELL_SIZE = 22
MARGIN = 2
GRID_ROWS = 25
GRID_COLS = 35

FPS = 120                   # taxa de frames
ANIMATION_SPEED = 10        # passos de busca por frame
FOOTER_HEIGHT = 90          # rodapé fixo

# -----------------------------
# Terrenos e cores
# -----------------------------
# Códigos de célula
EMPTY = 0   # passável, custo 1
WALL = 1   # impassável
MUD = 2   # passável, custo 3
SAND = 3   # passável, custo 5
WATER = 4   # passável, custo 8

# Cores
WHITE = (240, 240, 240)   # EMPTY
BLACK = (0, 0, 0)         # WALL
BROWN = (139, 101, 8)     # MUD
SANDY = (210, 180, 140)   # SAND
BLUE = (70, 130, 180)    # WATER

GREEN = (0, 200, 0)       # start
RED = (220, 0, 0)       # goal
YELLOW = (255, 235, 59)   # visited (expandidos)
LIGHT_BLUE = (160, 210, 255)  # frontier / open set
PATH_BLUE = (30, 60, 255)     # final path

FOOTER_BG = (40, 40, 40)
TEXT = (255, 255, 255)
TEXT_DIM = (170, 170, 170)

# mapa de custo
COST_MAP = {
    EMPTY: 1,
    WALL:  10**9,   # efetivamente impassável
    MUD:   3,
    SAND:  5,
    WATER: 8,
}

TERRAIN_COLOR = {
    EMPTY: WHITE,
    WALL:  BLACK,
    MUD:   BROWN,
    SAND:  SANDY,
    WATER: BLUE,
}

TERRAIN_NAME = {
    EMPTY: "livre",
    WALL:  "parede",
    MUD:   "lama",
    SAND:  "areia",
    WATER: "água",
}

# -----------------------------
# App
# -----------------------------


class MazeGUI:
    def __init__(self, rows, cols):
        pygame.init()
        self.rows = rows
        self.cols = cols

        width = cols * CELL_SIZE + (cols + 1) * MARGIN
        height = rows * CELL_SIZE + (rows + 1) * MARGIN + FOOTER_HEIGHT
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Labirinto com Obstáculos — BFS / DFS / A*")

        self.font = pygame.font.SysFont(None, 22)
        self.font2 = pygame.font.SysFont(None, 18)

        # grid e estado
        self.grid = [[EMPTY for _ in range(cols)] for _ in range(rows)]
        self.start = None
        self.goal = None

        # busca/anim
        self.algorithm = "bfs"          # "bfs" | "dfs" | "astar"
        self.frontier = deque()         # para bfs/dfs
        self.open_heap = []             # para A*: heap de (f, tie, (r,c))
        self.open_set_cells = set()     # para desenhar a fronteira do A*
        self.tie_counter = 0

        self.visited = set()            # nós expandidos (para colorir)
        self.came_from = {}
        self.path = None
        self.animating = False

        # A* estruturas
        self.g_score = {}
        self.f_score = {}

        # edição
        self.mode = "terrain"           # "start", "goal", "terrain", "erase"
        self.paint_terrain = WALL       # terreno atual p/ pintar quando em modo terrain
        self.mouse_down = False

        self.clock = pygame.time.Clock()

    # -------------------------
    # util
    # -------------------------
    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def passable(self, cell_value):
        return cell_value != WALL

    def cell_rect(self, r, c):
        return pygame.Rect(
            MARGIN + c * (CELL_SIZE + MARGIN),
            MARGIN + r * (CELL_SIZE + MARGIN),
            CELL_SIZE,
            CELL_SIZE
        )

    def cell_at_mouse(self, pos):
        x, y = pos
        for r in range(self.rows):
            for c in range(self.cols):
                if self.cell_rect(r, c).collidepoint(x, y):
                    return (r, c)
        return None

    def heuristic(self, a, b):
        # Manhattan
        (r1, c1), (r2, c2) = a, b
        return abs(r1 - r2) + abs(c1 - c2)

    # -------------------------
    # edição
    # -------------------------
    def handle_paint(self, pos):
        cell = self.cell_at_mouse(pos)
        if not cell:
            return
        r, c = cell
        if self.mode == "terrain":
            self.grid[r][c] = self.paint_terrain
        elif self.mode == "erase":
            self.grid[r][c] = EMPTY
        elif self.mode == "start":
            self.start = (r, c)
        elif self.mode == "goal":
            self.goal = (r, c)

        # mexeu no grid: invalida busca atual
        self.reset_search_state(soft=True)

    def randomize_terrain(self, wall_density=0.25, mud=0.1, sand=0.06, water=0.04):
        for r in range(self.rows):
            for c in range(self.cols):
                p = random.random()
                if p < wall_density:
                    self.grid[r][c] = WALL
                else:
                    q = random.random()
                    if q < water:
                        self.grid[r][c] = WATER
                    elif q < water + sand:
                        self.grid[r][c] = SAND
                    elif q < water + sand + mud:
                        self.grid[r][c] = MUD
                    else:
                        self.grid[r][c] = EMPTY
        self.reset_search_state(soft=True)

    def clear_all(self):
        self.grid = [[EMPTY for _ in range(self.cols)]
                     for _ in range(self.rows)]
        self.start = None
        self.goal = None
        self.reset_search_state(soft=False)

    def reset_search_state(self, soft=False):
        # soft: mantém grid/start/goal; limpa estruturas de busca
        self.frontier.clear()
        self.open_heap = []
        self.open_set_cells.clear()
        self.tie_counter = 0
        self.visited.clear()
        self.came_from.clear()
        self.path = None
        self.g_score.clear()
        self.f_score.clear()
        self.animating = False
        if not soft:
            self.algorithm = "bfs"

    # -------------------------
    # busca
    # -------------------------
    def start_search(self):
        if not self.start or not self.goal:
            return
        sr, sc = self.start
        gr, gc = self.goal
        if not self.in_bounds(sr, sc) or not self.in_bounds(gr, gc):
            return
        if not self.passable(self.grid[sr][sc]) or not self.passable(self.grid[gr][gc]):
            return

        # reset estruturas
        self.reset_search_state(soft=True)

        if self.algorithm in ("bfs", "dfs"):
            self.came_from[(sr, sc)] = None
            self.visited = {(sr, sc)}  # marcar origem como visitada p/ visual
            self.frontier.append((sr, sc))
        else:  # A*
            self.came_from[(sr, sc)] = None
            self.g_score[(sr, sc)] = 0
            h0 = self.heuristic((sr, sc), (gr, gc))
            self.f_score[(sr, sc)] = h0
            # heap de (f, tie, (r,c))
            heapq.heappush(self.open_heap, (h0, self.tie_counter, (sr, sc)))
            self.tie_counter += 1
            self.open_set_cells.add((sr, sc))
            # Em A*, consideramos "visitado" quando expandido (pop do heap)

        self.animating = True

    def step_search(self):
        if self.algorithm == "bfs":
            self._step_bfs()
        elif self.algorithm == "dfs":
            self._step_dfs()
        else:
            self._step_astar()

    def _neighbors4(self, r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc) and self.passable(self.grid[nr][nc]):
                yield (nr, nc)

    def _reconstruct_path(self, end_cell):
        path = []
        cur = end_cell
        while cur is not None:
            path.append(cur)
            cur = self.came_from[cur]
        self.path = list(reversed(path))
        self.animating = False

    def _step_bfs(self):
        if not self.frontier:
            self.animating = False
            return
        r, c = self.frontier.popleft()
        if (r, c) == self.goal:
            self._reconstruct_path((r, c))
            return
        # expande
        for nbr in self._neighbors4(r, c):
            if nbr not in self.came_from:
                self.came_from[nbr] = (r, c)
                self.frontier.append(nbr)
                self.visited.add(nbr)

    def _step_dfs(self):
        if not self.frontier:
            self.animating = False
            return
        r, c = self.frontier.pop()
        if (r, c) == self.goal:
            self._reconstruct_path((r, c))
            return
        # expande
        for nbr in self._neighbors4(r, c):
            if nbr not in self.came_from:
                self.came_from[nbr] = (r, c)
                self.frontier.append(nbr)
                self.visited.add(nbr)

    def _step_astar(self):
        if not self.open_heap:
            self.animating = False
            return

        # pop menor f
        f, _, (r, c) = heapq.heappop(self.open_heap)
        if (r, c) in self.open_set_cells:
            self.open_set_cells.remove((r, c))
        # agora é um nó expandido
        self.visited.add((r, c))

        if (r, c) == self.goal:
            self._reconstruct_path((r, c))
            return

        current_g = self.g_score.get((r, c), float('inf'))
        gr, gc = self.goal

        for (nr, nc) in self._neighbors4(r, c):
            # custo de mover para o vizinho
            step = COST_MAP[self.grid[nr][nc]]
            tentative_g = current_g + step

            if tentative_g < self.g_score.get((nr, nc), float('inf')):
                self.came_from[(nr, nc)] = (r, c)
                self.g_score[(nr, nc)] = tentative_g
                h = self.heuristic((nr, nc), (gr, gc))
                fn = tentative_g + h
                self.f_score[(nr, nc)] = fn
                heapq.heappush(
                    self.open_heap, (fn, self.tie_counter, (nr, nc)))
                self.tie_counter += 1
                self.open_set_cells.add((nr, nc))

    # -------------------------
    # desenho
    # -------------------------
    def draw_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.grid[r][c]
                color = TERRAIN_COLOR[val]
                pygame.draw.rect(self.screen, color, self.cell_rect(r, c))

        # fronteira / open set
        if self.algorithm in ("bfs", "dfs"):
            for (r, c) in self.frontier:
                pygame.draw.rect(self.screen, LIGHT_BLUE, self.cell_rect(r, c))
        else:
            for (r, c) in self.open_set_cells:
                pygame.draw.rect(self.screen, LIGHT_BLUE, self.cell_rect(r, c))

        # nós expandidos
        for (r, c) in self.visited:
            pygame.draw.rect(self.screen, YELLOW, self.cell_rect(r, c))

        # caminho final
        if self.path:
            for (r, c) in self.path:
                pygame.draw.rect(self.screen, PATH_BLUE, self.cell_rect(r, c))

        # start/goal por cima
        if self.start:
            pygame.draw.rect(self.screen, GREEN, self.cell_rect(*self.start))
        if self.goal:
            pygame.draw.rect(self.screen, RED, self.cell_rect(*self.goal))

    def draw_footer(self):
        top = self.rows * (CELL_SIZE + MARGIN) + MARGIN
        width = self.cols * (CELL_SIZE + MARGIN) + MARGIN
        pygame.draw.rect(self.screen, FOOTER_BG,
                         pygame.Rect(0, top, width, FOOTER_HEIGHT))

        # linha 1: estado
        t1 = self.font.render(
            f"Modo: {self.mode} | Terreno: {TERRAIN_NAME[self.paint_terrain]} | Algoritmo: {self.algorithm.upper()}",
            True, TEXT
        )
        self.screen.blit(t1, (10, top + 8))

        # linha 2: atalhos
        t2 = self.font2.render(
            "Teclas — s: início | g: objetivo | w: parede | 1: livre | 2: lama | 3: areia | 4: água | e: borracha | b: BFS | d: DFS | a: A* | espaço: executar | r: reset | f: aleatório",
            True, TEXT_DIM
        )
        self.screen.blit(t2, (10, top + 32))

        # legenda de cores
        legend_y = top + 56
        legend_items = [
            (WHITE, "livre"),
            (BLACK, "parede"),
            (BROWN, "lama"),
            (SANDY, "areia"),
            (BLUE,  "água"),
            (LIGHT_BLUE, "fronteira"),
            (YELLOW, "expandido"),
            (PATH_BLUE, "caminho"),
            (GREEN, "início"),
            (RED,   "objetivo"),
        ]

        x = 10
        for color, name in legend_items:
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(x, legend_y, 16, 16))
            self.screen.blit(self.font2.render(
                name, True, TEXT), (x + 20, legend_y))
            x += 95

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.draw_grid()
        self.draw_footer()
        pygame.display.flip()

    # -------------------------
    # loop principal
    # -------------------------
    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        self.mode = "start"
                    elif event.key == pygame.K_g:
                        self.mode = "goal"
                    elif event.key == pygame.K_e:
                        self.mode = "erase"
                    elif event.key == pygame.K_w:
                        self.mode = "terrain"
                        self.paint_terrain = WALL
                    elif event.key == pygame.K_1:
                        self.mode = "terrain"
                        self.paint_terrain = EMPTY
                    elif event.key == pygame.K_2:
                        self.mode = "terrain"
                        self.paint_terrain = MUD
                    elif event.key == pygame.K_3:
                        self.mode = "terrain"
                        self.paint_terrain = SAND
                    elif event.key == pygame.K_4:
                        self.mode = "terrain"
                        self.paint_terrain = WATER
                    elif event.key == pygame.K_b:
                        self.algorithm = "bfs"
                        self.reset_search_state(soft=True)
                    elif event.key == pygame.K_d:
                        self.algorithm = "dfs"
                        self.reset_search_state(soft=True)
                    elif event.key == pygame.K_a:
                        self.algorithm = "astar"
                        self.reset_search_state(soft=True)
                    elif event.key == pygame.K_r:
                        self.clear_all()
                    elif event.key == pygame.K_f:
                        self.randomize_terrain()
                    elif event.key == pygame.K_SPACE:
                        self.start_search()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.mouse_down = True
                    if not self.animating:
                        self.handle_paint(pygame.mouse.get_pos())

                elif event.type == pygame.MOUSEBUTTONUP:
                    self.mouse_down = False

            # pintura contínua enquanto arrasta
            if self.mouse_down and not self.animating:
                self.handle_paint(pygame.mouse.get_pos())

            # animação da busca
            if self.animating:
                for _ in range(ANIMATION_SPEED):
                    if not self.animating:
                        break
                    self.step_search()

            self.draw()

        pygame.quit()


if __name__ == "__main__":
    MazeGUI(GRID_ROWS, GRID_COLS).run()
