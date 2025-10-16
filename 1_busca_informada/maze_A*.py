import pygame
from collections import deque
import random

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
WALL  = 1   # impassável
MUD   = 2   # passável, custo 3 (para o futuro A*)
SAND  = 3   # passável, custo 5
WATER = 4   # passável, custo 8

# Cores
WHITE = (240, 240, 240)   # EMPTY
BLACK = (0, 0, 0)         # WALL
BROWN = (139, 101, 8)     # MUD
SANDY = (210, 180, 140)   # SAND
BLUE  = (70, 130, 180)    # WATER

GREEN = (0, 200, 0)       # start
RED   = (220, 0, 0)       # goal
YELLOW = (255, 235, 59)   # visited
LIGHT_BLUE = (160, 210, 255)  # frontier
PATH_BLUE = (30, 60, 255)     # final path

FOOTER_BG = (40, 40, 40)
TEXT = (255, 255, 255)
TEXT_DIM = (170, 170, 170)

# mapa de custo (usaremos quando ligar o A*)
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

        width  = cols * CELL_SIZE + (cols + 1) * MARGIN
        height = rows * CELL_SIZE + (rows + 1) * MARGIN + FOOTER_HEIGHT
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Labirinto com Obstáculos — BFS / DFS (A* em breve)")

        self.font  = pygame.font.SysFont(None, 22)
        self.font2 = pygame.font.SysFont(None, 18)

        # grid e estado
        self.grid = [[EMPTY for _ in range(cols)] for _ in range(rows)]
        self.start = None
        self.goal = None

        # busca/anim
        self.algorithm = "bfs"          # "bfs" ou "dfs"
        self.frontier = deque()
        self.visited = set()
        self.came_from = {}
        self.path = None
        self.animating = False

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

        # se mexeu no grid, invalida caminho e visita
        self.visited.clear()
        self.frontier.clear()
        self.came_from.clear()
        self.path = None
        self.animating = False

    def randomize_terrain(self, wall_density=0.25, mud=0.1, sand=0.06, water=0.04):
        for r in range(self.rows):
            for c in range(self.cols):
                p = random.random()
                if p < wall_density:
                    self.grid[r][c] = WALL
                else:
                    # escolhe passável variado
                    q = random.random()
                    if q < water:
                        self.grid[r][c] = WATER
                    elif q < water + sand:
                        self.grid[r][c] = SAND
                    elif q < water + sand + mud:
                        self.grid[r][c] = MUD
                    else:
                        self.grid[r][c] = EMPTY

    def clear_all(self):
        self.grid = [[EMPTY for _ in range(self.cols)] for _ in range(self.rows)]
        self.start = None
        self.goal = None
        self.frontier.clear()
        self.visited.clear()
        self.came_from.clear()
        self.path = None
        self.animating = False

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

        self.frontier.clear()
        self.visited = { (sr, sc) }
        self.came_from = { (sr, sc): None }
        self.path = None
        # fila/pilha
        self.frontier.append((sr, sc))
        self.animating = True

    def step_search(self):
        if not self.frontier:
            # acabou e não encontrou
            self.animating = False
            return

        if self.algorithm == "bfs":
            r, c = self.frontier.popleft()
        else:  # dfs
            r, c = self.frontier.pop()

        if (r, c) == self.goal:
            # reconstrói caminho
            path = []
            cur = (r, c)
            while cur is not None:
                path.append(cur)
                cur = self.came_from[cur]
            self.path = list(reversed(path))
            self.animating = False
            return

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc) and self.passable(self.grid[nr][nc]):
                if (nr, nc) not in self.came_from:
                    self.came_from[(nr, nc)] = (r, c)
                    self.frontier.append((nr, nc))
                    self.visited.add((nr, nc))

    # -------------------------
    # desenho
    # -------------------------
    def draw_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.grid[r][c]
                color = TERRAIN_COLOR[val]
                pygame.draw.rect(self.screen, color, self.cell_rect(r, c))

        # fronteira e visitados
        for (r, c) in self.visited:
            pygame.draw.rect(self.screen, YELLOW, self.cell_rect(r, c))
        for (r, c) in self.frontier:
            pygame.draw.rect(self.screen, LIGHT_BLUE, self.cell_rect(r, c))

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
        pygame.draw.rect(self.screen, FOOTER_BG, pygame.Rect(0, top, width, FOOTER_HEIGHT))

        # linha 1: estado
        t1 = self.font.render(
            f"Modo: {self.mode} | Terreno: {TERRAIN_NAME[self.paint_terrain]} | Algoritmo: {self.algorithm.upper()}",
            True, TEXT
        )
        self.screen.blit(t1, (10, top + 8))

        # linha 2: atalhos
        t2 = self.font2.render(
            "Teclas — s: início | g: objetivo | w: parede | 1: livre | 2: lama | 3: areia | 4: água | e: borracha | b: BFS | d: DFS | espaço: executar | r: reset | f: aleatório",
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
            (YELLOW, "visitado"),
            (LIGHT_BLUE, "fronteira"),
            (PATH_BLUE, "caminho"),
            (GREEN, "início"),
            (RED,   "objetivo"),
        ]

        x = 10
        for color, name in legend_items:
            pygame.draw.rect(self.screen, color, pygame.Rect(x, legend_y, 16, 16))
            self.screen.blit(self.font2.render(name, True, TEXT), (x + 20, legend_y))
            x += 95  # espaçamento

    def draw(self):
        self.screen.fill((0,0,0))
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
                    elif event.key == pygame.K_d:
                        self.algorithm = "dfs"
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
