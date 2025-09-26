import pygame
import sys
import heapq
import math

# Initialize PyGame
pygame.init()
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Enhanced Map Pathfinder")
FONT = pygame.font.SysFont('Arial', 16)
LARGE_FONT = pygame.font.SysFont('Arial', 24, bold=True)
SMALL_FONT = pygame.font.SysFont('Arial', 14)

# Colors
BACKGROUND = (240, 240, 240)
PANEL_BG = (50, 50, 60)
PANEL_TEXT = (220, 220, 220)
NODE_COLOR = (70, 130, 180)
START_COLOR = (50, 205, 50)
END_COLOR = (220, 60, 60)
PATH_COLOR = (255, 100, 100)
EDGE_COLOR = (150, 150, 150)
BUTTON_COLOR = (80, 80, 100)
BUTTON_HOVER = (100, 100, 120)
BUTTON_TEXT = (240, 240, 240)

# Graph representation (node: {neighbor: distance})
graph = {
    'A': {'B': 5, 'C': 1, 'F': 2},
    'B': {'A': 5, 'D': 3, 'E': 2},
    'C': {'A': 1, 'D': 2, 'G': 4},
    'D': {'B': 3, 'C': 2, 'H': 1},
    'E': {'B': 2, 'F': 3, 'I': 4},
    'F': {'A': 2, 'E': 3, 'G': 1},
    'G': {'C': 4, 'F': 1, 'H': 3, 'J': 2},
    'H': {'D': 1, 'G': 3, 'I': 2},
    'I': {'E': 4, 'H': 2, 'J': 3},
    'J': {'G': 2, 'I': 3}
}

# Node positions for visualization
node_positions = {
    'A': (150, 150), 'B': (350, 200), 'C': (250, 350),
    'D': (450, 300), 'E': (550, 150), 'F': (200, 500),
    'G': (400, 450), 'H': (600, 350), 'I': (700, 200),
    'J': (750, 450)
}


# Dijkstra's Algorithm Implementation
def dijkstra(graph, start, end):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    predecessors = {}
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node == end:
            break

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    # Reconstruct path
    path = []
    current = end
    while current != start:
        path.append(current)
        current = predecessors.get(current)
        if current is None:
            return [], float('inf')  # No path exists
    path.append(start)
    return path[::-1], distances[end]


# A* Algorithm Implementation
def astar(graph, start, end, positions):
    def heuristic(a, b):
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    open_set = [(0, start)]
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == end:
            break

        for neighbor, weight in graph[current].items():
            tentative_g = g_score[current] + weight
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score, neighbor))

    # Reconstruct path
    path = []
    current = end
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            return [], float('inf')
    path.append(start)
    return path[::-1], g_score[end]


# Button class for UI
class Button:
    def __init__(self, x, y, width, height, text, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.hovered = False

    def draw(self, surface):
        color = BUTTON_HOVER if self.hovered else BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, PANEL_TEXT, self.rect, 2, border_radius=5)

        text_surf = FONT.render(self.text, True, BUTTON_TEXT)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered and self.action:
                return self.action()
        return None


# Drawing functions
def draw_graph(selected_start=None, selected_end=None, path=[], distance=0, use_astar=False):
    screen.fill(BACKGROUND)

    # Draw control panel
    panel_rect = pygame.Rect(0, 0, WIDTH, 80)
    pygame.draw.rect(screen, PANEL_BG, panel_rect)

    # Draw title
    title_text = LARGE_FONT.render("Enhanced Map Pathfinder", True, PANEL_TEXT)
    screen.blit(title_text, (20, 20))

    # Draw algorithm info
    algo_text = FONT.render(f"Algorithm: {'A*' if use_astar else 'Dijkstra'}", True, PANEL_TEXT)
    screen.blit(algo_text, (20, 50))

    # Draw distance info
    if distance < float('inf'):
        dist_text = FONT.render(f"Path Distance: {distance}", True, PANEL_TEXT)
        screen.blit(dist_text, (WIDTH - 300, 50))

    # Draw instructions
    instr_text = SMALL_FONT.render("Click on nodes to set start and end points", True, PANEL_TEXT)
    screen.blit(instr_text, (WIDTH - 300, 25))

    # Draw edges
    for node, neighbors in graph.items():
        start_pos = node_positions[node]
        for neighbor, weight in neighbors.items():
            end_pos = node_positions[neighbor]
            color = EDGE_COLOR
            pygame.draw.line(screen, color, start_pos, end_pos, 2)

            # Draw weight
            mid_x = (start_pos[0] + end_pos[0]) // 2
            mid_y = (start_pos[1] + end_pos[1]) // 2
            text = SMALL_FONT.render(str(weight), True, (100, 100, 100))
            screen.blit(text, (mid_x - 10, mid_y - 10))

    # Draw path
    if len(path) > 1:
        for i in range(len(path) - 1):
            start_pos = node_positions[path[i]]
            end_pos = node_positions[path[i + 1]]
            pygame.draw.line(screen, PATH_COLOR, start_pos, end_pos, 6)

            # Draw direction indicator
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0:
                dx, dy = dx / length, dy / length
                arrow_size = 10
                arrow_x = start_pos[0] + dx * (length - 20)
                arrow_y = start_pos[1] + dy * (length - 20)

                # Draw arrowhead
                angle = math.atan2(dy, dx)
                arrow_points = [
                    (arrow_x, arrow_y),
                    (arrow_x - arrow_size * math.cos(angle - math.pi / 6),
                     arrow_y - arrow_size * math.sin(angle - math.pi / 6)),
                    (arrow_x - arrow_size * math.cos(angle + math.pi / 6),
                     arrow_y - arrow_size * math.sin(angle + math.pi / 6))
                ]
                pygame.draw.polygon(screen, PATH_COLOR, arrow_points)

    # Draw nodes
    for node, pos in node_positions.items():
        color = NODE_COLOR
        if node == selected_start:
            color = START_COLOR
            # Draw glow effect for start node
            pygame.draw.circle(screen, (50, 205, 50, 100), pos, 30)
        if node == selected_end:
            color = END_COLOR
            # Draw glow effect for end node
            pygame.draw.circle(screen, (220, 60, 60, 100), pos, 30)

        pygame.draw.circle(screen, color, pos, 20)
        pygame.draw.circle(screen, (255, 255, 255), pos, 20, 2)

        text = FONT.render(node, True, (255, 255, 255))
        text_rect = text.get_rect(center=pos)
        screen.blit(text, text_rect)


# Main application
def main():
    selected_start = None
    selected_end = None
    current_path = []
    current_distance = 0
    use_astar = False

    # Create buttons
    clear_button = Button(WIDTH - 150, HEIGHT - 60, 120, 40, "Clear Path")
    algo_button = Button(WIDTH - 290, HEIGHT - 60, 120, 40, "Switch Algorithm")

    def clear_path():
        nonlocal selected_start, selected_end, current_path, current_distance
        selected_start = None
        selected_end = None
        current_path = []
        current_distance = 0
        return True

    def switch_algorithm():
        nonlocal use_astar, current_path, current_distance
        use_astar = not use_astar
        if selected_start and selected_end:
            if use_astar:
                current_path, current_distance = astar(graph, selected_start, selected_end, node_positions)
            else:
                current_path, current_distance = dijkstra(graph, selected_start, selected_end)
        return True

    clear_button.action = clear_path
    algo_button.action = switch_algorithm

    while True:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()

                # Check if buttons were clicked
                clear_button.handle_event(event)
                algo_button.handle_event(event)

                # Only process node clicks if not on buttons
                if pos[1] > 100:  # Below the control panel
                    for node, node_pos in node_positions.items():
                        dist = math.sqrt((pos[0] - node_pos[0]) ** 2 + (pos[1] - node_pos[1]) ** 2)
                        if dist < 20:  # Clicked on node
                            if not selected_start:
                                selected_start = node
                            elif not selected_end and node != selected_start:
                                selected_end = node
                                # Calculate path immediately after selecting end point
                                if use_astar:
                                    current_path, current_distance = astar(graph, selected_start, selected_end,
                                                                           node_positions)
                                else:
                                    current_path, current_distance = dijkstra(graph, selected_start, selected_end)
                            break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    clear_path()
                elif event.key == pygame.K_a:
                    switch_algorithm()

        # Update button hover states
        clear_button.check_hover(mouse_pos)
        algo_button.check_hover(mouse_pos)

        draw_graph(selected_start, selected_end, current_path, current_distance, use_astar)

        # Draw buttons
        clear_button.draw(screen)
        algo_button.draw(screen)

        pygame.display.flip()
        pygame.time.delay(30)


if __name__ == "__main__":
    main()