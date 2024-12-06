from math import trunc
import pygame
import neat
import time
import os
import random
import pickle

pygame.font.init()  # Initialize Pygame fonts

# Global Variables
WINDOW_WIDTH = 576
WINDOW_HEIGHT = 800
GROUND_LEVEL = 730  # Y-coordinate for the ground
generation_counter = 0  # Generation counter

# Fonts
STAT_FONT = pygame.font.SysFont('Arial', 40)

# Load Images
AIBIRD_IMAGES = [
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))
]

GREENPIPE_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
GROUND_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BACKGROUND_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

# Initialize Pygame window
GAME_WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Flappy AI Bird")


class AIBird:
    IMAGES = AIBIRD_IMAGES
    MAX_ROTATION = 25  # Maximum tilt
    ROTATION_SPEED = 20  # Rotation velocity
    ANIMATION_CYCLE = 5  # Time each bird image is displayed

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt_angle = 0  # Degrees to tilt
        self.tick_count = 0
        self.velocity = 0
        self.starting_height = self.y
        self.image_cycle_count = 0
        self.current_image = self.IMAGES[0]

    def jump(self):
        self.velocity = -10.5  # Velocity upwards
        self.tick_count = 0
        self.starting_height = self.y

    def move(self):
        self.tick_count += 1

        # Calculate displacement
        displacement = self.velocity * self.tick_count + 1.5 * self.tick_count ** 2

        # Terminal velocity
        if displacement >= 16:
            displacement = 16
        if displacement < 0:
            displacement -= 2

        self.y += displacement

        # Tilt the AI bird
        if displacement < 0 or self.y < self.starting_height + 50:
            if self.tilt_angle < self.MAX_ROTATION:
                self.tilt_angle = self.MAX_ROTATION
        else:
            if self.tilt_angle > -90:
                self.tilt_angle -= self.ROTATION_SPEED

    def draw(self, window):
        self.image_cycle_count += 1

        # Handle bird animation
        if self.image_cycle_count < self.ANIMATION_CYCLE:
            self.current_image = self.IMAGES[0]
        elif self.image_cycle_count < self.ANIMATION_CYCLE * 2:
            self.current_image = self.IMAGES[1]
        elif self.image_cycle_count < self.ANIMATION_CYCLE * 3:
            self.current_image = self.IMAGES[2]
        elif self.image_cycle_count < self.ANIMATION_CYCLE * 4:
            self.current_image = self.IMAGES[1]
        elif self.image_cycle_count == self.ANIMATION_CYCLE * 4 + 1:
            self.current_image = self.IMAGES[0]
            self.image_cycle_count = 0

        # If bird is nose diving, set to second image
        if self.tilt_angle <= -80:
            self.current_image = self.IMAGES[1]
            self.image_cycle_count = self.ANIMATION_CYCLE * 2

        # Rotate the AI bird
        rotated_image = pygame.transform.rotate(self.current_image, self.tilt_angle)
        new_rect = rotated_image.get_rect(center=self.current_image.get_rect(topleft=(self.x, self.y)).center)
        window.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.current_image)


class GreenPipe:
    GAP_HEIGHT = 250  # Gap between top and bottom pipes
    MOVEMENT_SPEED = 5  # Pipe movement speed

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top_pipe_y = 0
        self.bottom_pipe_y = 0
        self.TOP_PIPE_IMAGE = pygame.transform.flip(GREENPIPE_IMAGE, False, True)
        self.BOTTOM_PIPE_IMAGE = GREENPIPE_IMAGE

        self.passed = False
        self.set_random_height()

    def set_random_height(self):
        self.height = random.randrange(50, 400)
        self.top_pipe_y = self.height - self.TOP_PIPE_IMAGE.get_height()
        self.bottom_pipe_y = self.height + self.GAP_HEIGHT

    def move(self):
        self.x -= self.MOVEMENT_SPEED

    def draw(self, window):
        window.blit(self.TOP_PIPE_IMAGE, (self.x, self.top_pipe_y))
        window.blit(self.BOTTOM_PIPE_IMAGE, (self.x, self.bottom_pipe_y))

    def check_collision(self, ai_bird):
        ai_bird_mask = ai_bird.get_mask()
        top_pipe_mask = pygame.mask.from_surface(self.TOP_PIPE_IMAGE)
        bottom_pipe_mask = pygame.mask.from_surface(self.BOTTOM_PIPE_IMAGE)

        top_offset = (self.x - ai_bird.x, self.top_pipe_y - round(ai_bird.y))
        bottom_offset = (self.x - ai_bird.x, self.bottom_pipe_y - round(ai_bird.y))

        top_collision_point = ai_bird_mask.overlap(top_pipe_mask, top_offset)
        bottom_collision_point = ai_bird_mask.overlap(bottom_pipe_mask, bottom_offset)

        if top_collision_point or bottom_collision_point:
            return True

        return False


class MovingGround:
    MOVEMENT_SPEED = 5
    IMAGE_WIDTH = GROUND_IMAGE.get_width()
    IMAGE = GROUND_IMAGE

    def __init__(self, y):
        self.y = y
        self.first_image_x = 0
        self.second_image_x = self.IMAGE_WIDTH

    def move(self):
        self.first_image_x -= self.MOVEMENT_SPEED
        self.second_image_x -= self.MOVEMENT_SPEED

        if self.first_image_x + self.IMAGE_WIDTH < 0:
            self.first_image_x = self.second_image_x + self.IMAGE_WIDTH

        if self.second_image_x + self.IMAGE_WIDTH < 0:
            self.second_image_x = self.first_image_x + self.IMAGE_WIDTH

    def draw(self, window):
        window.blit(self.IMAGE, (self.first_image_x, self.y))
        window.blit(self.IMAGE, (self.second_image_x, self.y))


def draw_game_window(window, ai_birds, green_pipes, moving_ground, score, generation_counter, active_pipe_index):
    window.blit(BACKGROUND_IMAGE, (0, 0))

    for green_pipe in green_pipes:
        green_pipe.draw(window)

    # Display score
    score_text = STAT_FONT.render('Score: ' + str(score), 1, (255, 255, 255))
    window.blit(score_text, (WINDOW_WIDTH - score_text.get_width() - 10, 10))

    # Display generation
    generation_text = STAT_FONT.render("Generation: " + str(generation_counter), 1, (255, 255, 255))
    window.blit(generation_text, (10, 10))

    # Display alive count
    alive_text = STAT_FONT.render("Alive: " + str(len(ai_birds)), 1, (255, 255, 255))
    window.blit(alive_text, (10, 50))

    moving_ground.draw(window)
    for ai_bird in ai_birds:
        ai_bird.draw(window)

    pygame.display.update()


def main_ai_simulation(genomes, config):
    global generation_counter  # Declare as global to modify it
    generation_counter += 1

    # Lists to hold neural networks, genomes, and AI birds
    neural_networks = []
    genome_list = []
    ai_birds = []

    for genome_id, genome in genomes:
        genome.fitness = 0  # Initialize fitness
        neural_network = neat.nn.FeedForwardNetwork.create(genome, config)
        neural_networks.append(neural_network)
        ai_birds.append(AIBird(230, 350))
        genome_list.append(genome)

    moving_ground = MovingGround(GROUND_LEVEL)
    green_pipes = [GreenPipe(700)]
    score = 0

    clock = pygame.time.Clock()

    simulation_running = True
    while simulation_running and len(ai_birds) > 0:
        clock.tick(60)  # Limit to 60 FPS for smoother gameplay
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                simulation_running = False
                pygame.quit()
                quit()

        # which green pipe to use as the reference
        active_pipe_index = 0
        if len(green_pipes) > 1 and ai_birds[0].x > green_pipes[0].x + green_pipes[0].TOP_PIPE_IMAGE.get_width():
            active_pipe_index = 1

        # current and next green pipes
        current_pipe = green_pipes[active_pipe_index]
        if active_pipe_index + 1 < len(green_pipes):
            next_pipe = green_pipes[active_pipe_index + 1]
        else:
            next_pipe = GreenPipe(WINDOW_WIDTH + 100)  # Dummy green pipe far away

        for idx, ai_bird in enumerate(ai_birds):
            genome_list[idx].fitness += 0.1  # Reward for staying alive
            ai_bird.move()

            # Inputs for the neural network
            neural_inputs = (
                ai_bird.y,
                ai_bird.velocity,
                current_pipe.x - ai_bird.x,                            # Horizontal distance to current pipe
                abs(ai_bird.y - current_pipe.height),                 # Vertical distance to top pipe
                abs(ai_bird.y - current_pipe.bottom_pipe_y),          # Vertical distance to bottom pipe
                next_pipe.x - ai_bird.x,                              # Horizontal distance to the next pipe
                (current_pipe.height + current_pipe.bottom_pipe_y) / 2 - ai_bird.y  # Distance to center of the gap
            )

            output = neural_networks[idx].activate(neural_inputs)

            if output[0] > 0.5:  # Adjusted threshold for jumping
                ai_bird.jump()

        add_new_pipe = False
        pipes_to_remove = []
        for green_pipe in green_pipes:
            green_pipe.move()
            for idx, ai_bird in enumerate(ai_birds):
                if green_pipe.check_collision(ai_bird):
                    genome_list[idx].fitness -= 2  # Penalize collision
                    ai_birds.pop(idx)
                    neural_networks.pop(idx)
                    genome_list.pop(idx)
                    break  # Exit after removing a bird to avoid index errors

            if not green_pipe.passed and len(ai_birds) > 0 and green_pipe.x < ai_birds[0].x:
                green_pipe.passed = True
                add_new_pipe = True

            if green_pipe.x + green_pipe.TOP_PIPE_IMAGE.get_width() < 0:
                pipes_to_remove.append(green_pipe)

        if add_new_pipe:
            score += 1
            for genome in genome_list:
                genome.fitness += 5  # Reward passing a pipe
            green_pipes.append(GreenPipe(700))  # Ensure consistent pipe spacing

        for pipe in pipes_to_remove:
            green_pipes.remove(pipe)

        if score >= 30:
            print("Score of 30 reached. Saving best genome...")
            best_genome = max(genome_list, key=lambda g: g.fitness)
            with open("best_genome.pkl", "wb") as file:
                pickle.dump(best_genome, file)
            print("Best genome saved as 'best_genome.pkl'")
            simulation_running = False  # Stop simulation after saving

        # Remove AI birds that have hit the ground or flown too high
        ai_birds_to_remove = []
        for idx, ai_bird in enumerate(ai_birds):
            if ai_bird.y + ai_bird.current_image.get_height() > GROUND_LEVEL or ai_bird.y < 0:
                ai_birds_to_remove.append(idx)
        for idx in reversed(ai_birds_to_remove):  # Remove from the end to prevent index shifting
            ai_birds.pop(idx)
            neural_networks.pop(idx)
            genome_list.pop(idx)

        moving_ground.move()
        draw_game_window(GAME_WINDOW, ai_birds, green_pipes, moving_ground, score, generation_counter, active_pipe_index)


def run_pretrained_simulation(network):
    bird = AIBird(230, 350)
    green_pipes = [GreenPipe(700)]
    moving_ground = MovingGround(GROUND_LEVEL)
    score = 0
    run = True
    clock = pygame.time.Clock()

    while run:
        clock.tick(60)  # 60 FPS for smoother gameplay

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        # Determine which pipe to use as input
        active_pipe_index = 0
        if len(green_pipes) > 1 and bird.x > green_pipes[0].x + green_pipes[0].TOP_PIPE_IMAGE.get_width():
            active_pipe_index = 1

        current_pipe = green_pipes[active_pipe_index]
        if active_pipe_index + 1 < len(green_pipes):
            next_pipe = green_pipes[active_pipe_index + 1]
        else:
            next_pipe = GreenPipe(WINDOW_WIDTH + 100)  # Dummy green pipe far away

        # Inputs for the neural network
        neural_inputs = (
            bird.y,
            bird.velocity,
            current_pipe.x - bird.x,                            # Horizontal distance to current pipe
            abs(bird.y - current_pipe.height),                 # Vertical distance to top pipe
            abs(bird.y - current_pipe.bottom_pipe_y),          # Vertical distance to bottom pipe
            next_pipe.x - bird.x,                              # Horizontal distance to the next pipe
            (current_pipe.height + current_pipe.bottom_pipe_y) / 2 - bird.y  # Distance to center of the gap
        )

        output = network.activate(neural_inputs)

        if output[0] > 0.5:  # Threshold for jumping
            bird.jump()

        bird.move()

        # Move pipes
        add_pipe = False
        pipes_to_remove = []
        for pipe in green_pipes:
            pipe.move()
            if pipe.check_collision(bird):
                run = False

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

            if pipe.x + pipe.TOP_PIPE_IMAGE.get_width() < 0:
                pipes_to_remove.append(pipe)

        if add_pipe:
            score += 1
            green_pipes.append(GreenPipe(700))  # Ensure consistent pipe spacing

        for pipe in pipes_to_remove:
            green_pipes.remove(pipe)

        # Move ground
        moving_ground.move()

        # Check if bird has hit the ground or gone above the screen
        if bird.y + bird.current_image.get_height() >= GROUND_LEVEL or bird.y < 0:
            print("Bird has hit the ground or flown too high! Game Over.")
            run = False

        draw_game_window(GAME_WINDOW, [bird], green_pipes, moving_ground, score, generation_counter, active_pipe_index)

    print(f"Game Over! Your score: {score}")


def train_new_ai(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    population = neat.Population(config)

    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    statistics_reporter = neat.StatisticsReporter()
    population.add_reporter(statistics_reporter)
    population.add_reporter(neat.Checkpointer(10))  # Save a checkpoint every 10 generations

    # Run NEAT
    best_genome = population.run(main_ai_simulation, 100)  # Run for 100 generations

    # Save the best genome
    with open("best_genome.pkl", "wb") as file:
        pickle.dump(best_genome, file)

    print('\nBest genome:\n{!s}'.format(best_genome))


def load_pretrained_ai(config_path):
    if not os.path.exists("best_genome.pkl"):
        print("No pre-trained genome found. Please train a new AI first.")
        return

    with open("best_genome.pkl", "rb") as file:
        best_genome = pickle.load(file)

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    best_network = neat.nn.FeedForwardNetwork.create(best_genome, config)
    print("Loaded pre-trained AI. Running simulation...")

    run_pretrained_simulation(best_network)


def run_simulation(config_path):
    while True:
        choice = input("Choose an option:\n1. Train new AI\n2. Load pre-trained AI\n3. Exit\nEnter your choice: ")

        if choice == "1":
            train_new_ai(config_path)
        elif choice == "2":
            load_pretrained_ai(config_path)
        elif choice == "3":
            print("Exiting the program.")
            break


if __name__ == "__main__":
    local_directory = os.path.dirname(__file__)
    config_file_path = os.path.join(local_directory, "config-feedforward.txt")

    run_simulation(config_file_path)
