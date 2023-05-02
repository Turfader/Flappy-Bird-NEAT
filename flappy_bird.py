
import pygame
import neat
import os


def run_neat(config):
    population = neat.Population(config)

    # Add reporters to show progress in the terminal and save checkpoints
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5))

    # Run the evolution for up to 100 generations
    winner = population.run(play_game, 25)

    # Show the winning genome's fitness and save it to a file
    print('\nBest genome:\n{!s}'.format(winner))
    with open('best_genome.txt', 'w') as f:
        f.write(str(winner))


class Bird:
    def __init__(self):
        self.pos = 250
        self.dy = 0
        self.jumps = 0

    def jump(self):
        self.dy -= 25
        self.jumps += 1

    def move(self):
        self.dy += 1
        self.pos += self.dy

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 0), (250, self.pos), 10)
        pygame.draw.line(screen, (0, 0, 0), (0, 50), (499, 50), 2)
        pygame.draw.line(screen, (0, 0, 0), (0, 450), (499, 450), 2)


def play_game(genomes, config):
    pygame.init()
    pygame.font.init()
    comic_sans = pygame.font.SysFont('Comic Sans MS', 10)
    size = [500, 500]
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Flappy Bird NEAT")

    for genome_id, genome in genomes:
        bird = Bird()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        done = False

        while not done:
            clock = pygame.time.Clock()
            text_surface = comic_sans.render(f"Bird pos: {bird.pos}  Bird dy: {bird.dy} "
                                             f"Genome ID: {genome_id} Jumps: {bird.jumps}"
                                             f" Fitness: {genome.fitness}", False, (0, 0, 0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    exit(0)

            inputs = (bird.pos - 50, 450 - bird.pos, bird.dy)
            output = net.activate(inputs)
            action = output.index(max(output))

            if action == 0:
                bird.jump()

            bird.move()

            #print("\r", bird.pos, bird.dy, genome_id)


            if bird.pos < 50 or bird.pos > 450:
                done = True
                genome.fitness -= 10

            screen.fill((255, 255, 255))
            bird.draw(screen)
            screen.blit(text_surface, (0, 0))
            pygame.display.flip()
            #clock.tick(100)

            genome.fitness += 1

            if genome.fitness >= 200:
                break


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)
