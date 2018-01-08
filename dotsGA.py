from sfml import sf, sfml

import random
import math


"""
The implementation:
Step One: Generate the initial population of individuals randomly. (First generation)

Step Two: Evaluate the fitness of each individual in that population (time limit, sufficient fitness achieved, etc.)

Step Three: Repeat the following regenerational steps until termination:

    Select the best-fit individuals for reproduction. (Parents)
    Breed new individuals through crossover and mutation operations to give birth to offspring.
    Evaluate the individual fitness of new individuals.
    Replace least-fit population with new individuals.
Wikipedia link: https://en.wikipedia.org/wiki/Evolutionary_algorithm

#################################Options
###########General options
TIME_SPAN            --->The amount of time they have until they get all killed and replaced
current_frame         --->The current frame.
SIZE                 --->Size of the window. (Changing this may break some stuff since some calculations are hard-coded)
SHOW_STATS           --->Print the actual maximum fitness, the current frame
###########//

###########Target options
TARGET_RADIUS        --->The radius of the target
TARGET_COLOR         --->The color of the target
TARGET_POSITION      --->The target's position
###########//

###########Obstacle options
obs.position        --->The obstacle's position
OBS_COLOR            --->The color of the obstacle
OBS_HEIGHT           --->The obstacle height (thickness)
OBS_LENGTH           --->The obstacle length
###########//

###########Particles options
PARTICLE_RADIUS      --->Particle radius
PARTICLE_COLOR       --->Particle color
MUTATION_CHANCE      --->The % of having a mutation is MUTATION_CHANCE * 100 (Ex: To have a 10% MUTATION_CHANCE will be 0.1)
NR_OF_PARTICLES      --->Number of particles in the population
#################################//


####Controls
Left Arrow          --->Speed up the process
Right Arrow         --->Slow down the process
"""


# General options
TIME_SPAN = 150
current_frame = 0
SIZE = 500
MUTATION_CHANCE = 0.005
NR_OF_PARTICLES = 100
SHOW_STATS = True

# Obstacle Variables
OBS_COLOR = sf.Color.YELLOW
OBS_HEIGHT = 20
OBS_LENGTH = 200

# Obstacle
obs = sf.RectangleShape((OBS_LENGTH, OBS_HEIGHT))
obs.fill_color = OBS_COLOR
obs.position = (150, 240)  # Middle of the screen

# Target Variables
TARGET_RADIUS = 4
TARGET_COLOR = sf.Color.RED
TARGET_POSITION = (int(SIZE / 2), 20)

# Target
target = sf.CircleShape()
target.position = TARGET_POSITION
target.radius = TARGET_RADIUS
target.fill_color = TARGET_COLOR

# Particle
PARTICLE_RADIUS = 3
PARTICLE_COLOR = sf.Color(255, 255, 255, 50)


def distance(x1, y1, x2, y2):
    """
    Calculate the distance from 2 points. (X1, Y1) and (X2, Y2)
    Formula: (  (X1-X2)**2 + (Y1-Y2)**2  )**1/2
    """
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


class DNA:
    """"""

    def __init__(self, genes=[]):
        """Initialize the DNA"""

        if genes:
            self.genes = genes
        else:
            self.genes = []
            for _ in range(TIME_SPAN):
                self.genes.append(
                    (random.uniform(-1, 1), random.uniform(-1, 1)))

    def crossover(self, partner):
        """
        Take the genes from 2 parents and create 1 offspring
        """
        random_point = random.randint(0, len(self.genes))
        new_dna = [
            value if i < random_point else partner.genes[i]
            for i, value in enumerate(self.genes)
        ]
        return DNA(new_dna)

    def mutation(self):
        """Mutate the DNA. The % can be changed using the variable MUTATION_CHANCE"""

        for x in range(len(self.genes)):
            if random.random() < MUTATION_CHANCE:
                self.genes[x] = (random.uniform(-1, 1), random.uniform(-1, 1))


def p5map(n, start1, stop1, start2, stop2):
    """
    Python implementation of the p5 map function
    https://p5js.org/reference/#/p5/map
    """

    return ((n - start1) / (stop1 - start1)) * (stop2 - start2) + start2


class Particle:
    """A 'smart' particle"""

    def __init__(self, dna=[]):
        """Initialize the Particle"""

        self.pos = sfml.system.Vector2(int(SIZE / 2), SIZE)
        self.acc = sfml.system.Vector2()
        self.vel = sfml.system.Vector2()
        self.done = False
        self.crashed = False
        self.fitness = 0
        self.finish_frame = 0
        if dna:
            self.dna = dna
        else:
            self.dna = DNA()

    def apply_force(self, force):
        """Apply the force to the acceleration"""

        self.acc += force

    def update(self):
        """Update the Particle"""

        if (not self.done) and (not self.crashed):
            self.apply_force(self.dna.genes[current_frame])
            self.vel += self.acc
            self.pos += self.vel
            self.acc = sfml.system.Vector2()
            d = distance(target.position.x, target.position.y,
                         self.pos.x, self.pos.y)

            if d < 4:  # If it reached the target
                self.done = True
                self.pos = target.position
                self.finish_frame = current_frame
            # If it's outside of the screen
            if (self.pos.x > SIZE) or (self.pos.y > SIZE) or \
               (self.pos.x < 0) or (self.pos.y < 0):
                self.crashed = True
            # If it hit the obstacle
            if (self.pos.x >= obs.position.x - 1 and
                (self.pos.x <= (obs.position.x + OBS_LENGTH + 1))) and \
               (self.pos.y >= obs.position.y - 1 and
                    (self.pos.y <= (obs.position.y + OBS_HEIGHT + 1))):
                self.crashed = True

    def show(self):
        """Draw the Particle to the screen"""

        circle = sf.CircleShape()
        circle.position = self.pos
        circle.radius = PARTICLE_RADIUS
        circle.fill_color = PARTICLE_COLOR
        window.draw(circle)

    def calculate_fitness(self):
        """
        Calculate the fitness of the particle using time it took to reach the target and the distance
        """
        dist = distance(target.position.x, target.position.y,
                        self.pos.x, self.pos.y)
        self.fitness = p5map(dist, 0, SIZE, SIZE, 0)

        if self.done:
            # If it hit the target, give it a bonus
            self.fitness *= 5
            self.fitness += (TIME_SPAN - self.finish_frame)
        elif self.crashed:
            # If it hit the obstacle, give it a penalty
            self.fitness /= 5


class Population:
    """A population of Particles"""

    def __init__(self):
        """Initialize the population"""

        self.particles = []
        self.particle_size = NR_OF_PARTICLES
        self.matingPool = []

    def createPop(self):
        """Create the population"""

        for _ in range(self.particle_size):
            self.particles.append(Particle())

    def evaluate(self):
        """Evaluate each Particle and chose who is the best fit"""

        max_fit = 0
        for particle in self.particles:  # Calculate the maximum fitness
            particle.calculate_fitness()
            if particle.fitness > max_fit:
                max_fit = particle.fitness

        # print(max_fit)

        for particle in self.particles:  # Make all of them be between 0 and 1, the best being 1
            if max_fit == 0:  # Avoid division by 0
                particle.fitness = 0.00001
            else:
                particle.fitness /= max_fit

        self.matingPool = []

        # The best particle gets 100 chances of having a offspring, the rest get a lower chance
        for particle in self.particles:
            n = particle.fitness * 100
            for _ in range(int(n)):
                self.matingPool.append(particle)

    def run(self):
        """Update and show the population"""

        for particle in self.particles:
            particle.update()
            particle.show()

    def selection(self):
        """Chose 2 random parents from the mating pool and crossover them"""

        new_population = []
        for _ in range(self.particle_size):
            parentA = random.choice(self.matingPool).dna
            parentB = random.choice(self.matingPool).dna
            offspring = parentA.crossover(parentB)
            offspring.mutation()  # Give the offspring the chance of mutating
            # Add the offspring to the new Population
            new_population.append(Particle(offspring))
        # Kill all the current Population and replace it with the new (hopefully better) Population
        self.particles = new_population


pop = Population()  # Initialize the population
pop.createPop()  # Create the population

window = sf.RenderWindow(sf.VideoMode(SIZE, SIZE), "Genetic Algorithm")
window.framerate_limit = 30


while window.is_open:
    # Events
    # We use a try/except because for some reason, pySFML keeps crashing when I use the scrollWheel
    try:
        for event in window.events:
            if type(event) == sf.CloseEvent:
                window.close()
            if sf.Keyboard.is_key_pressed(sf.Keyboard.RIGHT):
                window.framerate_limit = 30
            if sf.Keyboard.is_key_pressed(sf.Keyboard.LEFT):
                window.framerate_limit = 1000
    except Exception:
        pass

    # Screen
    window.clear(sf.Color.BLACK)

    # Draw the target and the obstacle
    window.draw(target)
    window.draw(obs)

    pop.run()
    if current_frame >= 150 - 1:
        current_frame = 0
        pop.evaluate()
        pop.selection()
    else:
        current_frame += 1
    if SHOW_STATS:
        print("Current frame: {} ".format(current_frame), end='  \r')
    window.display()
