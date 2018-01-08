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
timeSpan            --->The amount of time they have until they get all killed and replaced
currentFrame        --->The current frame.
SIZE                --->Size of the window. (Changing this may break some stuff since some calculations are hard-coded)
showStats           --->Print the actual maximum fitness, the current frame
###########//

###########Target options
targetRadius        --->The radius of the target
targetColor         --->The color of the target
targetPosition      --->The target's position
###########//

###########Obstacle options
obs.position        --->The obstacle's position
obsColor            --->The color of the obstacle
obsHeight           --->The obstacle height (thickness)
obsLength           --->The obstacle length
###########//

###########Particles options
particleRadius      --->Particle radius
particleColor       --->Particle color
mutationChance      --->The % of having a mutation is mutationChance * 100 (Ex: To have a 10% mutationChance will be 0.1)
nrOfParticles       --->Number of particles in the population
#################################//


####Controls
Left Arrow          --->Speed up the process
Right Arrow         --->Slow down the process
"""


#General options
timeSpan = 150
currentFrame = 0
SIZE = 500
mutationChance = 0.005
nrOfParticles = 100
showStats = True
#Obstacle Variables
obsColor = sf.Color.YELLOW
obsHeight = 20
obsLength = 200

#Obstacle
obs = sf.RectangleShape((obsLength, obsHeight))
obs.fill_color = obsColor
obs.position = (150, 240) #Middle of the screen

#Target Variables
targetRadius = 4
targetColor = sf.Color.RED
targetPosition = (int(SIZE/2), 20)

#Target
target = sf.CircleShape()
target.position = targetPosition
target.radius = targetRadius
target.fill_color = targetColor

#Particle
particleRadius = 3
particleColor = sf.Color(255, 255, 255, 50)

def distance(x1, y1, x2, y2):
    """
    Calculate the distance from 2 points. (X1, Y1) and (X2, Y2)
    Formula: (  (X1-X2)**2 + (Y1-Y2)**2  )**1/2
    """
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

class DNA():
    """"""
    def __init__(self, genes=[]):
        """Initialize the DNA"""

        if genes:
            self.genes = genes
        else:
            self.genes = []
            for _ in range(timeSpan):
                self.genes.append((random.uniform(-1,1), random.uniform(-1,1)))
    
    def crossover(self, partner):
        """
        Take the genes from 2 parents and create 1 offspring
        """

        newDNA = []
        #Pick one random point from the genes
        randomPoint = random.randint(0, len(self.genes))
        for x in range(len(self.genes)):
            if x < randomPoint:
                newDNA.append(self.genes[x])
            else:
                newDNA.append(partner.genes[x])
        return DNA(newDNA)

    def mutation(self):
        """Mutate the DNA. The % can be changed using the variable mutationChance"""

        for x in range(len(self.genes)):
            if random.random() < mutationChance:
                self.genes[x] = (random.uniform(-1,1), random.uniform(-1,1))

def p5map(n, start1, stop1, start2, stop2):
    """
    Python implementation of the p5 map function
    https://p5js.org/reference/#/p5/map
    """

    return ((n-start1)/(stop1-start1))*(stop2-start2)+start2

class Particle():
    """A 'smart' particle"""

    def __init__(self, dna=[]):
        """Initialize the Particle"""

        self.pos = sfml.system.Vector2(int(SIZE/2), SIZE)
        self.acc = sfml.system.Vector2()
        self.vel = sfml.system.Vector2()
        self.done = False
        self.crashed = False
        self.fitness = 0
        self.finishFrame = 0
        if dna:
            self.dna = dna
        else:
            self.dna = DNA()

    def applyForce(self, force):
        """Apply the force to the acceleration"""

        self.acc += force

    def update(self):
        """Update the Particle"""

        if (self.done == False) and (self.crashed == False):
            self.applyForce(self.dna.genes[currentFrame])
            self.vel += self.acc
            self.pos += self.vel
            self.acc = sfml.system.Vector2()
            d = distance(target.position.x, target.position.y, self.pos.x, self.pos.y)

            if d < 4: #If it reached the target
                self.done = True
                self.pos = target.position
                self.finishFrame = currentFrame
            #If it's outside of the screen
            if (self.pos.x > SIZE) or (self.pos.y > SIZE) or (self.pos.x < 0) or (self.pos.y < 0):
                self.crashed = True
            #If it hit the obstacle
            if (self.pos.x >= obs.position.x-1 and (self.pos.x <= (obs.position.x +obsLength+1))) and \
               (self.pos.y >= obs.position.y-1 and (self.pos.y <= (obs.position.y +obsHeight+1))):
               self.crashed = True

    def show(self):
        """Draw the Particle to the screen"""

        circle = sf.CircleShape()
        circle.position = self.pos
        circle.radius = particleRadius
        circle.fill_color = particleColor
        window.draw(circle)

    def calculateFitness(self):
        """
        Calculate the fitness of the particle using time it took to reach the target and the distance
        """
        dist = distance(target.position.x, target.position.y, self.pos.x, self.pos.y)
        self.fitness = p5map(dist, 0, SIZE, SIZE, 0)

        if self.done:
            #If it hit the target, give it a bonus
            self.fitness *= 5
            self.fitness += (timeSpan - self.finishFrame)
        elif self.crashed:
            #If it hit the obstacle, give it a penalty
             self.fitness /= 5

class Population(object):
    """A population of Particles"""

    def __init__(self):
        """Initialize the population"""

        self.particles = []
        self.particleSize = nrOfParticles
        self.matingPool = []

    def createPop(self):
        """Create the population"""

        for _ in range(self.particleSize):
            self.particles.append(Particle())

    def evaluate(self):
        """Evaluate each Particle and chose who is the best fit"""

        maxFit = 0
        for particle in self.particles: #Calculate the maximum fitness
            particle.calculateFitness()
            if particle.fitness > maxFit:
                maxFit = particle.fitness

        #print(maxFit) 

        for particle in self.particles: #Make all of them be between 0 and 1, the best being 1
            if maxFit == 0: #Avoid division by 0
                particle.fitness = 0.00001
            else:
                particle.fitness /= maxFit

        self.matingPool = []

        #The best particle gets 100 chances of having a offspring, the rest get a lower chance 
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

        newPopulation = []
        for _ in range(self.particleSize):
            parentA = random.choice(self.matingPool).dna
            parentB = random.choice(self.matingPool).dna
            offspring = parentA.crossover(parentB)
            offspring.mutation() #Give the offspring the chance of mutating
            newPopulation.append(Particle(offspring)) #Add the offspring to the new Population
        self.particles = newPopulation #Kill all the current Population and replace it with the new (hopefully better) Population


pop = Population() #Initialize the population
pop.createPop()    #Create the population

window = sf.RenderWindow(sf.VideoMode(SIZE, SIZE), "Genetic Algorithm")
window.framerate_limit = 30



while window.is_open:
    #Events
    #We use a try/except because for some reason, pySFML keeps crashing when I use the scrollWheel
    try:
        for event in window.events:
            if type(event) == sf.CloseEvent:
                window.close()
            if sf.Keyboard.is_key_pressed(sf.Keyboard.RIGHT):
                window.framerate_limit = 30
            if sf.Keyboard.is_key_pressed(sf.Keyboard.LEFT):
                window.framerate_limit = 1000
    except:
        pass

    # Screen
    window.clear(sf.Color.BLACK)

    #Draw the target and the obstacle
    window.draw(target)
    window.draw(obs)

    pop.run()
    if currentFrame >= 150 - 1:
        currentFrame = 0
        pop.evaluate()
        pop.selection()
    else:
        currentFrame += 1
    if showStats:
        print("Current frame: {} ".format(currentFrame), end='  \r')
    window.display()
