from numpy import random, linalg
import numpy as np

def random_ball(num_points, dimension, radius=1,center=None):
    

    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    if(center is None):
        center = np.zeros(dimension)
    else:
        center = np.array(center)
        assert len(center) == dimension

    random_directions = random.normal(size=(dimension,num_points))
    random_directions /= linalg.norm(random_directions, axis=0)
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = random.random(num_points) ** (1/dimension)
    # Return the list of random (direction & length) points.
    X =  radius * (random_directions * random_radii).T
    return X + center 
