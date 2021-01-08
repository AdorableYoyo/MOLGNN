import numpy as np
from scipy import special
import matplotlib.pyplot as plt
"""
x = np.linspace(-3, 3)
plt.plot(x, special.erf(x))
plt.xlabel('$x$')
plt.ylabel('$erf(x)$')
plt.show()
"""
# a function used for dynamic fusion
# We are using the CDF of a gaussian distribution, pls check the CDF curves from WIKI
# https://en.wikipedia.org/wiki/Normal_distribution
# the default curve is the orange one
def dynamic_fusion(epoch, total_epoch = 200, mu = 0, sigma_square = 5.0, original_x_range = 12):
    """
    inputs:
    epoch: the number of epoch
    total_epoch: the total epoch number. It is used to map the epoch number to the original CDF curve
    mu: check the wiki CDF figure
    sigma_square: check the wiki CDF figure
    original_x_range: in the visualization of the WIKI CDF figure, x is from -5.0 to 5.0. We are using
    from -6.0 to 6.0 to make the f(epoch) can have more time to stay at 1.0 when the epoch is approaching to 
    the total epoch.
    outputs:
    y = f(epoch) is a weights in the scope of 0 to 1.
    """
    # map the epoch to the x value shown in the WIKI CDF. For example, if we have TOTAL EPOCH = 100, we will map
    # [0 to 100] to [-6 to 6]
    x = -(original_x_range / 2) + (epoch / total_epoch)*original_x_range 
    return gaussian_cdf(x, mu, sigma_square**0.5)


def gaussian_cdf(x, mu, sigma):
    # yeah. This is the WIKI CDF function.
    return 0.5*(1 + special.erf((x - mu)/(sigma*(2**0.5))))

def vis_dynamic_curve(total_epoch=100,
                      mu=0,
                      sigma_square=5.0):
    """
    Meaning of the arguments see the function f
    """
    x = list(range(total_epoch +1 ))
    y = [f(x_, total_epoch, mu, sigma_square) for x_ in x]
    plt.plot(x, y, "#fc8d59")
    plt.xlabel('epoch')
    plt.ylabel('dynamic fusion weight')
    plt.savefig("dynamic_fusion_weights.png")
    plt.show()

if __name__ == "__main__":
   vis_dynamic_curve(total_epoch = 100, mu = 0, sigma_square = 5.0)
