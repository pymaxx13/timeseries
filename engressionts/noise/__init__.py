from engressionts.noise.gaussian import GaussianNoise
from engressionts.noise.uniform import UniformNoise


NOISE_REGISTRY = {
    "gaussian": GaussianNoise,
    "uniform": UniformNoise,
}

__all__ = ["GaussianNoise", "UniformNoise", "NOISE_REGISTRY"]
