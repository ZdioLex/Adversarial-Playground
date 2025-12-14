"""CNN-specific adversarial attacks exploiting convolutional architecture properties."""

from .high_freq import high_freq_attack
from .texture import texture_attack

__all__ = ['high_freq_attack', 'texture_attack']
