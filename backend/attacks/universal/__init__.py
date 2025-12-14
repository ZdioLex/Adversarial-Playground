"""Universal adversarial attacks that work on both CNN and ViT models."""

from .fgsm import fgsm_attack
from .pgd import pgd_attack
from .mim import mim_attack
from .cw import cw_attack
from .adversarial_patch import adversarial_patch_attack
from .uap import uap_attack

__all__ = ['fgsm_attack', 'pgd_attack', 'mim_attack', 'cw_attack', 'adversarial_patch_attack', 'uap_attack']
