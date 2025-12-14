"""
Adversarial Attack Library

Organized into three categories:
- universal: Attacks that work on both CNN and ViT models
- cnn_specific: Attacks exploiting CNN-specific vulnerabilities
- vit_specific: Attacks exploiting ViT-specific vulnerabilities
"""

# Universal attacks
from .universal import fgsm_attack, pgd_attack, mim_attack, cw_attack, adversarial_patch_attack, uap_attack

# CNN-specific attacks
from .cnn_specific import high_freq_attack, texture_attack

# ViT-specific attacks
from .vit_specific import tgr_attack, low_freq_attack, saga_attack

__all__ = [
    # Universal
    'fgsm_attack',
    'pgd_attack',
    'mim_attack',
    'cw_attack',
    'adversarial_patch_attack',
    'uap_attack',
    # CNN-specific
    'high_freq_attack',
    'texture_attack',
    # ViT-specific
    'tgr_attack',
    'low_freq_attack',
    'saga_attack',
]

# Attack registry for dynamic lookup
UNIVERSAL_ATTACKS = {
    'fgsm': fgsm_attack,
    'pgd': pgd_attack,
    'mim': mim_attack,
    'cw': cw_attack,
    'adversarial_patch': adversarial_patch_attack,
    'uap': uap_attack,
}

CNN_ATTACKS = {
    'high_freq': high_freq_attack,
    'texture': texture_attack,
}

VIT_ATTACKS = {
    'tgr': tgr_attack,
    'low_freq': low_freq_attack,
    'saga': saga_attack,
}

ALL_ATTACKS = {
    **UNIVERSAL_ATTACKS,
    **CNN_ATTACKS,
    **VIT_ATTACKS,
}
