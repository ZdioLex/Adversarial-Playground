export type AttackType = 'fgsm' | 'pgd' | 'dct';

export interface FGSMParams {
  epsilon: number;
}

export interface PGDParams {
  epsilon: number;
  alpha: number;
  steps: number;
}

export interface DCTParams {
  epsilon: number;
  freq_threshold: number;
}

export type AttackParams = FGSMParams | PGDParams | DCTParams;

export interface PerturbationStats {
  l2_norm: number;
  linf_norm: number;
  mean_abs: number;
}

export interface AttackResponse {
  original_prediction: string;
  original_confidence: number;
  original_class_idx: number;
  adversarial_prediction: string;
  adversarial_confidence: number;
  adversarial_class_idx: number;
  attack_success: boolean;
  adversarial_image_base64: string;
  original_image_base64: string;
  perturbation_stats: PerturbationStats;
}

export interface AttackConfig {
  type: AttackType;
  params: AttackParams;
}
