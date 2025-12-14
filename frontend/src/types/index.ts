export type ModelType = 'cnn' | 'vit';
export type CNNModel = 'resnet50' | 'resnet18';
export type ViTModel = 'vit_base_patch16_224' | 'vit_small_patch16_224' | 'vit_tiny_patch16_224';

export type AttackCategory = 'universal' | 'cnn' | 'vit';

export type UniversalAttack = 'fgsm' | 'pgd' | 'mim' | 'cw' | 'adversarial_patch' | 'uap';
export type CNNAttack = 'high_freq' | 'texture';
export type ViTAttack = 'tgr' | 'low_freq' | 'saga';
export type AttackType = UniversalAttack | CNNAttack | ViTAttack;

export interface BaseParams {
  epsilon: number;
}

export interface FGSMParams extends BaseParams {}

export interface PGDParams extends BaseParams {
  alpha?: number;
  steps?: number;
}

export interface MIMParams extends BaseParams {
  alpha?: number;
  steps?: number;
  decay?: number;
}

export interface CWParams {
  target_label: number;
  kappa?: number;
  steps?: number;
  lr?: number;
  c_init?: number;
  c_steps?: number;
}

export interface HighFreqParams extends BaseParams {
  freq_threshold?: number;
  steps?: number;
}

export interface UAPParams extends BaseParams {
  steps?: number;
  overshoot?: number;
}

export interface TextureParams extends BaseParams {
  patch_size?: number;
  steps?: number;
}

export interface TGRParams extends BaseParams {
  steps?: number;
  alpha?: number;
  top_k?: number;
  attn_scale?: number;
  mlp_scale?: number;
  qkv_scale?: number;
}

export interface SAGAParams extends BaseParams {
  steps?: number;
  alpha?: number;
  attention_weight?: number;
}

export interface LowFreqParams extends BaseParams {
  freq_threshold?: number;
  steps?: number;
  alpha?: number;
}

export interface AdversarialPatchParams extends BaseParams {
  target_label?: number;
  patch_size?: number;
  steps?: number;
  lr?: number;
  use_eot?: boolean;
  circular?: boolean;
}

export type AttackParams =
  | FGSMParams
  | PGDParams
  | MIMParams
  | CWParams
  | AdversarialPatchParams
  | HighFreqParams
  | UAPParams
  | TextureParams
  | TGRParams
  | SAGAParams
  | LowFreqParams;

export interface PerturbationStats {
  l2_norm: number;
  linf_norm: number;
  mean_abs: number;
}

export interface Top5Prediction {
  class_idx: number;
  confidence: number;
  class_name: string;
}

export interface AdversarialResult {
  prediction: string;
  confidence: number;
  class_idx: number;
  top5: Top5Prediction[];
  image_base64: string;
  perturbation_heatmap_base64?: string;
  perturbation_diff_base64?: string;
  perturbation_stats: PerturbationStats;
}

export interface AttackResponse {
  original_prediction: string;
  original_confidence: number;
  original_class_idx: number;
  original_top5: Top5Prediction[];
  adversarial_prediction: string;
  adversarial_confidence: number;
  adversarial_class_idx: number;
  adversarial_top5: Top5Prediction[];
  attack_success: boolean;
  adversarial_image_base64: string;
  original_image_base64: string;
  perturbation_heatmap_base64?: string;
  perturbation_diff_base64?: string;
  perturbation_stats: PerturbationStats;
  model_type: string;
  attack_method: string;
  first_result?: AdversarialResult;
  last_result?: AdversarialResult;
  is_iterative: boolean;
}

export interface AttackConfig {
  category: AttackCategory;
  type: AttackType;
  modelType: ModelType;
  modelName: string;
  params: AttackParams;
}

export interface ModelsInfo {
  cnn_models: string[];
  vit_models: string[];
  universal_attacks: string[];
  cnn_attacks: string[];
  vit_attacks: string[];
}

export interface HealthStatus {
  status: string;
  cnn_device: string;
  vit_device: string;
  cnn_loaded: boolean;
  vit_available: boolean;
  gpu_name?: string;
  gpu_memory_total?: string;
  gpu_memory_used?: string;
  cuda_version?: string;
  gpu_available: boolean;
}

export interface ParamSpec {
  name: string;
  type: 'float' | 'int' | 'bool';
  default: number | boolean;
  min?: number;
  max?: number;
  step?: number;
  description: string;
}

export interface AttackSpec {
  id: string;
  name: string;
  description: string;
  category: AttackCategory;
  params: ParamSpec[];
}

export type AttackSpecsResponse = Record<string, AttackSpec>;
