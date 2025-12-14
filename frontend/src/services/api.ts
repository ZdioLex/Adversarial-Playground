import axios from 'axios';
import type {
  AttackType,
  AttackCategory,
  AttackResponse,
  AttackParams,
  ModelType,
  ModelsInfo,
  HealthStatus,
  AttackSpecsResponse
} from '../types';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes for complex attacks like C&W
});

/**
 * Get the API endpoint based on attack category
 */
const getAttackEndpoint = (category: AttackCategory, attackType: AttackType): string => {
  switch (category) {
    case 'universal':
      return `/attack/universal/${attackType}`;
    case 'cnn':
      return `/attack/cnn/${attackType}`;
    case 'vit':
      return `/attack/vit/${attackType}`;
    default:
      return `/attack/universal/${attackType}`;
  }
};

/**
 * Perform an adversarial attack
 */
export const performAttack = async (
  category: AttackCategory,
  attackType: AttackType,
  file: File,
  modelType: ModelType,
  modelName: string,
  params: AttackParams
): Promise<AttackResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model_type', modelType);
  formData.append('model_name', modelName);

  // Add all parameters
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      formData.append(key, value.toString());
    }
  });

  const endpoint = getAttackEndpoint(category, attackType);
  const response = await api.post<AttackResponse>(endpoint, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

/**
 * Check API health status
 */
export const checkHealth = async (): Promise<HealthStatus> => {
  const response = await api.get<HealthStatus>('/health');
  return response.data;
};

/**
 * Get available models and attacks
 */
export const getModelsInfo = async (): Promise<ModelsInfo> => {
  const response = await api.get<ModelsInfo>('/models/info');
  return response.data;
};

/**
 * Compare attacks on CNN vs ViT
 */
export const compareAttacks = async (
  attackType: AttackType,
  file: File,
  params: AttackParams
): Promise<{ cnn: AttackResponse; vit: AttackResponse }> => {
  const formData = new FormData();
  formData.append('file', file);

  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      formData.append(key, value.toString());
    }
  });

  // Perform attack on both models in parallel
  const [cnnResponse, vitResponse] = await Promise.all([
    performAttack('universal', attackType as any, file, 'cnn', 'resnet50', params),
    performAttack('universal', attackType as any, file, 'vit', 'vit_base_patch16_224', params),
  ]);

  return {
    cnn: cnnResponse,
    vit: vitResponse,
  };
};

/**
 * Switch computation device between CPU and GPU
 */
export const switchDevice = async (device: 'cpu' | 'gpu'): Promise<{ success: boolean; device: string; message: string }> => {
  const response = await api.post('/device/switch', { device });
  return response.data;
};

/**
 * Get attack specifications with parameter metadata
 */
export const getAttackSpecs = async (): Promise<AttackSpecsResponse> => {
  const response = await api.get<AttackSpecsResponse>('/attacks/specs');
  return response.data;
};

/**
 * Get attack specifications for a specific category
 */
export const getAttackSpecsByCategory = async (category: AttackCategory): Promise<AttackSpecsResponse> => {
  const response = await api.get<AttackSpecsResponse>(`/attacks/specs/${category}`);
  return response.data;
};

/**
 * Get all ImageNet class names
 */
export const getImageNetClasses = async (): Promise<{ index: number; name: string }[]> => {
  const response = await api.get<{ index: number; name: string }[]>('/imagenet/classes');
  return response.data;
};

/**
 * Get a single ImageNet class name by index
 */
export const getImageNetClassName = async (classIdx: number): Promise<{ index: number; name: string }> => {
  const response = await api.get<{ index: number; name: string }>(`/imagenet/classes/${classIdx}`);
  return response.data;
};
