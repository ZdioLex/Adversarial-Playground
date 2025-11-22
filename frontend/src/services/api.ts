import axios from 'axios';
import type { AttackType, AttackResponse, FGSMParams, PGDParams, DCTParams } from '../types';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds for attack generation
});

export const performAttack = async (
  attackType: AttackType,
  file: File,
  params: FGSMParams | PGDParams | DCTParams
): Promise<AttackResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  // Add parameters based on attack type
  if (attackType === 'fgsm') {
    const fgsmParams = params as FGSMParams;
    formData.append('epsilon', fgsmParams.epsilon.toString());
  } else if (attackType === 'pgd') {
    const pgdParams = params as PGDParams;
    formData.append('epsilon', pgdParams.epsilon.toString());
    formData.append('alpha', pgdParams.alpha.toString());
    formData.append('steps', pgdParams.steps.toString());
  } else if (attackType === 'dct') {
    const dctParams = params as DCTParams;
    formData.append('epsilon', dctParams.epsilon.toString());
    formData.append('freq_threshold', dctParams.freq_threshold.toString());
  }

  const response = await api.post<AttackResponse>(`/attack/${attackType}`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

export const checkHealth = async (): Promise<{ status: string; device: string; model_loaded: boolean }> => {
  const response = await api.get('/health');
  return response.data;
};
