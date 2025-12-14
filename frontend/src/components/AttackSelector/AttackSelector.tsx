import React, { useEffect, useState } from 'react';
import {
  AttackType,
  AttackCategory,
  ModelType,
  AttackParams,
  AttackSpec,
  AttackSpecsResponse
} from '../../types';
import { getAttackSpecs } from '../../services/api';
import DynamicParamInputs from '../DynamicParamInputs';
import './AttackSelector.css';

interface AttackSelectorProps {
  category: AttackCategory;
  attackType: AttackType;
  modelType: ModelType;
  modelName: string;
  params: AttackParams;
  onCategoryChange: (category: AttackCategory) => void;
  onAttackTypeChange: (type: AttackType) => void;
  onModelTypeChange: (type: ModelType) => void;
  onModelNameChange: (name: string) => void;
  onParamsChange: (params: AttackParams) => void;
  vitAvailable: boolean;
  disabled?: boolean;
}

const CNN_MODELS = [
  { id: 'resnet50', name: 'ResNet-50' },
  { id: 'resnet18', name: 'ResNet-18' }
];

const VIT_MODELS = [
  { id: 'vit_base_patch16_224', name: 'ViT-Base' },
  { id: 'vit_small_patch16_224', name: 'ViT-Small' },
  { id: 'vit_tiny_patch16_224', name: 'ViT-Tiny' }
];

const AttackSelector: React.FC<AttackSelectorProps> = ({
  category,
  attackType,
  modelType,
  modelName,
  params,
  onCategoryChange,
  onAttackTypeChange,
  onModelTypeChange,
  onModelNameChange,
  onParamsChange,
  vitAvailable,
  disabled
}) => {
  const [attackSpecs, setAttackSpecs] = useState<AttackSpecsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSpecs = async () => {
      try {
        const specs = await getAttackSpecs();
        setAttackSpecs(specs);
      } catch (error) {
        console.error('Failed to fetch attack specs:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchSpecs();
  }, []);

  const getAttacksForCategory = (cat: AttackCategory): string[] => {
    if (!attackSpecs) return [];
    return Object.keys(attackSpecs).filter(id => attackSpecs[id].category === cat);
  };

  const getCurrentSpec = (): AttackSpec | null => {
    if (!attackSpecs) return null;
    return attackSpecs[attackType] || null;
  };

  const getDefaultParams = (attackId: string): AttackParams => {
    if (!attackSpecs || !attackSpecs[attackId]) return { epsilon: 0.03 };
    const spec = attackSpecs[attackId];
    const defaults: Record<string, number | boolean> = {};
    spec.params.forEach(p => {
      defaults[p.name] = p.default;
    });
    return defaults as unknown as AttackParams;
  };

  const handleCategoryChange = (newCategory: AttackCategory) => {
    onCategoryChange(newCategory);

    if (newCategory === 'cnn') {
      onModelTypeChange('cnn');
      onModelNameChange('resnet50');
    } else if (newCategory === 'vit') {
      onModelTypeChange('vit');
      onModelNameChange('vit_base_patch16_224');
    }

    const attacks = getAttacksForCategory(newCategory);
    if (attacks.length > 0) {
      const firstAttack = attacks[0] as AttackType;
      onAttackTypeChange(firstAttack);
      onParamsChange(getDefaultParams(firstAttack));
    }
  };

  const handleAttackChange = (newAttack: AttackType) => {
    onAttackTypeChange(newAttack);
    onParamsChange(getDefaultParams(newAttack));
  };

  const handleParamChange = (key: string, value: number | boolean) => {
    onParamsChange({ ...params, [key]: value });
  };

  const currentSpec = getCurrentSpec();

  if (loading) {
    return <div className="attack-selector">Loading attack configurations...</div>;
  }

  if (!attackSpecs) {
    return <div className="attack-selector">Failed to load attack configurations</div>;
  }

  return (
    <div className="attack-selector">
      {/* Category Selection */}
      <div className="category-tabs">
        <button
          className={`tab ${category === 'universal' ? 'active' : ''}`}
          onClick={() => handleCategoryChange('universal')}
          disabled={disabled}
        >
          Universal
        </button>
        <button
          className={`tab ${category === 'cnn' ? 'active' : ''}`}
          onClick={() => handleCategoryChange('cnn')}
          disabled={disabled}
        >
          CNN-Specific
        </button>
        <button
          className={`tab ${category === 'vit' ? 'active' : ''} ${!vitAvailable ? 'disabled' : ''}`}
          onClick={() => vitAvailable && handleCategoryChange('vit')}
          disabled={disabled || !vitAvailable}
          title={!vitAvailable ? 'ViT models not available' : ''}
        >
          ViT-Specific
        </button>
      </div>

      {/* Model Selection */}
      <div className="model-selection">
        {/* Model type selector - only for universal attacks */}
        {category === 'universal' && (
          <label>
            <span className="param-label">Target Model</span>
            <select
              value={modelType}
              onChange={(e) => {
                const newType = e.target.value as ModelType;
                onModelTypeChange(newType);
                onModelNameChange(newType === 'cnn' ? 'resnet50' : 'vit_base_patch16_224');
              }}
              disabled={disabled}
            >
              <option value="cnn">CNN (ResNet)</option>
              {vitAvailable && <option value="vit">ViT (Vision Transformer)</option>}
            </select>
          </label>
        )}

        {/* Model variant selector - always show for the appropriate model type */}
        <label>
          <span className="param-label">
            {category === 'universal' ? 'Model Variant' : `${category.toUpperCase()} Model`}
          </span>
          <select
            value={modelName}
            onChange={(e) => onModelNameChange(e.target.value)}
            disabled={disabled}
          >
            {(category === 'cnn' || (category === 'universal' && modelType === 'cnn'))
              ? CNN_MODELS.map(m => <option key={m.id} value={m.id}>{m.name}</option>)
              : VIT_MODELS.map(m => <option key={m.id} value={m.id}>{m.name}</option>)
            }
          </select>
        </label>
      </div>

      {/* Attack Selection */}
      <div className="attack-list">
        <span className="param-label">Attack Method</span>
        <div className="attack-buttons">
          {getAttacksForCategory(category).map(attackId => {
            const spec = attackSpecs[attackId];
            return (
              <button
                key={attackId}
                className={`attack-btn ${attackType === attackId ? 'active' : ''}`}
                onClick={() => handleAttackChange(attackId as AttackType)}
                disabled={disabled}
                title={spec.description}
              >
                {spec.name}
              </button>
            );
          })}
        </div>
      </div>

      {/* Attack Description */}
      {currentSpec && (
        <div className="attack-description">
          <p>{currentSpec.description}</p>
        </div>
      )}

      {/* Parameters - Dynamic rendering from API specs */}
      {currentSpec && (
        <div className="params-section">
          <h4>Parameters</h4>
          <DynamicParamInputs
            paramSpecs={currentSpec.params}
            params={params}
            onParamChange={handleParamChange}
            disabled={disabled}
          />
        </div>
      )}
    </div>
  );
};

export default AttackSelector;
