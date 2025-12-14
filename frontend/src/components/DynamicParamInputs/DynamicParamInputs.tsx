import React, { useState, useEffect } from 'react';
import type { ParamSpec, AttackParams } from '../../types';
import { getImageNetClassName } from '../../services/api';
import './DynamicParamInputs.css';

interface DynamicParamInputsProps {
  paramSpecs: ParamSpec[];
  params: AttackParams;
  onParamChange: (key: string, value: number | boolean) => void;
  disabled?: boolean;
}

const DynamicParamInputs: React.FC<DynamicParamInputsProps> = ({
  paramSpecs,
  params,
  onParamChange,
  disabled
}) => {
  const [targetClassName, setTargetClassName] = useState<string>('');

  // Helper to safely get param value
  const getParamValue = (name: string): unknown => {
    return (params as unknown as Record<string, unknown>)[name];
  };

  useEffect(() => {
    const targetLabel = getParamValue('target_label');
    if (targetLabel !== undefined && targetLabel !== null && typeof targetLabel === 'number') {
      getImageNetClassName(targetLabel)
        .then(result => setTargetClassName(result.name))
        .catch(() => setTargetClassName(''));
    }
  }, [getParamValue('target_label')]);

  const renderInput = (spec: ParamSpec) => {
    const currentValue = getParamValue(spec.name) ?? spec.default;

    if (spec.type === 'bool') {
      return (
        <div key={spec.name} className="param-row checkbox-row">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={currentValue as boolean}
              onChange={(e) => onParamChange(spec.name, e.target.checked)}
              disabled={disabled}
            />
            <span className="param-label">{formatParamName(spec.name)}</span>
          </label>
          {spec.description && (
            <span className="param-hint">{spec.description}</span>
          )}
        </div>
      );
    }

    if (spec.name === 'epsilon') {
      return (
        <div key={spec.name} className="param-row">
          <label>
            <span className="param-label">{formatParamName(spec.name)}</span>
            <input
              type="range"
              min={spec.min ?? 0.001}
              max={spec.max ?? 0.3}
              step={spec.step ?? 0.001}
              value={currentValue as number}
              onChange={(e) => onParamChange(spec.name, parseFloat(e.target.value))}
              disabled={disabled}
            />
            <span className="param-value">{(currentValue as number).toFixed(3)}</span>
          </label>
          {spec.description && (
            <span className="param-hint">{spec.description}</span>
          )}
        </div>
      );
    }

    if (spec.name === 'target_label') {
      return (
        <div key={spec.name} className="param-row">
          <label>
            <span className="param-label">{formatParamName(spec.name)}</span>
            <input
              type="number"
              step={spec.step ?? 1}
              min={spec.min}
              max={spec.max}
              value={currentValue as number}
              onChange={(e) => {
                const val = parseInt(e.target.value);
                if (!isNaN(val) && val >= 0 && val <= 999) {
                  onParamChange(spec.name, val);
                }
              }}
              disabled={disabled}
            />
          </label>
          {targetClassName && (
            <span className="param-hint target-class-name">{targetClassName}</span>
          )}
          {!targetClassName && spec.description && (
            <span className="param-hint">{spec.description}</span>
          )}
        </div>
      );
    }

    return (
      <div key={spec.name} className="param-row">
        <label>
          <span className="param-label">{formatParamName(spec.name)}</span>
          <input
            type="number"
            step={spec.step ?? (spec.type === 'int' ? 1 : 0.001)}
            min={spec.min}
            max={spec.max}
            value={currentValue as number}
            onChange={(e) => {
              const val = spec.type === 'int'
                ? parseInt(e.target.value)
                : parseFloat(e.target.value);
              onParamChange(spec.name, val);
            }}
            disabled={disabled}
          />
        </label>
        {spec.description && (
          <span className="param-hint">{spec.description}</span>
        )}
      </div>
    );
  };

  return (
    <div className="param-inputs">
      {paramSpecs.map(renderInput)}
    </div>
  );
};

function formatParamName(name: string): string {
  const nameMap: Record<string, string> = {
    epsilon: 'Epsilon',
    alpha: 'Alpha (step size)',
    steps: 'Steps',
    decay: 'Decay (momentum)',
    target_label: 'Target Class (0-999)',
    kappa: 'Kappa (margin)',
    lr: 'Learning Rate',
    c_init: 'C Init',
    c_steps: 'C Steps (binary search)',
    patch_size: 'Patch Size',
    random_location: 'Random Location',
    freq_threshold: 'Freq Threshold',
    use_cached: 'Use Cached UAP',
    num_tokens: 'Num Tokens',
    entropy_weight: 'Entropy Weight',
    attention_threshold: 'Attention Threshold',
    attention_weight: 'Attention Weight',
    top_k: 'Top K Tokens',
    attn_scale: 'Attention Scale',
    mlp_scale: 'MLP Scale',
    qkv_scale: 'QKV Scale',
    random_start: 'Random Start'
  };
  return nameMap[name] || name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

export default DynamicParamInputs;
