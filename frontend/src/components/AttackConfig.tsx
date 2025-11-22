import React from 'react';
import  { AttackType, FGSMParams, PGDParams, DCTParams } from '../types';

interface AttackConfigProps {
  attackType: AttackType;
  onAttackTypeChange: (type: AttackType) => void;
  params: FGSMParams | PGDParams | DCTParams;
  onParamsChange: (params: FGSMParams | PGDParams | DCTParams) => void;
  disabled?: boolean;
}

const AttackConfig: React.FC<AttackConfigProps> = ({
  attackType,
  onAttackTypeChange,
  params,
  onParamsChange,
  disabled
}) => {
  const handleAttackTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newType = e.target.value as AttackType;
    onAttackTypeChange(newType);

    // Set default params for new attack type
    if (newType === 'fgsm') {
      onParamsChange({ epsilon: 0.03 });
    } else if (newType === 'pgd') {
      onParamsChange({ epsilon: 0.03, alpha: 0.01, steps: 10 });
    } else if (newType === 'dct') {
      onParamsChange({ epsilon: 0.05, freq_threshold: 50 });
    }
  };

  const renderParamsInputs = () => {
    switch (attackType) {
      case 'fgsm':
        const fgsmParams = params as FGSMParams;
        return (
          <div className="param-group">
            <label>
              <span className="param-label">Epsilon (perturbation size)</span>
              <input
                type="number"
                step="0.001"
                min="0.001"
                max="1"
                value={fgsmParams.epsilon}
                onChange={(e) => onParamsChange({ epsilon: parseFloat(e.target.value) })}
                disabled={disabled}
              />
              <span className="param-hint">Recommended: 0.01 - 0.1</span>
            </label>
          </div>
        );

      case 'pgd':
        const pgdParams = params as PGDParams;
        return (
          <div className="param-group">
            <label>
              <span className="param-label">Epsilon (max perturbation)</span>
              <input
                type="number"
                step="0.001"
                min="0.001"
                max="1"
                value={pgdParams.epsilon}
                onChange={(e) => onParamsChange({
                  ...pgdParams,
                  epsilon: parseFloat(e.target.value)
                })}
                disabled={disabled}
              />
              <span className="param-hint">Recommended: 0.01 - 0.1</span>
            </label>
            <label>
              <span className="param-label">Alpha (step size)</span>
              <input
                type="number"
                step="0.001"
                min="0.001"
                max="1"
                value={pgdParams.alpha}
                onChange={(e) => onParamsChange({
                  ...pgdParams,
                  alpha: parseFloat(e.target.value)
                })}
                disabled={disabled}
              />
              <span className="param-hint">Recommended: epsilon / 4</span>
            </label>
            <label>
              <span className="param-label">Steps (iterations)</span>
              <input
                type="number"
                step="1"
                min="1"
                max="100"
                value={pgdParams.steps}
                onChange={(e) => onParamsChange({
                  ...pgdParams,
                  steps: parseInt(e.target.value)
                })}
                disabled={disabled}
              />
              <span className="param-hint">Recommended: 7 - 20</span>
            </label>
          </div>
        );

      case 'dct':
        const dctParams = params as DCTParams;
        return (
          <div className="param-group">
            <label>
              <span className="param-label">Epsilon (perturbation size)</span>
              <input
                type="number"
                step="0.001"
                min="0.001"
                max="1"
                value={dctParams.epsilon}
                onChange={(e) => onParamsChange({
                  ...dctParams,
                  epsilon: parseFloat(e.target.value)
                })}
                disabled={disabled}
              />
              <span className="param-hint">Recommended: 0.03 - 0.1</span>
            </label>
            <label>
              <span className="param-label">Frequency threshold</span>
              <input
                type="number"
                step="1"
                min="1"
                max="448"
                value={dctParams.freq_threshold}
                onChange={(e) => onParamsChange({
                  ...dctParams,
                  freq_threshold: parseInt(e.target.value)
                })}
                disabled={disabled}
              />
              <span className="param-hint">Higher = more high-freq noise (30-100)</span>
            </label>
          </div>
        );
    }
  };

  return (
    <div className="attack-config">
      <h3>Attack Configuration</h3>

      <div className="attack-type-selector">
        <label>
          <span className="param-label">Attack Type</span>
          <select
            value={attackType}
            onChange={handleAttackTypeChange}
            disabled={disabled}
          >
            <option value="fgsm">FGSM (Fast Gradient Sign Method)</option>
            <option value="pgd">PGD (Projected Gradient Descent)</option>
            <option value="dct">DCT (Frequency-based Attack)</option>
          </select>
        </label>
      </div>

      <div className="attack-description">
        {attackType === 'fgsm' && (
          <p>Single-step attack using the sign of the gradient. Fast but less effective.</p>
        )}
        {attackType === 'pgd' && (
          <p>Iterative attack with multiple gradient steps. More powerful than FGSM.</p>
        )}
        {attackType === 'dct' && (
          <p>Perturbs high-frequency DCT components. Less visible to humans.</p>
        )}
      </div>

      {renderParamsInputs()}
    </div>
  );
};

export default AttackConfig;
