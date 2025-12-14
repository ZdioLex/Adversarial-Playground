import React from 'react';

export type AdversarialViewMode = 'final' | 'first' | 'last';

interface AdversarialViewToggleProps {
  currentMode: AdversarialViewMode;
  onModeChange: (mode: AdversarialViewMode) => void;
  hasFirst: boolean;
  hasLast: boolean;
}

const AdversarialViewToggle: React.FC<AdversarialViewToggleProps> = ({
  currentMode,
  onModeChange,
  hasFirst,
  hasLast
}) => {
  const getLabel = (): string => {
    switch (currentMode) {
      case 'first': return 'First Misclassification';
      case 'last': return 'Last Misclassification';
      default: return '\u00A0';
    }
  };

  return (
    <div className="adv-controls">
      <div className={`adv-view-label ${currentMode === 'final' ? 'hidden' : ''}`}>
        {getLabel()}
      </div>
      <div className="adv-view-toggle">
        <button
          className={`adv-toggle-btn ${currentMode === 'final' ? 'active' : ''}`}
          onClick={() => onModeChange('final')}
          title="Final result after all iterations"
        >
          Final
        </button>
        {hasFirst && (
          <button
            className={`adv-toggle-btn ${currentMode === 'first' ? 'active' : ''}`}
            onClick={() => onModeChange('first')}
            title="First successful misclassification"
          >
            First
          </button>
        )}
        {hasLast && (
          <button
            className={`adv-toggle-btn ${currentMode === 'last' ? 'active' : ''}`}
            onClick={() => onModeChange('last')}
            title="Last successful misclassification"
          >
            Last
          </button>
        )}
      </div>
    </div>
  );
};

export default AdversarialViewToggle;
