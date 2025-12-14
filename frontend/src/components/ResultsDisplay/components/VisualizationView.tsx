import React from 'react';

interface VisualizationViewProps {
  type: 'heatmap' | 'difference';
  imageBase64: string;
  viewModeLabel?: string;
}

const VisualizationView: React.FC<VisualizationViewProps> = ({
  type,
  imageBase64,
  viewModeLabel
}) => {
  const isHeatmap = type === 'heatmap';

  return (
    <div className="visualization-view">
      <h4>{isHeatmap ? 'Perturbation Heatmap' : 'Amplified Difference'}</h4>
      <p className="viz-description">
        {isHeatmap
          ? 'Brighter areas indicate larger perturbations'
          : 'Gray = no change, colored = perturbation (amplified 10x)'
        }
        {viewModeLabel && (
          <span className="view-mode-indicator"> ({viewModeLabel})</span>
        )}
      </p>
      <img
        src={`data:image/png;base64,${imageBase64}`}
        alt={isHeatmap ? 'Perturbation Heatmap' : 'Difference Image'}
        className="viz-image"
      />
    </div>
  );
};

export default VisualizationView;
