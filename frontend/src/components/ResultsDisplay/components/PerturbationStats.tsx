import React from 'react';
import { PerturbationStats as PerturbationStatsType } from '../../../types';

interface PerturbationStatsProps {
  stats: PerturbationStatsType;
  viewModeLabel?: string;
}

const PerturbationStats: React.FC<PerturbationStatsProps> = ({
  stats,
  viewModeLabel
}) => {
  return (
    <div className="perturbation-stats">
      <h4>
        Perturbation Statistics
        {viewModeLabel && (
          <span className="stats-view-indicator"> ({viewModeLabel})</span>
        )}
      </h4>
      <div className="stats-grid">
        <div className="stat">
          <span className="stat-label">L2 Norm</span>
          <span className="stat-value">{stats.l2_norm.toFixed(4)}</span>
          <span className="stat-hint">Overall magnitude</span>
        </div>
        <div className="stat">
          <span className="stat-label">L∞ Norm</span>
          <span className="stat-value">{stats.linf_norm.toFixed(4)}</span>
          <span className="stat-hint">Max pixel change</span>
        </div>
        <div className="stat">
          <span className="stat-label">Mean Abs</span>
          <span className="stat-value">{stats.mean_abs.toFixed(6)}</span>
          <span className="stat-hint">Average change</span>
        </div>
      </div>
    </div>
  );
};

export default PerturbationStats;
