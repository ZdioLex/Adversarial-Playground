import React from 'react';
import  { AttackResponse } from '../types';

interface ResultsDisplayProps {
  result: AttackResponse | null;
  loading: boolean;
  error: string | null;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ result, loading, error }) => {
  if (loading) {
    return (
      <div className="results-container">
        <div className="loading">
          <div className="spinner"></div>
          <p>Generating adversarial example...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="results-container">
        <div className="error-message">
          <h3>Error</h3>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="results-container">
        <div className="placeholder">
          <p>Upload an image and configure attack parameters to see results</p>
        </div>
      </div>
    );
  }

  return (
    <div className="results-container">
      <h3>Attack Results</h3>

      <div className={`attack-status ${result.attack_success ? 'success' : 'failed'}`}>
        {result.attack_success ? (
          <>
            <span className="status-icon">✓</span>
            <span>Attack Successful!</span>
          </>
        ) : (
          <>
            <span className="status-icon">✗</span>
            <span>Attack Failed - Same Prediction</span>
          </>
        )}
      </div>

      <div className="images-comparison">
        <div className="image-result">
          <h4>Original Image</h4>
          <img
            src={`data:image/png;base64,${result.original_image_base64}`}
            alt="Original"
          />
          <div className="prediction">
            <span className="label">{result.original_prediction}</span>
            <span className="confidence">{result.original_confidence}%</span>
          </div>
        </div>

        <div className="arrow">→</div>

        <div className="image-result">
          <h4>Adversarial Image</h4>
          <img
            src={`data:image/png;base64,${result.adversarial_image_base64}`}
            alt="Adversarial"
          />
          <div className="prediction">
            <span className="label">{result.adversarial_prediction}</span>
            <span className="confidence">{result.adversarial_confidence}%</span>
          </div>
        </div>
      </div>

      <div className="perturbation-stats">
        <h4>Perturbation Statistics</h4>
        <div className="stats-grid">
          <div className="stat">
            <span className="stat-label">L2 Norm</span>
            <span className="stat-value">{result.perturbation_stats.l2_norm.toFixed(4)}</span>
          </div>
          <div className="stat">
            <span className="stat-label">L∞ Norm</span>
            <span className="stat-value">{result.perturbation_stats.linf_norm.toFixed(4)}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Mean Abs</span>
            <span className="stat-value">{result.perturbation_stats.mean_abs.toFixed(6)}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;
