import React, { useState } from 'react';
import { AttackResponse } from '../../types';
import { Spinner, Badge } from '../common';
import {
  PredictionCard,
  PerturbationStats,
  VisualizationView,
  AdversarialViewToggle,
  AdversarialViewMode
} from './components';
import './ResultsDisplay.css';

interface ResultsDisplayProps {
  result: AttackResponse | null;
  loading: boolean;
  error: string | null;
}

type ViewMode = 'comparison' | 'heatmap' | 'difference';

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ result, loading, error }) => {
  const [viewMode, setViewMode] = useState<ViewMode>('comparison');
  const [advViewMode, setAdvViewMode] = useState<AdversarialViewMode>('final');

  if (loading) {
    return (
      <div className="results-container">
        <Spinner
          message="Generating adversarial example..."
          hint="This may take a moment for complex attacks"
        />
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

  const hasHeatmap = !!result.perturbation_heatmap_base64;
  const hasDiff = !!result.perturbation_diff_base64;
  const hasIterativeResults = result.is_iterative && (result.first_result || result.last_result);

  const getAdversarialData = () => {
    if (advViewMode === 'first' && result.first_result) {
      return {
        imageBase64: result.first_result.image_base64,
        prediction: result.first_result.prediction,
        confidence: result.first_result.confidence,
        top5: result.first_result.top5,
        stats: result.first_result.perturbation_stats,
        heatmap: result.first_result.perturbation_heatmap_base64,
        diff: result.first_result.perturbation_diff_base64
      };
    }
    if (advViewMode === 'last' && result.last_result) {
      return {
        imageBase64: result.last_result.image_base64,
        prediction: result.last_result.prediction,
        confidence: result.last_result.confidence,
        top5: result.last_result.top5,
        stats: result.last_result.perturbation_stats,
        heatmap: result.last_result.perturbation_heatmap_base64,
        diff: result.last_result.perturbation_diff_base64
      };
    }
    return {
      imageBase64: result.adversarial_image_base64,
      prediction: result.adversarial_prediction,
      confidence: result.adversarial_confidence,
      top5: result.adversarial_top5,
      stats: result.perturbation_stats,
      heatmap: result.perturbation_heatmap_base64,
      diff: result.perturbation_diff_base64
    };
  };

  const advData = getAdversarialData();
  const viewModeLabel = advViewMode !== 'final' && result.is_iterative
    ? (advViewMode === 'first' ? 'First' : 'Last') + ' Misclassification'
    : undefined;

  return (
    <div className="results-container">
      <div className="result-header">
        <h3>Attack Results</h3>
        <div className="result-meta">
          <Badge variant="model">{result.model_type.toUpperCase()}</Badge>
          <Badge variant="attack">{result.attack_method}</Badge>
        </div>
      </div>

      <div className={`attack-status ${result.attack_success ? 'success' : 'failed'}`}>
        {result.attack_success ? (
          <>
            <span className="status-icon">✓</span>
            <span>Attack Successful! Prediction changed.</span>
          </>
        ) : (
          <>
            <span className="status-icon">✗</span>
            <span>Attack Failed - Same Prediction</span>
          </>
        )}
      </div>

      {/* View Mode Tabs */}
      <div className="view-tabs">
        <button
          className={`view-tab ${viewMode === 'comparison' ? 'active' : ''}`}
          onClick={() => setViewMode('comparison')}
        >
          Comparison
        </button>
        {hasHeatmap && (
          <button
            className={`view-tab ${viewMode === 'heatmap' ? 'active' : ''}`}
            onClick={() => setViewMode('heatmap')}
          >
            Perturbation Heatmap
          </button>
        )}
        {hasDiff && (
          <button
            className={`view-tab ${viewMode === 'difference' ? 'active' : ''}`}
            onClick={() => setViewMode('difference')}
          >
            Difference
          </button>
        )}
      </div>

      {/* Comparison View */}
      {viewMode === 'comparison' && (
        <div className="images-comparison">
          <PredictionCard
            title="Original Image"
            imageBase64={result.original_image_base64}
            prediction={result.original_prediction}
            confidence={result.original_confidence}
            top5={result.original_top5}
          />

          <div className="arrow">→</div>

          <PredictionCard
            title="Adversarial Image"
            imageBase64={advData.imageBase64}
            prediction={advData.prediction}
            confidence={advData.confidence}
            top5={advData.top5}
            isChanged={result.attack_success}
            controls={
              hasIterativeResults && (
                <AdversarialViewToggle
                  currentMode={advViewMode}
                  onModeChange={setAdvViewMode}
                  hasFirst={!!result.first_result}
                  hasLast={!!result.last_result}
                />
              )
            }
          />
        </div>
      )}

      {/* Heatmap View */}
      {viewMode === 'heatmap' && advData.heatmap && (
        <VisualizationView
          type="heatmap"
          imageBase64={advData.heatmap}
          viewModeLabel={viewModeLabel}
        />
      )}

      {/* Difference View */}
      {viewMode === 'difference' && advData.diff && (
        <VisualizationView
          type="difference"
          imageBase64={advData.diff}
          viewModeLabel={viewModeLabel}
        />
      )}

      {/* Perturbation Statistics */}
      <PerturbationStats
        stats={advData.stats}
        viewModeLabel={viewModeLabel}
      />
    </div>
  );
};

export default ResultsDisplay;
