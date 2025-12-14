import React from 'react';
import { Top5Prediction } from '../../../types';

interface PredictionCardProps {
  title: string;
  imageBase64: string;
  prediction: string;
  confidence: number;
  top5: Top5Prediction[];
  isChanged?: boolean;
  controls?: React.ReactNode;
}

const PredictionCard: React.FC<PredictionCardProps> = ({
  title,
  imageBase64,
  prediction,
  confidence,
  top5,
  isChanged = false,
  controls
}) => {
  return (
    <div className="image-result">
      <div className="prediction-card-header">
        {controls}
        <h4>{title}</h4>
      </div>
      <img
        src={`data:image/png;base64,${imageBase64}`}
        alt={title}
      />
      <div className="prediction">
        <span className={`label ${isChanged ? 'changed' : ''}`}>
          {prediction}
        </span>
        <span className="confidence">{confidence}%</span>
      </div>
      <div className="top5-predictions">
        <h5>Top-5 Predictions</h5>
        {top5.map((pred, idx) => (
          <div
            key={idx}
            className={`top5-item ${idx === 0 ? 'top1' : ''} ${isChanged && idx === 0 ? 'changed' : ''}`}
          >
            <span className="top5-rank">#{idx + 1}</span>
            <span className="top5-idx">[{pred.class_idx}]</span>
            <span className="top5-class">{pred.class_name}</span>
            <span className="top5-conf">{pred.confidence}%</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PredictionCard;
