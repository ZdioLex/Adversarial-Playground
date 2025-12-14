import React from 'react';
import './Spinner.css';

interface SpinnerProps {
  message?: string;
  hint?: string;
}

const Spinner: React.FC<SpinnerProps> = ({
  message = 'Loading...',
  hint
}) => {
  return (
    <div className="spinner-container">
      <div className="spinner"></div>
      <p className="spinner-message">{message}</p>
      {hint && <p className="spinner-hint">{hint}</p>}
    </div>
  );
};

export default Spinner;
