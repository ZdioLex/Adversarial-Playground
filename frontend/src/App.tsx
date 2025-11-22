import { useState, useEffect } from 'react';
import { ImageUpload, AttackConfig, ResultsDisplay } from './components';
import { performAttack, checkHealth } from './services/api';
import { AttackType, AttackResponse, FGSMParams, PGDParams, DCTParams } from './types';
import './App.css';

function App() {
  // State for image
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  // State for attack configuration
  const [attackType, setAttackType] = useState<AttackType>('fgsm');
  const [params, setParams] = useState<FGSMParams | PGDParams | DCTParams>({
    epsilon: 0.03
  });

  // State for results
  const [result, setResult] = useState<AttackResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // State for API health
  const [apiStatus, setApiStatus] = useState<{
    connected: boolean;
    device: string;
  }>({ connected: false, device: 'unknown' });

  // Check API health on mount
  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        const health = await checkHealth();
        setApiStatus({
          connected: health.model_loaded,
          device: health.device
        });
      } catch {
        setApiStatus({ connected: false, device: 'error' });
      }
    };

    checkApiHealth();
    // Check every 30 seconds
    const interval = setInterval(checkApiHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleImageSelect = (file: File, preview: string) => {
    setSelectedFile(file);
    setImagePreview(preview);
    setResult(null);
    setError(null);
  };

  const handleAttack = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await performAttack(attackType, selectedFile, params);
      setResult(response);
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'response' in err) {
        const axiosError = err as { response?: { data?: { detail?: string } } };
        setError(axiosError.response?.data?.detail || 'Attack failed. Please try again.');
      } else {
        setError('Network error. Is the backend running?');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Adversarial Playground</h1>
        <p className="subtitle">
          Explore adversarial attacks on ResNet18 image classification
        </p>
        <div className={`api-status ${apiStatus.connected ? 'connected' : 'disconnected'}`}>
          <span className="status-dot"></span>
          {apiStatus.connected
            ? `Connected (${apiStatus.device})`
            : 'Backend Disconnected'
          }
        </div>
      </header>

      <main className="main-content">
        <div className="left-panel">
          <section className="upload-section">
            <h2>1. Upload Image</h2>
            <ImageUpload
              onImageSelect={handleImageSelect}
              disabled={loading}
            />
            {imagePreview && (
              <div className="preview-container">
                <img src={imagePreview} alt="Preview" className="image-preview" />
              </div>
            )}
          </section>

          <section className="config-section">
            <h2>2. Configure Attack</h2>
            <AttackConfig
              attackType={attackType}
              onAttackTypeChange={setAttackType}
              params={params}
              onParamsChange={setParams}
              disabled={loading}
            />
          </section>

          <button
            className="attack-button"
            onClick={handleAttack}
            disabled={!selectedFile || loading || !apiStatus.connected}
          >
            {loading ? 'Generating...' : 'Generate Attack'}
          </button>
        </div>

        <div className="right-panel">
          <h2>3. Results</h2>
          <ResultsDisplay
            result={result}
            loading={loading}
            error={error}
          />
        </div>
      </main>

      <footer className="footer">
        <p>
          Adversarial Playground - Educational tool for understanding neural network vulnerabilities
        </p>
      </footer>
    </div>
  );
}

export default App;
