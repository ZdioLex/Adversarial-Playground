import { useState, useEffect } from 'react';
import { ImageUpload, AttackSelector, ResultsDisplay } from './components';
import { performAttack, checkHealth, switchDevice } from './services/api';
import {
  AttackType,
  AttackCategory,
  AttackResponse,
  AttackParams,
  ModelType,
  HealthStatus
} from './types';
import './App.css';

function App() {
  // State for image
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  // State for attack configuration
  const [category, setCategory] = useState<AttackCategory>('universal');
  const [attackType, setAttackType] = useState<AttackType>('fgsm');
  const [modelType, setModelType] = useState<ModelType>('cnn');
  const [modelName, setModelName] = useState<string>('resnet50');
  const [params, setParams] = useState<AttackParams>({ epsilon: 0.03 });

  // State for results
  const [result, setResult] = useState<AttackResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // State for API health
  const [apiStatus, setApiStatus] = useState<HealthStatus | null>(null);
  const [switching, setSwitching] = useState(false);

  // Check API health on mount
  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        const health = await checkHealth();
        setApiStatus(health);
      } catch {
        setApiStatus(null);
      }
    };

    checkApiHealth();
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
      const response = await performAttack(
        category,
        attackType,
        selectedFile,
        modelType,
        modelName,
        params
      );
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

  const isConnected = apiStatus?.status === 'healthy' && apiStatus?.cnn_loaded;
  const vitAvailable = apiStatus?.vit_available ?? false;
  const isUsingGpu = apiStatus?.cnn_device === 'GPU';
  const gpuAvailable = apiStatus?.gpu_available ?? false;

  const handleDeviceSwitch = async () => {
    if (switching || !gpuAvailable) return;

    const targetDevice = isUsingGpu ? 'cpu' : 'gpu';
    setSwitching(true);

    try {
      await switchDevice(targetDevice);
      // Refresh health status
      const health = await checkHealth();
      setApiStatus(health);
    } catch (err) {
      console.error('Failed to switch device:', err);
    } finally {
      setSwitching(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>Adversarial Playground</h1>
          <p className="subtitle">
            Explore adversarial attacks on CNN and ViT image classifiers
          </p>
        </div>
        <div className="status-bar">
          {apiStatus?.gpu_name && (
            <div className="api-status gpu-status connected">
              <span className="status-dot gpu"></span>
              {apiStatus.gpu_name} | CUDA {apiStatus.cuda_version} | {apiStatus.gpu_memory_used} / {apiStatus.gpu_memory_total}
            </div>
          )}
          <div className={`api-status ${isConnected ? 'connected' : 'disconnected'}`}>
            <span className="status-dot"></span>
            {isConnected
              ? `CNN: ${apiStatus?.cnn_device}`
              : 'Backend Disconnected'
            }
          </div>
          {vitAvailable && (
            <div className="api-status connected">
              <span className="status-dot"></span>
              ViT: {apiStatus?.vit_device}
            </div>
          )}
          {gpuAvailable && isConnected && (
            <button
              className={`device-switch-btn ${isUsingGpu ? 'gpu-active' : 'cpu-active'}`}
              onClick={handleDeviceSwitch}
              disabled={switching || loading}
              title={isUsingGpu ? 'Switch to CPU' : 'Switch to GPU'}
            >
              {switching ? '...' : isUsingGpu ? 'GPU' : 'CPU'}
            </button>
          )}
        </div>
      </header>

      <main className="main-content">
        {/* Step 1: Upload */}
        <section className="step-panel">
          <div className="step-header">
            <span className="step-number">1</span>
            <h2>Upload Image</h2>
          </div>
          <div className="step-content">
            <ImageUpload
              onImageSelect={handleImageSelect}
              disabled={loading}
            />
            {imagePreview && (
              <div className="preview-container">
                <img src={imagePreview} alt="Preview" className="image-preview" />
              </div>
            )}
          </div>
        </section>

        {/* Step 2: Configure */}
        <section className="step-panel">
          <div className="step-header">
            <span className="step-number">2</span>
            <h2>Configure Attack</h2>
          </div>
          <div className="step-content">
            <AttackSelector
              category={category}
              attackType={attackType}
              modelType={modelType}
              modelName={modelName}
              params={params}
              onCategoryChange={setCategory}
              onAttackTypeChange={setAttackType}
              onModelTypeChange={setModelType}
              onModelNameChange={setModelName}
              onParamsChange={setParams}
              vitAvailable={vitAvailable}
              disabled={loading}
            />
            <button
              className="attack-button"
              onClick={handleAttack}
              disabled={!selectedFile || loading || !isConnected}
            >
              {loading ? 'Generating...' : 'Generate Attack'}
            </button>
          </div>
        </section>

        {/* Step 3: Results */}
        <section className="step-panel results-panel">
          <div className="step-header">
            <span className="step-number">3</span>
            <h2>Results</h2>
          </div>
          <div className="step-content">
            <ResultsDisplay
              result={result}
              loading={loading}
              error={error}
            />
          </div>
        </section>
      </main>

      <footer className="footer">
        <p>
          Adversarial Playground - Educational tool for understanding neural network vulnerabilities
        </p>
        <p className="footer-info">
          Supports: FGSM, PGD, MIM, C&W, Adversarial Patch | CNN: ResNet | ViT: Vision Transformer
        </p>
      </footer>
    </div>
  );
}

export default App;
