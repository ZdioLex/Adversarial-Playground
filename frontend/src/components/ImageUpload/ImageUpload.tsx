import React, { useRef, useState } from 'react';
import './ImageUpload.css';

interface ImageUploadProps {
  onImageSelect: (file: File, preview: string) => void;
  disabled?: boolean;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onImageSelect, disabled }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateFile = (file: File): boolean => {
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
      setError('Invalid file type. Please upload a JPG or PNG image.');
      return false;
    }

    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      setError('File too large. Maximum size is 10MB.');
      return false;
    }

    setError(null);
    return true;
  };

  const handleFile = (file: File) => {
    if (validateFile(file)) {
      const reader = new FileReader();
      reader.onloadend = () => {
        onImageSelect(file, reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="image-upload-container">
      <div
        className={`upload-zone ${dragActive ? 'drag-active' : ''} ${disabled ? 'disabled' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/jpg"
          onChange={handleChange}
          disabled={disabled}
          style={{ display: 'none' }}
        />
        <div className="upload-content">
          <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <p className="upload-text">
            <span className="upload-highlight">Click to upload</span> or drag and drop
          </p>
          <p className="upload-hint">PNG or JPG (max 10MB)</p>
        </div>
      </div>
      {error && <p className="upload-error">{error}</p>}
    </div>
  );
};

export default ImageUpload;
