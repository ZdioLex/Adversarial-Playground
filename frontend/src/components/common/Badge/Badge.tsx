import React from 'react';
import './Badge.css';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'model' | 'attack' | 'success' | 'error';
}

const Badge: React.FC<BadgeProps> = ({ children, variant = 'model' }) => {
  return (
    <span className={`badge badge-${variant}`}>
      {children}
    </span>
  );
};

export default Badge;
