import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Login.css';
// import statioLogo from './assets/statio-logo.svg';

const Login = ({ setIsAuthenticated }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      // Basic validation
      if (!email || !password) {
        throw new Error('Please enter both email and password');
      }

      // Simulate API call - replace with actual authentication
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // On successful login:
      setIsAuthenticated(true);
      if (rememberMe) {
        localStorage.setItem('isAuthenticated', 'true');
      }
      navigate('/landing');
    } catch (err) {
      setError(err.message || 'Login failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-header">
        {/* <img src={statioLogo} alt="Statio" className="login-logo" /> */}
        <h1>Sstatio</h1>
      </div>
      
      <div className="login-card">
        <h2 className="login-title">Welcome back</h2>
        <p className="login-subtitle">Start excelling with digital analytics</p>
        
        {error && <div className="error-message">{error}</div>}
        
        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter your email"
              required
              disabled={isLoading}
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              required
              disabled={isLoading}
            />
          </div>
          
          <div className="login-options">
            <div className="remember-me">
              <input
                type="checkbox"
                id="rememberMe"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
                disabled={isLoading}
              />
              <label htmlFor="rememberMe">Remember me</label>
            </div>
            <a href="/forgot-password" className="forgot-password">Forgot password?</a>
          </div>
          
          <button 
            type="submit" 
            className="login-button"
            disabled={isLoading}
          >
            {isLoading ? 'Logging in...' : 'Log in'}
          </button>
        </form>
        
        <div className="login-footer">
          <p>Don't have an account? <a href="/signup">Get started</a></p>
        </div>
      </div>
      
      <div className="login-stats">
        <div className="stat-item">
          <h3>85.78k</h3>
          <p>Current live users</p>
        </div>
        <div className="stat-item">
          <h3>34.6k</h3>
          <p>Positive users</p>
        </div>
        <div className="stat-item">
          <h3>4m 21s</h3>
          <p>Avg session duration</p>
        </div>
      </div>
    </div>
  );
};

export default Login;