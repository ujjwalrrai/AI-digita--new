import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './PredictionForm.css';

const PredictionForm = () => {
  const [formData, setFormData] = useState({
    campaign_type: 'Email',
    target_audience: 'Men 18-24',
    duration_days: 20,
    channel_used: 'Google Ads',
    acquisition_cost: 10000,
    roi: 5,
    location: 'New York',
    language: 'English',
    engagement_score: 5,
    customer_segment: 'Tech Enthusiasts',
    conversion_rate: 0.1
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);

  // Check model status on component mount
  useEffect(() => {
    const checkModelStatus = async () => {
      try {
        const response = await axios.get('http://localhost:8000/model-status');
        setModelStatus(response.data);
      } catch (err) {
        console.error("Error checking model status:", err);
      }
    };
    checkModelStatus();
  }, []);const [validCategories, setValidCategories] = useState(null);

  useEffect(() => {
    const fetchCategories = async () => {
      try {
        const response = await axios.get('http://localhost:8000/valid-categories');
        setValidCategories(response.data);
      } catch (error) {
        console.error("Error fetching categories:", error);
      }
    };
    fetchCategories();
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: ['duration_days', 'acquisition_cost', 'roi', 'engagement_score', 'conversion_rate'].includes(name)
        ? parseFloat(value) || 0
        : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post(
        'http://localhost:8000/predict', 
        formData,
        {
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );
      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to get prediction');
    } finally {
      setLoading(false);
    }
  };

  // Dropdown options - should match your encoders
  const campaignTypes = ['Email', 'Social Media', 'Search', 'Display'];
  const targetAudiences = ['Men 18-24', 'Men 25-34', 'Women 25-34', 'Women 35-44', 'All Ages'];
  const channels = ['Google Ads', 'Facebook', 'Instagram', 'YouTube', 'Email', 'Organic'];
  const locations = ['New York', 'California', 'Texas', 'Florida', 'National'];
  const languages = ['English', 'Spanish', 'French', 'German'];
  const segments = ['Tech Enthusiasts', 'Frequent Shoppers', 'Budget Conscious', 'Luxury Buyers'];

  return (
    <div className="prediction-form">
      <h2>Campaign Performance Predictor</h2>
      
      {modelStatus && !modelStatus.status === 'ready' && (
        <div className="model-status-warning">
          ⚠️ Models are still loading. Predictions may not work.
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          <div className="form-group">
            <label>Campaign Type:</label>
            <select 
              name="campaign_type" 
              value={formData.campaign_type}
              onChange={handleChange}
              required
            >
              {campaignTypes.map(type => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Target Audience:</label>
            <select 
              name="target_audience" 
              value={formData.target_audience}
              onChange={handleChange}
              required
            >
              {targetAudiences.map(audience => (
                <option key={audience} value={audience}>{audience}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Duration (days):</label>
            <input
              type="number"
              name="duration_days"
              min="1"
              max="60"
              value={formData.duration_days}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label>Channel Used:</label>
            <select 
              name="channel_used" 
              value={formData.channel_used}
              onChange={handleChange}
              required
            >
              {channels.map(channel => (
                <option key={channel} value={channel}>{channel}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Acquisition Cost ($):</label>
            <input
              type="number"
              name="acquisition_cost"
              min="1000"
              max="50000"
              step="1000"
              value={formData.acquisition_cost}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label>ROI:</label>
            <input
              type="number"
              name="roi"
              min="0"
              max="10"
              step="0.1"
              value={formData.roi}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label>Location:</label>
           <select 
  name="location" 
  value={formData.location}
  onChange={handleChange}
  required
>
  {validCategories?.location.map(option => (
    <option key={option} value={option}>{option}</option>
  ))}
</select>
          </div>

          <div className="form-group">
            <label>Language:</label>
            <select 
              name="language" 
              value={formData.language}
              onChange={handleChange}
              required
            >
              {languages.map(lang => (
                <option key={lang} value={lang}>{lang}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Engagement Score:</label>
            <input
              type="number"
              name="engagement_score"
              min="0"
              max="10"
              step="0.1"
              value={formData.engagement_score}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label>Customer Segment:</label>
            <select 
              name="customer_segment" 
              value={formData.customer_segment}
              onChange={handleChange}
              required
            >
              {segments.map(segment => (
                <option key={segment} value={segment}>{segment}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Conversion Rate:</label>
            <input
              type="number"
              name="conversion_rate"
              min="0"
              max="1"
              step="0.01"
              value={formData.conversion_rate}
              onChange={handleChange}
              required
            />
          </div>
        </div>
        
        <button 
          type="submit" 
          disabled={loading || (modelStatus && modelStatus.status !== 'ready')}
          className="predict-button"
        >
          {loading ? (
            <>
              <span className="spinner"></span>
              Predicting...
            </>
          ) : 'Predict Performance'}
        </button>
      </form>
      
      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}
      
      {prediction && (
        <div className="prediction-results">
          <h3>Prediction Results</h3>
          <div className="result-item">
            <span className="result-label">Predicted Clicks:</span>
            <span className="result-value">{prediction.predicted_clicks.toLocaleString()}</span>
          </div>
          <div className="result-item">
            <span className="result-label">Predicted Impressions:</span>
            <span className="result-value">{prediction.predicted_impressions.toLocaleString()}</span>
          </div>
          <div className="result-item">
            <span className="result-label">Click-Through Rate:</span>
            <span className="result-value">{prediction.click_through_rate}%</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionForm;