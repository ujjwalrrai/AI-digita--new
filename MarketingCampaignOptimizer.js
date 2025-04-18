import React, { useState } from 'react';
import axios from 'axios';
import './MarketingCampaignOptimizer.css';

// Campaign configuration options
const CAMPAIGN_TYPES = ['Email', 'Influencer', 'Display', 'Search', 'Social Media'];
const CUSTOMER_SEGMENTS = ['Health & Wellness', 'Fashionistas', 'Outdoor Adventurers', 'Foodies', 'Tech Enthusiasts'];
const LOCATIONS = ['New York', 'Los Angeles', 'Chicago', 'Miami', 'Houston'];
const LANGUAGES = ['English', 'Spanish', 'French', 'German', 'Mandarin'];

const MarketingCampaignOptimizer = () => {
  // State for form inputs
  const [formData, setFormData] = useState({
    campaign_type: CAMPAIGN_TYPES[0],
    customer_segment: CUSTOMER_SEGMENTS[0],
    location: LOCATIONS[0],
    language: LANGUAGES[0],
    duration_days: 30,
    conversion_weight: 1.0,
    roi_weight: 1.0,
    engagement_weight: 1.0
  });

  // UI state
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  // API endpoint - adjust based on your deployment
  const API_URL = 'http://localhost:8080/optimize'; // FastAPI default port

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name.endsWith('_weight') || name === 'duration_days' 
        ? parseFloat(value) 
        : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(false);

    try {
      const response = await axios.post(API_URL, formData);
      
      if (response.data && response.data.length > 0) {
        setRecommendations(response.data);
        setSuccess(true);
      } else {
        setError('No recommendations returned from the server');
      }
    } catch (err) {
      console.error("API Error:", err);
      setError(err.response?.data?.detail || err.message || 'Failed to fetch recommendations');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="marketing-campaign-container">
      <h1 className="page-title">Marketing Campaign Optimizer</h1>
      
      <form onSubmit={handleSubmit} className="campaign-form">
        <div className="form-grid">
          <div className="form-group">
            <label>Campaign Type</label>
            <select 
              name="campaign_type"
              value={formData.campaign_type}
              onChange={handleChange}
            >
              {CAMPAIGN_TYPES.map(type => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Customer Segment</label>
            <select 
              name="customer_segment"
              value={formData.customer_segment}
              onChange={handleChange}
            >
              {CUSTOMER_SEGMENTS.map(segment => (
                <option key={segment} value={segment}>{segment}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Location</label>
            <select 
              name="location"
              value={formData.location}
              onChange={handleChange}
            >
              {LOCATIONS.map(loc => (
                <option key={loc} value={loc}>{loc}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Language</label>
            <select 
              name="language"
              value={formData.language}
              onChange={handleChange}
            >
              {LANGUAGES.map(lang => (
                <option key={lang} value={lang}>{lang}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="additional-details">
          <div className="form-group">
            <label>Campaign Duration (Days)</label>
            <input 
              type="number" 
              name="duration_days"
              value={formData.duration_days}
              onChange={handleChange}
              min="7" 
              max="90"
              required
            />
          </div>
        </div>

        <div className="optimization-priorities">
          <h2>Optimization Priorities</h2>
          <p className="priority-description">
            Adjust the weights to prioritize different metrics (0.5-2.0)
          </p>
          <div className="priorities-grid">
            <div className="form-group">
              <label>Conversion Rate Weight</label>
              <input 
                type="number" 
                name="conversion_weight"
                step="0.1"
                value={formData.conversion_weight}
                onChange={handleChange}
                min="0.5" 
                max="2.0"
              />
            </div>
            <div className="form-group">
              <label>ROI Weight</label>
              <input 
                type="number" 
                name="roi_weight"
                step="0.1"
                value={formData.roi_weight}
                onChange={handleChange}
                min="0.5" 
                max="2.0"
              />
            </div>
            <div className="form-group">
              <label>Engagement Weight</label>
              <input 
                type="number" 
                name="engagement_weight"
                step="0.1"
                value={formData.engagement_weight}
                onChange={handleChange}
                min="0.5" 
                max="2.0"
              />
            </div>
          </div>
        </div>

        <div className="submit-container">
          <button 
            type="submit" 
            disabled={loading} 
            className={`submit-button ${loading ? 'loading' : ''}`}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Generating Recommendations...
              </>
            ) : (
              'Get Campaign Recommendations'
            )}
          </button>
        </div>
      </form>

      {error && (
        <div className="error-message">
          <p>❌ {error}</p>
        </div>
      )}

      {success && recommendations.length > 0 && (
        <div className="recommendations-section">
          <h2>Top Campaign Recommendations</h2>
          <p className="results-count">
            Showing {recommendations.length} recommendations sorted by overall score
          </p>
          
          <div className="table-responsive">
            <table className="recommendations-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Channel</th>
                  <th>Target Audience</th>
                  <th>Conversion Rate</th>
                  <th>ROI</th>
                  <th>Engagement</th>
                  <th>Overall Score</th>
                </tr>
              </thead>
              <tbody>
                {recommendations.map((rec, index) => (
                  <tr key={index} className={index < 3 ? 'top-recommendation' : ''}>
                    <td>{index + 1}</td>
                    <td>{rec.channel_used}</td>
                    <td>{rec.target_audience}</td>
                    <td>{(rec.predicted_conversion_rate * 100).toFixed(2)}%</td>
                    <td>{rec.predicted_roi.toFixed(2)}x</td>
                    <td>{rec.predicted_engagement.toFixed(2)}</td>
                    <td>{rec.overall_score.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="recommendation-notes">
            <p>
              <strong>Note:</strong> These recommendations are generated based on historical campaign data 
              and machine learning predictions. Actual results may vary.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default MarketingCampaignOptimizer;