import React from 'react';
import { useNavigate } from 'react-router-dom';

const Landing = () => {
  const navigate = useNavigate();

  const handleDemoClick = () => {
    navigate('/demo');
  };

  const handleContactClick = () => {
    navigate('/contact'); // change to your desired route
  };

  return (
    <div className="statio-container">
      {/* Header */}
      <header className="statio-header">
        <div className="header-content">
          <div className="logo">
            <img src="./assets/statio-logo.svg" alt="Statio" />
          </div>
          <nav className="navigation">
            <ul className="nav-links">
              <li><a href="#platform">Platform</a></li>
              <li><a href="#solutions">Solutions</a></li>
              <li><a href="#resources">Resources</a></li>
              <li><a href="#pricing">Pricing</a></li>
            </ul>
          </nav>
          <div className="auth-buttons">
            <button className="login-button">Log in</button>
            <button className="signup-button">Get Started</button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">Start excelling with digital analytics</h1>
          <p className="hero-subtitle">Get reliable data and the insights you need to take action and promote growth.</p>
          <div className="hero-buttons">
            <button className="contact-button"onClick={handleContactClick}>Contact Sales</button>
            <button className="demo-button" onClick={handleDemoClick}>Get the Demo</button>
          </div>
        </div>
      </section>

      {/* Expansion Section */}
      <section className="expansion-section">
        <div className="expansion-content">
          <div className="expansion-text">
            <h2 className="expansion-title">Begin accelerating expansion now</h2>
          </div>
          <div className="expansion-details">
            <p className="expansion-description">
              Explore essential tools to streamline workflow, enhance team collaboration, and ensure project success.
            </p>
          </div>
        </div>

        {/* Card Section */}
        <div className="cards-container">
          <div className="card yellow-card">
            <h3 className="card-title">Try for free</h3>
            <p className="card-description">
              Don't just take our word for it—experience the power of self-service analytics firsthand with a free trial, using your own data to unlock its potential.
            </p>
            <button className="card-button">Sign Up →</button>
          </div>

          <div className="card green-card">
            <h3 className="card-title">Explore demo</h3>
            <p className="card-description">
              Discover Statio's popular features & see how they can significantly impact.
            </p>
            <button className="card-button">Sign Up →</button>
          </div>

          <div className="card orange-card">
            <h3 className="card-title">On-demand</h3>
            <p className="card-description">
              Explore the journey Statio users follow to harness its digital analytics platform.
            </p>
            <button className="card-button">Sign Up →</button>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Landing;