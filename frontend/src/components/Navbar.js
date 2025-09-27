import { useState } from 'react';

const Navbar = ({ user, onLogout }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
      <div className="container">
        <span className="navbar-brand">🌩️ Cloud File Manager</span>
        
        <button 
          className="navbar-toggler" 
          onClick={() => setIsOpen(!isOpen)}
        >
          <span className="navbar-toggler-icon"></span>
        </button>

        <div className={`collapse navbar-collapse ${isOpen ? 'show' : ''}`}>
          <div className="navbar-nav ms-auto">
            {user && (
              <>
                <span className="navbar-text me-3">
                  Welcome, {user.name} ({user.role})
                </span>
                <button className="btn btn-outline-light btn-sm" onClick={onLogout}>
                  Logout
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;