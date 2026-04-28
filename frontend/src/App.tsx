import React from 'react';
import {BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import Dashboard from './pages/Dashboard';
import Pipeline from './pages/Pipeline';
import './App.css';

function App() {
  return (
    <BrowserRouter>
    <nav style={{
      background: '#1F4E79', padding: '0.8rem 2rem', display: 'flex', gap: '2rem', alignItems: 'center'
    }}>
      <span style={{color: '#fff', fontWeight: 600, fontSize: 16 }}>
        Sentiment Dashboard
      </span>
      <Link to="/"     style={{ color: '#B5D4F4', textDecoration: 'none' }}>Dashboard</Link>
      <Link to="/pipeline"     style={{ color: '#B5D4F4', textDecoration: 'none' }}>Pipeline</Link>
    </nav>
    <Routes>
      <Route path='/'         element={<Dashboard />} />
      <Route path='/pipeline' element={<Pipeline />} />
    </Routes>
    </BrowserRouter>
  );
}

export default App;
