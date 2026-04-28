import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  PieChart, Pie, Cell, ResponsiveContainer
} from 'recharts';

const API = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const COLORS: Record<string, string> = {
  positive: '#1D9E75',
  neutral: '#BA7517',
  negative: '#E24B4A'
};

export default function Dashboard() {

  const [text, setText] = useState('');
  const [aspect, setAspect] = useState('general');
  const [result, setResult] = useState<any>(null);
  const [reviews, setReviews] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchReviews();
  }, []);

  const fetchReviews = async () => {
    try {
      const res = await axios.get(`${API}/reviews?limit=200`);
      setReviews(res.data);
    } catch (e) {
      console.error('Failed to fetch reviews');
    }
  };

  const handleSubmit = async () => {
    if (!text.trim()) return;

    setLoading(true);
    try {
      const res = await axios.post(`${API}/predict`, {
        review_text: text,
        aspect
      });
      setResult(res.data);
      await fetchReviews();
    } catch (e) {
      alert('Prediction failed');
    }
    setLoading(false);
  };

  // Pie chart data
  const sentimentCounts = ['positive', 'neutral', 'negative'].map(s => ({
    name: s,
    value: reviews.filter(r => r.predicted === s).length
  }));

  // Bar chart data
  const aspectData = ['price', 'quality', 'delivery', 'service', 'general'].map(a => ({
    aspect: a,
    positive: reviews.filter(r => r.aspect === a && r.predicted === 'positive').length,
    neutral:  reviews.filter(r => r.aspect === a && r.predicted === 'neutral').length,
    negative: reviews.filter(r => r.aspect === a && r.predicted === 'negative').length,
  }));

  return (
    <div style={{maxWidth: 1100, margin: '0 auto', padding: '2rem', fontFamily: 'sans-serif'}}>
        <h1 style={{ color:'#1F4E79' }}> Electronics Product Review Sentiment Dashboard </h1>

      {/* Input Section */}
      <div style={{background: '#f8f9fa', padding: '1.5rem', borderRadius: 8, marginBottom: '2rem'}}>
        <h2 style={{marginTop: 0}}>Classify a review</h2>

        <textarea rows={4} style={{width: '100%', marginBottom: '1rem', padding: '0.5rem', fontSize: 14, borderRadius: 4, border: '1px solid #ccc'}} placeholder="Paste a product review here..." value={text} onChange={e => setText(e.target.value)}/>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <select value={aspect} onChange={e => setAspect(e.target.value)} style={{ padding: '0.4rem', borderRadius: 4 }}>
            {['general', 'price', 'quality', 'delivery', 'service'].map(a => (
              <option key={a} value={a}>{a}</option>
            ))}
          </select>

          <button onClick={handleSubmit} disabled={loading}
            style={{padding: '0.5rem 1.5rem', background: '#2E75B6', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer'}}>
            {loading ? 'Analysing...' : 'Classify'}
          </button>
        </div>

        {result && (
          <div style={{marginTop: '1rem', padding: '1rem', borderRadius: 6, background: result.low_confidence ? '#FFF3CD' : '#D4EDDA', border: `1px solid ${result.low_confidence ? '#FFEAA7' : '#C3E6CB'}`}}>
            <strong>Sentiment:</strong> {result.predicted} &nbsp;|&nbsp;
            <strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%
            {result.low_confidence && (
              <span style={{ color: '#856404' }}> — Low confidence, verify manually </span>
            )}
          </div>
        )}
      </div>

      {/* Charts */}
      <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2rem'}}>

        {/* Pie Chart */}
        <div>
          <h3>Overall sentiment distribution</h3>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={sentimentCounts}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={80}
                label={({ name, value }) => `${name}: ${value}`}
              >
                {sentimentCounts.map(entry => (
                  <Cell key={entry.name} fill={COLORS[entry.name]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Bar Chart */}
        <div>
          <h3>Sentiment by aspect</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={aspectData}>
              <XAxis dataKey="aspect" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="positive" fill="#1D9E75" stackId="a" />
              <Bar dataKey="neutral"  fill="#BA7517" stackId="a" />
              <Bar dataKey="negative" fill="#E24B4A" stackId="a" />
            </BarChart>
          </ResponsiveContainer>
        </div>

      </div>

      {/* Table */}
      <h3>Recent predictions</h3>
      <table style={{width: '100%', borderCollapse: 'collapse', fontSize: 13}}>
        <thead>
          <tr style={{ background: '#1F4E79', color: '#fff' }}>
            {['Review', 'Predicted', 'Confidence', 'Aspect', 'Time'].map(h => (
              <th key={h} style={{ padding: '8px', textAlign: 'left' }}>{h}</th>
            ))}
          </tr></thead>

        <tbody>
          {reviews.slice(0, 20).map((r, i) => (
            <tr key={r.id} style={{background: i % 2 === 0 ? '#f8f9fa' : '#fff'}}>
              <td style={{padding: '6px'}}>{r.text?.slice(0, 60)}...</td>
              <td style={{padding: '6px', color: COLORS[r.predicted] || '#333', fontWeight: 500}}>
                {r.predicted}
              </td>
              <td style={{padding: '6px'}}>
                {(r.confidence*100).toFixed(1)}%
              </td>
              <td style={{padding: '6px'}}>{r.aspect}</td>
              <td style={{padding: '6px'}}>
                {new Date(r.timestamp).toLocaleTimeString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

    </div>
  );
}