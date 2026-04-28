// import React, { useEffect, useState } from 'react';
// import axios from 'axios';

// const API = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// export default function Pipeline() {
//     const [status, setStatus] = useState<any>(null);

//     useEffect(() => {
//         axios.get(`${API}/pipeline-status`)
//         .then(r => setStatus(r.data))
//         .catch(() => setStatus({ status: 'Could not reach backend' }));
//     }, []);

//     return (
//         <div style={{ maxWidth: 1100, margin: '0 auto', padding: '2rem', fontFamily: 'sans-serif'}}>
//             <h2 style={{ color: '#1F4E79' }}>ML pipeline visualization</h2>

//             <div style={{ background: '#f8f9fa', padding: '1.5rem', borderRadius: 8, marginBottom: '2rem' }}>
//                 <h3 style={{ marginTop: 0 }}>DVC pipeline status</h3>
//                 <pre style={{ background: '#1e1e1e', color: '#d4d4d4', padding: '1rem', borderRadius: 6, overflowX: 'auto', fontSize: 13 }}>
//                     {status ? status.status : 'Loading...'}
//                 </pre>
//             </div>
//         </div>
//     );
// }

import React from 'react';
 
const COLORS = {
  gray:   { fill: '#D3D1C7', stroke: '#5F5E5A', text: '#2C2C2A' },
  teal:   { fill: '#9FE1CB', stroke: '#0F6E56', text: '#04342C' },
  purple: { fill: '#CECBF6', stroke: '#534AB7', text: '#26215C' },
  coral:  { fill: '#F5C4B3', stroke: '#993C1D', text: '#4A1B0C' },
};
 
interface NodeBoxProps {
  x: number; y: number; width: number; height: number;
  label: string; color: keyof typeof COLORS;
  dashed?: boolean; small?: boolean;
}
 
function NodeBox({ x, y, width, height, label, color, dashed, small }: NodeBoxProps) {
  const c = COLORS[color];
  return (
    <g>
      <rect
        x={x} y={y} width={width} height={height} rx={8}
        fill={c.fill} stroke={c.stroke} strokeWidth={0.5}
        strokeDasharray={dashed ? '5 3' : undefined}
      />
      <text
        x={x + width / 2} y={y + height / 2}
        textAnchor="middle" dominantBaseline="central"
        fontSize={small ? 11 : 13}
        fontWeight={small ? 400 : 500}
        fill={c.text}
        fontFamily="sans-serif"
      >
        {label}
      </text>
    </g>
  );
}
 
const ARROW_MARKER = (
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
      markerWidth={6} markerHeight={6} orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#888780"
        strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
    </marker>
  </defs>
);
 
/** Shared SVG presentation attributes for edges — typed as SVGProps to avoid
 *  the CSSProperties / SVGLineElementAttributes incompatibility. */
const edge = {
  stroke: '#888780' as const,
  strokeWidth: 1.5,
  fill: 'none' as const,
  markerEnd: 'url(#arrow)',
} satisfies React.SVGProps<SVGElement>;
 
export default function Pipeline() {
  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '2rem', fontFamily: 'sans-serif' }}>
      <h2 style={{ color: '#1F4E79' }}>ML pipeline visualization</h2>
 
      <div style={{
        background: '#f8f9fa', padding: '1.5rem',
        borderRadius: 8, marginBottom: '2rem',
      }}>
        <h3 style={{ marginTop: 0, marginBottom: '1rem' }}>DVC pipeline DAG</h3>
 
        <svg
          width="100%"
          viewBox="0 0 680 510"
          style={{ display: 'block', maxWidth: 640 }}
          aria-label="DVC ML pipeline directed acyclic graph"
        >
          {ARROW_MARKER}
 
          {/* ── Raw data artifact ── */}
          <NodeBox
            x={150} y={30} width={380} height={52}
            label="data/raw/Electronics_5.json.zip.dvc"
            color="gray" dashed small
          />
 
          {/* raw data → ingest */}
          <line x1={340} y1={82} x2={340} y2={140} {...edge} />
 
          {/* ── ingest ── */}
          <NodeBox x={255} y={140} width={170} height={48} label="ingest" color="teal" />
 
          {/* ingest → featurise */}
          <line x1={340} y1={188} x2={340} y2={246} {...edge} />
 
          {/* ── featurise ── */}
          <NodeBox x={245} y={246} width={190} height={48} label="featurise" color="teal" />
 
          {/* featurise → train_baseline (left) */}
          <path d="M300 294 L300 355 L175 355 L175 385" {...edge} />
 
          {/* featurise → train_distilbert (right) */}
          <path d="M380 294 L380 355 L505 355 L505 385" {...edge} />
 
          {/* ── train_baseline ── */}
          <NodeBox x={55} y={385} width={240} height={52} label="train_baseline" color="purple" />
 
          {/* ── train_distilbert ── */}
          <NodeBox x={385} y={385} width={240} height={52} label="train_distilbert" color="coral" />
 
          {/* ── Legend ── */}
          <rect x={90} y={462} width={12} height={12} rx={2}
            fill={COLORS.gray.fill} stroke={COLORS.gray.stroke}
            strokeWidth={0.5} strokeDasharray="4 2" />
          <text x={108} y={468} dominantBaseline="central"
            fontSize={11} fill="#5F5E5A" fontFamily="sans-serif">Data artifact</text>
 
          <rect x={232} y={462} width={12} height={12} rx={2}
            fill={COLORS.teal.fill} stroke={COLORS.teal.stroke} strokeWidth={0.5} />
          <text x={250} y={468} dominantBaseline="central"
            fontSize={11} fill="#5F5E5A" fontFamily="sans-serif">Processing stage</text>
 
          <rect x={390} y={462} width={12} height={12} rx={2}
            fill={COLORS.purple.fill} stroke={COLORS.purple.stroke} strokeWidth={0.5} />
          <text x={408} y={468} dominantBaseline="central"
            fontSize={11} fill="#5F5E5A" fontFamily="sans-serif">Baseline model</text>
 
          <rect x={540} y={462} width={12} height={12} rx={2}
            fill={COLORS.coral.fill} stroke={COLORS.coral.stroke} strokeWidth={0.5} />
          <text x={558} y={468} dominantBaseline="central"
            fontSize={11} fill="#5F5E5A" fontFamily="sans-serif">Distilbert model</text>
        </svg>
      </div>
    </div>
  );
}