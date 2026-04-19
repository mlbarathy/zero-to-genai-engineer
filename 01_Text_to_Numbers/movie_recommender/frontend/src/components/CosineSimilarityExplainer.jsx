import { useState } from 'react'

export default function CosineSimilarityExplainer() {
  const [open, setOpen] = useState(false)
  const [angle, setAngle] = useState(30)

  const rad = (angle * Math.PI) / 180
  const cosine = Math.cos(rad).toFixed(3)

  const A  = { x: 120, y: -50 }
  const Bx = 120 * Math.cos(rad) + (-50) * Math.sin(rad)
  const By = -120 * Math.sin(rad) + (-50) * Math.cos(rad)
  const cx = 150, cy = 160

  return (
    <div className="bg-white border border-[#E5E7EB] rounded-xl overflow-hidden shadow-sm">
      {/* Toggle header — Cloudscape expandable section */}
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-4 hover:bg-[#F9FAFB] transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#E8F2FF] flex items-center justify-center">
            <span className="text-lg">📐</span>
          </div>
          <div className="text-left">
            <p className="font-semibold text-[#0D1926] text-sm">Why Cosine Similarity? (Not Euclidean Distance)</p>
            <p className="text-xs text-[#8D99A5]">Interactive explainer — click to expand</p>
          </div>
        </div>
        <svg
          className={`w-5 h-5 text-[#5F6B7A] transition-transform ${open ? 'rotate-180' : ''}`}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="border-t border-[#E5E7EB] px-5 py-5 bg-[#FAFAFA] fade-up">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">

            {/* Left: explanation */}
            <div>
              <h3 className="font-bold text-[#0972D3] text-sm mb-3">The Problem with Euclidean Distance</h3>
              <div className="space-y-3 text-sm text-[#374151] leading-relaxed">
                <p>Imagine two documents:</p>
                <div className="bg-white border border-[#E5E7EB] rounded-lg p-3 font-mono text-xs space-y-1 shadow-sm">
                  <div><span className="text-[#0972D3] font-bold">Doc A:</span> "crime crime crime crime"</div>
                  <div><span className="text-[#DB2777] font-bold">Doc B:</span> "crime"</div>
                </div>
                <p>Both are about crime. They should be <strong className="text-[#0D1926]">very similar</strong>.</p>
                <div className="bg-white border border-[#E5E7EB] rounded-lg p-3 space-y-2 text-xs shadow-sm">
                  <div className="flex justify-between items-center">
                    <span className="text-[#DC2626] font-medium">❌ Euclidean distance</span>
                    <span className="text-[#DC2626] font-bold">LARGE (different lengths!)</span>
                  </div>
                  <div className="h-px bg-[#F3F4F6]" />
                  <div className="flex justify-between items-center">
                    <span className="text-[#059669] font-medium">✅ Cosine similarity</span>
                    <span className="text-[#059669] font-bold">≈ 1.0 (same direction!)</span>
                  </div>
                </div>
                <p className="text-[#6B7280]">
                  Euclidean distance measures the <em>length of the gap</em> between vector tips.
                  Cosine similarity measures only the <em>angle</em> — for text, topic matters, not length.
                </p>
                <div className="border-l-4 border-[#0972D3] pl-3 bg-[#E8F2FF] py-2 rounded-r-lg">
                  <strong className="text-[#0D1926] text-xs">Formula:</strong><br />
                  <code className="text-[#0972D3] text-xs">cos(θ) = (A · B) / (‖A‖ × ‖B‖)</code><br />
                  <span className="text-xs text-[#6B7280]">Dividing by magnitudes removes the length effect.</span>
                </div>
              </div>
            </div>

            {/* Right: interactive SVG */}
            <div>
              <h3 className="font-bold text-[#0972D3] text-sm mb-3">Interactive: Drag the Angle</h3>
              <div className="flex flex-col items-center gap-4">
                <svg width="300" height="220" className="bg-white border border-[#E5E7EB] rounded-xl shadow-sm">
                  <line x1={cx} y1="0" x2={cx} y2="220" stroke="#E5E7EB" strokeWidth="1" />
                  <line x1="0" y1={cy} x2="300" y2={cy} stroke="#E5E7EB" strokeWidth="1" />

                  <path
                    d={`M ${cx + 35} ${cy} A 35 35 0 0 0 ${cx + 35 * Math.cos(-Math.atan2(A.y, A.x))} ${cy + 35 * Math.sin(-Math.atan2(A.y, A.x))}`}
                    fill="none" stroke="#D97706" strokeWidth="1.5" strokeDasharray="4 2"
                  />

                  <defs>
                    <marker id="arrowA" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                      <path d="M0,0 L0,6 L8,3 z" fill="#0972D3" />
                    </marker>
                    <marker id="arrowB" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                      <path d="M0,0 L0,6 L8,3 z" fill="#DB2777" />
                    </marker>
                  </defs>
                  <line x1={cx} y1={cy} x2={cx + A.x} y2={cy + A.y} stroke="#0972D3" strokeWidth="2.5" markerEnd="url(#arrowA)" />
                  <text x={cx + A.x + 6} y={cy + A.y} fill="#0972D3" fontSize="12" fontWeight="bold">A</text>

                  <line x1={cx} y1={cy} x2={cx + Bx} y2={cy + By} stroke="#DB2777" strokeWidth="2.5" markerEnd="url(#arrowB)" />
                  <text x={cx + Bx + 6} y={cy + By} fill="#DB2777" fontSize="12" fontWeight="bold">B</text>

                  <line x1={cx + A.x} y1={cy + A.y} x2={cx + Bx} y2={cy + By} stroke="#DC2626" strokeWidth="1.5" strokeDasharray="5 3" />
                  <text x={(cx + A.x + cx + Bx) / 2 + 4} y={(cy + A.y + cy + By) / 2} fill="#DC2626" fontSize="9">Euclidean</text>

                  <text x={cx + 40} y={cy - 5} fill="#D97706" fontSize="11">θ = {angle}°</text>
                  <circle cx={cx} cy={cy} r="4" fill="#0972D3" />
                </svg>

                <div className="w-full space-y-1.5">
                  <div className="flex justify-between text-[11px] text-[#8D99A5]">
                    <span>0° (identical)</span>
                    <span>90° (unrelated)</span>
                    <span>180° (opposite)</span>
                  </div>
                  <input
                    type="range" min="0" max="175" step="5"
                    value={angle}
                    onChange={(e) => setAngle(Number(e.target.value))}
                    className="w-full accent-[#0972D3]"
                  />
                </div>

                <div className="grid grid-cols-2 gap-3 w-full text-sm">
                  <div className="bg-[#ECFDF5] border border-[#A7F3D0] rounded-lg p-3 text-center">
                    <div className="text-xs text-[#6B7280] mb-1">Cosine Similarity</div>
                    <div className="text-xl font-bold text-[#059669]">{cosine}</div>
                    <div className="text-xs text-[#6B7280] mt-1">
                      {cosine > 0.8 ? '🟢 Very similar' : cosine > 0.5 ? '🟡 Somewhat similar' : cosine > 0 ? '🟠 Loosely related' : '🔴 Opposite'}
                    </div>
                  </div>
                  <div className="bg-[#FEF2F2] border border-[#FECACA] rounded-lg p-3 text-center">
                    <div className="text-xs text-[#6B7280] mb-1">Range</div>
                    <div className="text-xl font-bold text-[#DC2626]">[-1, 1]</div>
                    <div className="text-xs text-[#6B7280] mt-1">1 = identical · 0 = unrelated</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
