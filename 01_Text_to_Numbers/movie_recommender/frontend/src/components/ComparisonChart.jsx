import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, RadarChart, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Radar,
} from 'recharts'
import { useState } from 'react'

const COLORS = {
  bow:      '#DC2626',
  tfidf:    '#D97706',
  word2vec: '#059669',
  glove:    '#4F46E5',
  fasttext: '#DB2777',
}

const METHOD_NAMES = {
  bow:      'Bag of Words',
  tfidf:    'TF-IDF',
  word2vec: 'Word2Vec',
  glove:    'GloVe',
  fasttext: 'FastText',
}

export default function ComparisonChart({ comparisonData, queryMovie }) {
  const [view, setView] = useState('bar')
  const [radarFocus, setRadarFocus] = useState(null)

  if (!comparisonData) return null

  const methods = Object.keys(comparisonData)

  const titleSet = new Set()
  methods.forEach((m) => {
    comparisonData[m]?.slice(0, 5).forEach((r) => titleSet.add(r.title))
  })
  const topTitles = [...titleSet].slice(0, 8)

  const barData = topTitles.map((title) => {
    const entry = { title: title.length > 22 ? title.slice(0, 22) + '…' : title }
    methods.forEach((m) => {
      const rec = comparisonData[m]?.find((r) => r.title === title)
      entry[m] = rec ? parseFloat((rec.similarity * 100).toFixed(1)) : 0
    })
    return entry
  })

  const radarMovieTitle = radarFocus || topTitles[0]
  const radarData = methods.map((m) => {
    const rec = comparisonData[m]?.find((r) => r.title === radarMovieTitle)
    return {
      method: METHOD_NAMES[m],
      similarity: rec ? parseFloat((rec.similarity * 100).toFixed(1)) : 0,
      color: COLORS[m],
    }
  })

  return (
    <div className="bg-white border border-[#E5E7EB] rounded-xl shadow-sm fade-up">
      {/* Panel header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-[#E5E7EB]">
        <div>
          <h2 className="text-base font-bold text-[#0D1926]">Algorithm Comparison</h2>
          <p className="text-xs text-[#8D99A5] mt-0.5">
            How differently each method scores the same movies for "{queryMovie}"
          </p>
        </div>
        {/* View toggle — Cloudscape segmented control */}
        <div className="flex border border-[#D1D5DB] rounded-lg overflow-hidden bg-[#F9FAFB]">
          {['bar', 'radar'].map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`px-3 py-1.5 text-xs font-semibold transition-colors ${
                view === v
                  ? 'bg-[#0972D3] text-white'
                  : 'text-[#5F6B7A] hover:bg-[#F3F4F6]'
              }`}
            >
              {v === 'bar' ? '📊 Bar' : '🕸️ Radar'}
            </button>
          ))}
        </div>
      </div>

      <div className="p-5">
        {view === 'bar' && (
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={barData} margin={{ top: 5, right: 10, bottom: 60, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
              <XAxis dataKey="title" tick={{ fill: '#6B7280', fontSize: 10 }} angle={-35} textAnchor="end" interval={0} />
              <YAxis
                tick={{ fill: '#9CA3AF', fontSize: 11 }}
                label={{ value: 'Similarity %', angle: -90, position: 'insideLeft', fill: '#9CA3AF', fontSize: 11 }}
                domain={[0, 100]}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #E5E7EB', borderRadius: 8, boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                labelStyle={{ color: '#0D1926', fontWeight: 'bold', marginBottom: 4 }}
                itemStyle={{ color: '#5F6B7A', fontSize: 12 }}
                formatter={(v, name) => [`${v}%`, METHOD_NAMES[name] || name]}
              />
              <Legend
                wrapperStyle={{ paddingTop: 12 }}
                formatter={(value) => (
                  <span style={{ color: COLORS[value] || '#6B7280', fontSize: 11 }}>
                    {METHOD_NAMES[value] || value}
                  </span>
                )}
              />
              {methods.map((m) => (
                <Bar key={m} dataKey={m} fill={COLORS[m]} radius={[3, 3, 0, 0]} maxBarSize={16} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        )}

        {view === 'radar' && (
          <div>
            <div className="flex flex-wrap gap-2 mb-4">
              {topTitles.map((t) => (
                <button
                  key={t}
                  onClick={() => setRadarFocus(t)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors border ${
                    radarMovieTitle === t
                      ? 'bg-[#0972D3] text-white border-[#0972D3]'
                      : 'bg-white text-[#5F6B7A] border-[#D1D5DB] hover:border-[#0972D3] hover:text-[#0972D3]'
                  }`}
                >
                  {t.length > 18 ? t.slice(0, 18) + '…' : t}
                </button>
              ))}
            </div>
            <p className="text-xs text-[#8D99A5] mb-3">
              Similarity profile for <span className="text-[#0972D3] font-medium">"{radarMovieTitle}"</span> across all algorithms
            </p>
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#E5E7EB" />
                <PolarAngleAxis dataKey="method" tick={{ fill: '#6B7280', fontSize: 11 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#9CA3AF', fontSize: 10 }} />
                <Radar name="Similarity %" dataKey="similarity" stroke="#0972D3" fill="#0972D3" fillOpacity={0.15} strokeWidth={2} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#fff', border: '1px solid #E5E7EB', borderRadius: 8 }}
                  formatter={(v) => [`${v}%`, 'Similarity']}
                />
              </RadarChart>
            </ResponsiveContainer>

            <div className="grid grid-cols-3 md:grid-cols-5 gap-2 mt-4">
              {radarData.map((d) => (
                <div key={d.method} className="bg-[#F9FAFB] border border-[#E5E7EB] rounded-lg p-2.5 text-center">
                  <div className="text-[10px] text-[#8D99A5] mb-1 font-medium">{d.method}</div>
                  <div className="text-sm font-bold" style={{ color: d.color }}>{d.similarity}%</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Insight callout */}
        <div className="mt-5 bg-[#E8F2FF] border border-[#B5D3F5] rounded-lg px-4 py-3">
          <p className="text-xs text-[#0551A3] leading-relaxed">
            <span className="font-bold">💡 What to notice: </span>
            TF-IDF catches movies sharing rare keywords. Word2Vec (Google News) and GloVe find semantic relatives trained on billions of words.
            FastText (Wiki News) handles unusual words via character n-grams. BoW just counts — still works well for genre/director matches.
          </p>
        </div>
      </div>
    </div>
  )
}
