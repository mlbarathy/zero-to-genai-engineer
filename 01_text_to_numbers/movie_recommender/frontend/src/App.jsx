import { useState, useEffect } from 'react'
import axios from 'axios'

import Header from './components/Header'
import CosineSimilarityExplainer from './components/CosineSimilarityExplainer'
import AlgorithmSelector from './components/AlgorithmSelector'
import MovieSearch from './components/MovieSearch'
import MovieGrid from './components/MovieGrid'
import ComparisonChart from './components/ComparisonChart'

const API = '/api'

export default function App() {
  const [movies, setMovies]               = useState([])
  const [selectedMovie, setSelectedMovie] = useState('')
  const [method, setMethod]               = useState('tfidf')
  const [recommendations, setRecs]        = useState(null)
  const [comparisonData, setComparison]   = useState(null)
  const [loading, setLoading]             = useState(false)
  const [comparing, setComparing]         = useState(false)
  const [error, setError]                 = useState('')
  const [tab, setTab]                     = useState('recommend')

  useEffect(() => {
    axios.get(`${API}/movies`)
      .then((r) => setMovies(r.data.movies))
      .catch(() => setError('Could not connect to API. Is the backend running on port 8000?'))
  }, [])

  const handleRecommend = async () => {
    if (!selectedMovie) return
    setLoading(true); setError(''); setRecs(null)
    try {
      const r = await axios.post(`${API}/recommend`, { movie: selectedMovie, method, top_n: 10 })
      setRecs(r.data.recommendations)
      setTab('recommend')
    } catch (e) {
      setError(e.response?.data?.detail || 'Request failed.')
    } finally {
      setLoading(false)
    }
  }

  const handleCompare = async () => {
    if (!selectedMovie) return
    setComparing(true); setError(''); setComparison(null)
    try {
      const r = await axios.post(`${API}/compare`, { movie: selectedMovie, top_n: 8 })
      setComparison(r.data.results)
      setTab('compare')
    } catch (e) {
      setError(e.response?.data?.detail || 'Request failed.')
    } finally {
      setComparing(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#F2F3F3]">
      <Header />

      <main className="max-w-7xl mx-auto px-4 py-6 space-y-5">

        {/* Cosine explainer */}
        <CosineSimilarityExplainer />

        {/* Search + Algorithm selector — Cloudscape two-column panel */}
        <div className="bg-white border border-[#E5E7EB] rounded-xl shadow-sm overflow-hidden">
          <div className="bg-[#F9FAFB] border-b border-[#E5E7EB] px-5 py-3">
            <h2 className="text-sm font-semibold text-[#0D1926]">Configuration</h2>
            <p className="text-xs text-[#8D99A5]">Select a movie and choose an embedding method</p>
          </div>
          <div className="p-5 grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div>
              <MovieSearch movies={movies} selected={selectedMovie} onSelect={setSelectedMovie} />
            </div>
            <div className="lg:col-span-2">
              <AlgorithmSelector selected={method} onSelect={setMethod} />
            </div>
          </div>

          {/* Action bar */}
          <div className="px-5 py-4 border-t border-[#E5E7EB] bg-[#F9FAFB] flex flex-col sm:flex-row gap-3">
            <button
              onClick={handleRecommend}
              disabled={!selectedMovie || loading}
              className="flex-1 py-2.5 px-5 rounded-lg font-semibold text-sm text-white transition-all
                disabled:opacity-40 disabled:cursor-not-allowed shadow-sm
                hover:brightness-95 active:brightness-90"
              style={{ backgroundColor: loading ? '#FF9900aa' : '#FF9900' }}
            >
              {loading ? (
                <span className="pulse-glow">Finding similar movies…</span>
              ) : (
                `🎯 Recommend with ${method.toUpperCase()}`
              )}
            </button>

            <button
              onClick={handleCompare}
              disabled={!selectedMovie || comparing}
              className="flex-1 py-2.5 px-5 rounded-lg font-semibold text-sm transition-all border
                bg-white text-[#0972D3] border-[#0972D3]
                hover:bg-[#E8F2FF] active:bg-[#CCE4F8]
                disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {comparing ? (
                <span className="pulse-glow">Running all 5 algorithms…</span>
              ) : (
                '⚡ Compare All 5 Methods'
              )}
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-start gap-3 bg-[#FEF2F2] border border-[#FECACA] rounded-lg p-4 text-sm text-[#DC2626]">
            <span className="text-lg">⚠️</span>
            <span>{error}</span>
          </div>
        )}

        {/* Results tabs */}
        {(recommendations || comparisonData) && (
          <div className="bg-white border border-[#E5E7EB] rounded-xl shadow-sm overflow-hidden">
            {/* Tab bar */}
            <div className="flex border-b border-[#E5E7EB] bg-[#F9FAFB] px-4">
              {recommendations && (
                <button
                  onClick={() => setTab('recommend')}
                  className={`px-4 py-3 text-sm font-semibold border-b-2 transition-colors -mb-px ${
                    tab === 'recommend'
                      ? 'border-[#FF9900] text-[#FF9900]'
                      : 'border-transparent text-[#5F6B7A] hover:text-[#0D1926]'
                  }`}
                >
                  🎬 Recommendations
                </button>
              )}
              {comparisonData && (
                <button
                  onClick={() => setTab('compare')}
                  className={`px-4 py-3 text-sm font-semibold border-b-2 transition-colors -mb-px ${
                    tab === 'compare'
                      ? 'border-[#FF9900] text-[#FF9900]'
                      : 'border-transparent text-[#5F6B7A] hover:text-[#0D1926]'
                  }`}
                >
                  📊 Method Comparison
                </button>
              )}
            </div>

            <div className="p-5">
              {tab === 'recommend' && recommendations && (
                <MovieGrid recommendations={recommendations} method={method} queryMovie={selectedMovie} />
              )}

              {tab === 'compare' && comparisonData && (
                <div className="space-y-6">
                  <ComparisonChart comparisonData={comparisonData} queryMovie={selectedMovie} />

                  {/* Per-method mini grids */}
                  {Object.entries(comparisonData).map(([m, recs]) => {
                    const color = {
                      bow:'#DC2626', tfidf:'#D97706', word2vec:'#059669', glove:'#4F46E5', fasttext:'#DB2777'
                    }[m]
                    const label = {
                      bow:'Bag of Words', tfidf:'TF-IDF', word2vec:'Word2Vec (Google News)',
                      glove:'GloVe', fasttext:'FastText (Wiki News)',
                    }[m]
                    return (
                      <div key={m}>
                        <div className="flex items-center gap-2 mb-3">
                          <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                          <h3 className="font-semibold text-[#0D1926] text-sm">{label}</h3>
                          <span className="text-xs text-[#8D99A5]">top 8</span>
                        </div>
                        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-2">
                          {recs.map((movie, i) => (
                            <div
                              key={`${m}-${i}`}
                              className="bg-[#F9FAFB] border border-[#E5E7EB] rounded-lg p-2 text-center hover:border-[#D1D5DB] transition-colors"
                            >
                              <div className="font-medium text-[#0D1926] line-clamp-2 mb-1 leading-tight text-[11px]">
                                {movie.title}
                              </div>
                              <div className="font-bold text-sm" style={{ color }}>
                                {Math.round(movie.similarity * 100)}%
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
        )}

        {/* How it works footer */}
        <div className="bg-white border border-[#E5E7EB] rounded-xl shadow-sm overflow-hidden">
          <div className="bg-[#F9FAFB] border-b border-[#E5E7EB] px-5 py-3">
            <h3 className="text-sm font-semibold text-[#0D1926]">How Each Method Works</h3>
          </div>
          <div className="p-5 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-3">
            {[
              { name: 'Bag of Words', step: 'Count word occurrences → sparse vector → cosine sim', color: '#DC2626', bg: '#FEF2F2' },
              { name: 'TF-IDF',       step: 'Count × rarity weight → rare words matter more → cosine sim', color: '#D97706', bg: '#FFFBEB' },
              { name: 'Word2Vec',     step: 'Google News 300d vectors → dense semantic space → cosine sim', color: '#059669', bg: '#ECFDF5' },
              { name: 'GloVe',        step: 'Global co-occurrence matrix → 50d pre-trained vectors → cosine sim', color: '#4F46E5', bg: '#EEF2FF' },
              { name: 'FastText',     step: 'Wiki News 300d + char n-grams → handles OOV words → cosine sim', color: '#DB2777', bg: '#FDF2F8' },
            ].map((m) => (
              <div key={m.name} className="rounded-lg p-3 border" style={{ backgroundColor: m.bg, borderColor: m.color + '33' }}>
                <div className="font-semibold text-sm mb-1.5" style={{ color: m.color }}>{m.name}</div>
                <div className="text-[11px] text-[#6B7280] leading-snug">{m.step}</div>
              </div>
            ))}
          </div>
          <div className="px-5 py-3 border-t border-[#E5E7EB] bg-[#F9FAFB] text-center text-xs text-[#8D99A5]">
            Built for GenAI-2026 · Zero to GenAI Engineer · Text to Numbers session
          </div>
        </div>

      </main>
    </div>
  )
}
