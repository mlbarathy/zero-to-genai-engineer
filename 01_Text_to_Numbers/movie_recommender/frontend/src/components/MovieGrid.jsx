import MovieCard from './MovieCard'

const ALGO_LABEL = {
  bow:      'Bag of Words',
  tfidf:    'TF-IDF',
  word2vec: 'Word2Vec (Google News)',
  glove:    'GloVe',
  fasttext: 'FastText (Wiki News)',
}

const ALGO_COLORS = {
  bow:      '#DC2626',
  tfidf:    '#D97706',
  word2vec: '#059669',
  glove:    '#4F46E5',
  fasttext: '#DB2777',
}

export default function MovieGrid({ recommendations, method, queryMovie }) {
  if (!recommendations || recommendations.length === 0) return null

  const color = ALGO_COLORS[method] || '#0972D3'

  return (
    <div className="fade-up">
      {/* Section header — Cloudscape panel header style */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-[#E5E7EB]">
        <div>
          <h2 className="text-base font-bold text-[#0D1926]">
            Because you liked{' '}
            <span className="text-[#0972D3]">{queryMovie}</span>
          </h2>
          <p className="text-xs text-[#8D99A5] mt-0.5">
            Ranked by{' '}
            <span className="font-semibold" style={{ color }}>
              {ALGO_LABEL[method] || method}
            </span>
            {' '}cosine similarity
          </p>
        </div>
        <span
          className="text-sm font-semibold px-3 py-1 rounded-full border"
          style={{ color, backgroundColor: color + '14', borderColor: color + '44' }}
        >
          {recommendations.length} results
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
        {recommendations.map((movie, i) => (
          <MovieCard
            key={`${movie.title}-${i}`}
            movie={movie}
            rank={i + 1}
            method={method}
          />
        ))}
      </div>
    </div>
  )
}
