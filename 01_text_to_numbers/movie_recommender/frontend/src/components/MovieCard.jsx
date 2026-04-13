import { useState } from 'react'

const ALGO_COLORS = {
  bow:      '#DC2626',
  tfidf:    '#D97706',
  word2vec: '#059669',
  glove:    '#4F46E5',
  fasttext: '#DB2777',
  sbert:    '#0972D3',
}

const ALGO_BG = {
  bow:      '#FEF2F2',
  tfidf:    '#FFFBEB',
  word2vec: '#ECFDF5',
  glove:    '#EEF2FF',
  fasttext: '#FDF2F8',
  sbert:    '#E8F2FF',
}

export default function MovieCard({ movie, rank, method }) {
  const [imgError, setImgError] = useState(false)
  const [expanded, setExpanded] = useState(false)

  const pct = Math.round(movie.similarity * 100)
  const color = ALGO_COLORS[method] || '#0972D3'
  const bg    = ALGO_BG[method]    || '#E8F2FF'

  return (
    <div className="movie-card bg-white border border-[#E5E7EB] rounded-xl overflow-hidden flex flex-col shadow-sm">
      {/* Poster */}
      <div className="relative aspect-[2/3] bg-[#F3F4F6] overflow-hidden">
        {!imgError && movie.poster && movie.poster !== 'nan' ? (
          <img
            src={movie.poster}
            alt={movie.title}
            className="w-full h-full object-cover"
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="poster-placeholder w-full h-full flex items-center justify-center">
            <div className="text-center text-[#9CA3AF] px-3">
              <div className="text-4xl mb-2">🎬</div>
              <div className="text-xs text-center leading-snug">{movie.title}</div>
            </div>
          </div>
        )}

        {/* Rank badge — AWS orange */}
        <div
          className="absolute top-2 left-2 w-6 h-6 rounded-full flex items-center justify-center text-[11px] font-bold text-white shadow"
          style={{ backgroundColor: '#FF9900' }}
        >
          {rank}
        </div>

        {/* IMDb rating */}
        {movie.rating && movie.rating !== 'N/A' && (
          <div className="absolute top-2 right-2 flex items-center gap-0.5 bg-black/60 backdrop-blur-sm rounded px-1.5 py-0.5">
            <span className="text-[#F5C518] text-[10px]">★</span>
            <span className="text-white text-[10px] font-bold">{movie.rating}</span>
          </div>
        )}

        {/* Match score bar */}
        <div className="absolute bottom-0 left-0 right-0 bg-white/90 backdrop-blur-sm px-3 py-1.5">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] text-[#5F6B7A] font-medium">Match</span>
            <span className="text-[10px] font-bold" style={{ color }}>{pct}%</span>
          </div>
          <div className="bg-[#E5E7EB] rounded-full h-1.5">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${pct}%`, backgroundColor: color }}
            />
          </div>
        </div>
      </div>

      {/* Info */}
      <div className="p-3 flex flex-col flex-1">
        <h3 className="text-sm font-bold text-[#0D1926] leading-tight mb-1 line-clamp-2">
          {movie.title}
        </h3>
        <div className="flex items-center gap-1.5 text-[11px] text-[#8D99A5] mb-2 flex-wrap">
          <span>{movie.year}</span>
          <span>·</span>
          <span
            className="px-1.5 py-0.5 rounded text-[10px] font-semibold"
            style={{ backgroundColor: bg, color }}
          >
            {movie.genre?.split(',')[0]}
          </span>
        </div>

        <button
          onClick={() => setExpanded(!expanded)}
          className="text-[11px] text-[#0972D3] hover:text-[#0551A3] text-left transition-colors mt-auto font-medium"
        >
          {expanded ? '▲ Hide details' : '▼ Details'}
        </button>

        {expanded && (
          <div className="mt-2 space-y-1 text-xs text-[#5F6B7A] fade-up border-t border-[#F3F4F6] pt-2">
            <p><span className="text-[#8D99A5]">Director:</span> {movie.director}</p>
            {movie.stars?.length > 0 && (
              <p><span className="text-[#8D99A5]">Stars:</span> {movie.stars.join(', ')}</p>
            )}
            <p><span className="text-[#8D99A5]">Runtime:</span> {movie.runtime}</p>
            <p className="text-[#8D99A5] leading-snug line-clamp-3">{movie.overview}</p>
          </div>
        )}
      </div>
    </div>
  )
}
