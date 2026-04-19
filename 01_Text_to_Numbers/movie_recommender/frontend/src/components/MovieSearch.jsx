import { useState, useMemo } from 'react'

export default function MovieSearch({ movies, selected, onSelect }) {
  const [query, setQuery] = useState('')

  const filtered = useMemo(() => {
    if (!query.trim()) return movies
    const q = query.toLowerCase()
    return movies.filter((m) => m.toLowerCase().includes(q))
  }, [query, movies])

  return (
    <div>
      <label className="block text-xs font-semibold text-[#5F6B7A] uppercase tracking-wide mb-2">
        Search Movie
      </label>

      {/* Google-style search input */}
      <div className="relative mb-2">
        <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#8D99A5]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z" />
        </svg>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Type a movie name…"
          className="w-full bg-white border border-[#D1D5DB] rounded-lg pl-9 pr-9 py-2.5 text-sm text-[#0D1926] placeholder-[#8D99A5] focus:outline-none focus:border-[#0972D3] focus:ring-1 focus:ring-[#0972D3]"
        />
        {query && (
          <button
            onClick={() => setQuery('')}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-[#8D99A5] hover:text-[#0D1926] transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {/* Dropdown list */}
      <div className="bg-white border border-[#D1D5DB] rounded-lg overflow-hidden max-h-52 overflow-y-auto shadow-sm">
        {filtered.length === 0 ? (
          <p className="p-3 text-[#8D99A5] text-sm text-center">No movies found</p>
        ) : (
          filtered.slice(0, 50).map((movie) => (
            <button
              key={movie}
              onClick={() => { onSelect(movie); setQuery(movie) }}
              className={`w-full text-left px-4 py-2 text-sm transition-colors border-b border-[#F3F4F6] last:border-0 flex items-center gap-2 ${
                selected === movie
                  ? 'bg-[#E8F2FF] text-[#0972D3] font-medium'
                  : 'text-[#0D1926] hover:bg-[#F9FAFB]'
              }`}
            >
              {selected === movie && (
                <svg className="w-3.5 h-3.5 text-[#0972D3] shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414L8.414 15l-4.121-4.121a1 1 0 011.414-1.414L8.414 12.172l7.879-7.879a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              )}
              {movie}
            </button>
          ))
        )}
      </div>

      {/* Selected pill */}
      {selected && (
        <div className="flex items-center gap-2 mt-2 px-3 py-2 bg-[#E8F2FF] border border-[#B5D3F5] rounded-lg">
          <svg className="w-4 h-4 text-[#0972D3]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
          </svg>
          <span className="text-[#0972D3] text-sm font-medium truncate">{selected}</span>
        </div>
      )}
    </div>
  )
}
