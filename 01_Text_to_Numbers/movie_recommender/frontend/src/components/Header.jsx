export default function Header() {
  return (
    <header>
      {/* AWS-style top nav bar */}
      <div className="bg-[#232F3E] px-6 py-3 flex items-center gap-4 shadow-md">
        {/* AWS-style logo mark */}
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded flex items-center justify-center"
               style={{ background: 'linear-gradient(135deg,#FF9900,#EC7211)' }}>
            <span className="text-white text-sm font-black">C</span>
          </div>
          <span className="text-white font-semibold text-base tracking-tight">CineMatch</span>
        </div>

        <div className="h-5 w-px bg-[#3A4A5C] mx-1" />

        <span className="text-[#8EA6BE] text-sm">Movie Recommendation Engine</span>

        <div className="ml-auto flex items-center gap-3">
          {[
            { name: 'BoW',      color: '#EF4444' },
            { name: 'TF-IDF',   color: '#F59E0B' },
            { name: 'Word2Vec', color: '#10B981' },
            { name: 'GloVe',    color: '#6366F1' },
            { name: 'FastText', color: '#EC4899' },
          ].map((m) => (
            <span
              key={m.name}
              className="hidden sm:inline-block px-2 py-0.5 rounded text-[11px] font-semibold"
              style={{ backgroundColor: m.color + '22', color: m.color, border: `1px solid ${m.color}44` }}
            >
              {m.name}
            </span>
          ))}
        </div>
      </div>

      {/* Page title band — Google Console style */}
      <div className="bg-white border-b border-[#D1D5DB] px-6 py-5 shadow-sm">
        <div className="max-w-7xl mx-auto flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-2xl">🎬</span>
              <h1 className="text-2xl font-bold text-[#0D1926]">Movie Recommender</h1>
              <span className="ml-2 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-[#E8F2FF] text-[#0972D3] border border-[#B5D3F5]">
                IMDB Top 1000
              </span>
            </div>
            <p className="text-sm text-[#5F6B7A]">
              Compare five text embedding algorithms — from word counting (1954) to neural vectors (2016).
              Pick a movie and see how each method thinks.
            </p>
          </div>
          <div className="hidden md:flex items-center gap-3 mt-1">
            <div className="text-right">
              <div className="text-xs text-[#8D99A5] uppercase tracking-wide font-semibold">Models</div>
              <div className="text-lg font-bold text-[#0D1926]">5</div>
            </div>
            <div className="h-10 w-px bg-[#E5E7EB]" />
            <div className="text-right">
              <div className="text-xs text-[#8D99A5] uppercase tracking-wide font-semibold">Movies</div>
              <div className="text-lg font-bold text-[#0D1926]">1,000</div>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
