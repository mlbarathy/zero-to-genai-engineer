const ALGORITHMS = [
  {
    id: 'bow',
    name: 'Bag of Words',
    year: '1954',
    icon: '🛍️',
    color: '#DC2626',
    bg: '#FEF2F2',
    border: '#FECACA',
    tagline: 'Count words — order ignored',
    pro: 'Fast, interpretable',
    con: '"dog bites man" = "man bites dog"',
  },
  {
    id: 'tfidf',
    name: 'TF-IDF',
    year: '1972',
    icon: '⚖️',
    color: '#D97706',
    bg: '#FFFBEB',
    border: '#FDE68A',
    tagline: 'Rare words → higher weight',
    pro: 'Down-weights "the", "a" automatically',
    con: 'Still no semantic understanding',
  },
  {
    id: 'word2vec',
    name: 'Word2Vec',
    year: '2013',
    icon: '🧠',
    color: '#059669',
    bg: '#ECFDF5',
    border: '#A7F3D0',
    tagline: 'Google News — 3M vocab, 300d',
    pro: 'king − man + woman ≈ queen',
    con: 'No subword model for OOV words',
  },
  {
    id: 'glove',
    name: 'GloVe',
    year: '2014',
    icon: '🌍',
    color: '#4F46E5',
    bg: '#EEF2FF',
    border: '#C7D2FE',
    tagline: 'Global statistics + local context',
    pro: 'Pre-trained on 6B Wikipedia words',
    con: 'One vector per word ("bank" ambiguous)',
  },
  {
    id: 'fasttext',
    name: 'FastText',
    year: '2016',
    icon: '⚡',
    color: '#DB2777',
    bg: '#FDF2F8',
    border: '#FBCFE8',
    tagline: 'Wiki News — 1M vocab + subwords',
    pro: '"cinematographic" works even unseen',
    con: 'Subword noise on short texts',
  },
]

export default function AlgorithmSelector({ selected, onSelect }) {
  return (
    <div>
      <label className="block text-xs font-semibold text-[#5F6B7A] uppercase tracking-wide mb-2">
        Embedding Method
      </label>
      <p className="text-xs text-[#8D99A5] mb-3">
        5 methods · from word counting (1954) to neural embeddings (2016)
      </p>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
        {ALGORITHMS.map((algo) => {
          const isActive = selected === algo.id
          return (
            <button
              key={algo.id}
              onClick={() => onSelect(algo.id)}
              className={`relative flex flex-col p-3 rounded-lg border text-left transition-all duration-150 ${
                isActive ? 'shadow-md' : 'bg-white border-[#D1D5DB] hover:border-[#9CA3AF] hover:shadow-sm'
              }`}
              style={isActive ? { backgroundColor: algo.bg, borderColor: algo.color, borderWidth: 2 } : {}}
            >
              {/* Year badge */}
              <span
                className="absolute top-2 right-2 text-[10px] font-bold px-1.5 py-0.5 rounded"
                style={{ backgroundColor: algo.bg, color: algo.color, border: `1px solid ${algo.border}` }}
              >
                {algo.year}
              </span>

              <span className="text-xl mb-1.5">{algo.icon}</span>
              <span
                className="font-bold text-sm mb-0.5"
                style={{ color: isActive ? algo.color : '#0D1926' }}
              >
                {algo.name}
              </span>
              <span className="text-[11px] text-[#8D99A5] leading-snug">{algo.tagline}</span>

              {isActive && (
                <div
                  className="mt-2 pt-2 space-y-1 border-t fade-up"
                  style={{ borderColor: algo.border }}
                >
                  <p className="text-[10px] text-[#059669] font-medium">✓ {algo.pro}</p>
                  <p className="text-[10px] text-[#DC2626] font-medium">✗ {algo.con}</p>
                </div>
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}
