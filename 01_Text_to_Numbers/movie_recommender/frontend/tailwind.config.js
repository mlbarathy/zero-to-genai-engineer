/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        cinema: {
          bg:      '#0a0a12',
          card:    '#12121e',
          border:  '#1e1e32',
          accent:  '#6d28d9',
          gold:    '#f5c518',
          text:    '#e2e8f0',
          muted:   '#64748b',
        },
      },
      fontFamily: {
        display: ['Georgia', 'serif'],
      },
    },
  },
  plugins: [],
}
