/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        serif: ['"Instrument Serif"', 'Georgia', 'serif'],
        mono: ['"Geist Mono"', 'ui-monospace', 'monospace'],
        sans: ['"Geist"', '-apple-system', 'system-ui', 'sans-serif'],
      },
      colors: {
        parchment: {
          DEFAULT: '#efebe2',
          light: '#f7f4ed',
          card: '#f4efe6',
          dark: '#e3ddd1',
        },
        ink: {
          DEFAULT: '#1d1d1f',
          muted: 'rgba(29, 29, 31, 0.65)',
          subtle: 'rgba(29, 29, 31, 0.40)',
          faint: 'rgba(29, 29, 31, 0.12)',
        },
        terracotta: {
          DEFAULT: '#b8452a',
          hover: '#96351e',
          glow: 'rgba(184, 69, 42, 0.25)',
        },
        brand: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          900: '#0c4a6e',
        }
      }
    },
  },
  plugins: [],
}
