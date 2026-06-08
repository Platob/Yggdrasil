/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        bg: '#0a0e1a',
        surface: '#111827',
        border: '#1f2937',
        accent: '#3b82f6',
        green: '#10b981',
        red: '#ef4444',
        muted: '#6b7280',
      },
    },
  },
  plugins: [],
};
