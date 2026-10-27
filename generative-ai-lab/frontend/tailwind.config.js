/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        border: "#e4e4e7",
        background: "#ffffff",
        foreground: "#09090b",
        muted: {
          DEFAULT: "#f4f4f5",
          foreground: "#71717a",
        },
        accent: {
          DEFAULT: "#f4f4f5",
          foreground: "#18181b",
        },
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      }
    },
  },
  plugins: [],
}
