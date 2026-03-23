/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0E131F",
        mist: "#EFF4F8",
        coral: "#FF6B4A",
        moss: "#6B8E23",
        ocean: "#146C94",
        sun: "#F2C14E"
      },
      fontFamily: {
        display: ["'Space Grotesk'", "system-ui", "sans-serif"],
        body: ["'Sora'", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "monospace"]
      },
      boxShadow: {
        card: "0 12px 40px rgba(20, 108, 148, 0.2)"
      }
    },
  },
  plugins: [],
};
