/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#1f2933",
        panel: "#f7f8f5",
        safety: "#f3c316",
        warning: "#c2410c",
        danger: "#b91c1c",
        steel: "#51606d",
        mint: "#0f766e"
      },
      boxShadow: {
        soft: "0 14px 40px rgba(31, 41, 51, 0.08)"
      }
    }
  },
  plugins: []
};
