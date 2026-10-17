import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // Forward ARIA API calls to the FastAPI backend so the frontend
      // never needs a CORS exception for port 5173.
      // Override with VITE_BACKEND_TARGET=http://host:8000 for Docker/LAN.
      "/api": {
        target: process.env.VITE_BACKEND_TARGET || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  preview: {
    port: 5173,
  },
  build: {
    // Route-level React.lazy() splits pages into per-route chunks; the
    // shared libs below stay in stable manualChunks so repeat visits and
    // deploys re-download less.
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        // Vite 8 (rolldown) requires the function form of manualChunks.
        // Route-level React.lazy() splits pages into per-route chunks;
        // the shared libs below stay in stable named chunks so repeat
        // visits and deploys re-download less.
        manualChunks: (id) => {
          if (id.includes("node_modules")) {
            if (id.includes("recharts")) return "charts";
            if (id.includes("framer-motion")) return "motion";
            if (/node_modules[\\/](react|react-dom|scheduler)($|[\\/])/.test(id)) return "vendor";
          }
          return undefined;
        },
      },
    },
  },
})
