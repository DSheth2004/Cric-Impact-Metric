import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  // API_TARGET lets you point the dev proxy at another backend, e.g. API_TARGET=http://127.0.0.1:8010
  const env = loadEnv(mode, '.', '')
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: { '/api': env.API_TARGET || 'http://127.0.0.1:8000' },
    },
  }
})
