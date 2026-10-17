import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './figma/index.css'
import App from './figma/App.tsx'
import { ToastProvider } from './components/Toast.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ToastProvider>
      <App />
    </ToastProvider>
  </StrictMode>,
)
