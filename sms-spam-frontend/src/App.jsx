import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import PredictPage from './pages/PredictPage'
import BatchPage from './pages/BatchPage'
import ModelsPage from './pages/ModelsPage'
import ExplainPage from './pages/ExplainPage'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/"        element={<PredictPage />} />
        <Route path="/batch"   element={<BatchPage />} />
        <Route path="/models"  element={<ModelsPage />} />
        <Route path="/explain" element={<ExplainPage />} />
        <Route path="*"        element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  )
}
