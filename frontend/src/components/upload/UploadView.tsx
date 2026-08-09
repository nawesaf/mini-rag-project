import { Check, Sparkles } from 'lucide-react'
import type { AppPhase } from '../../hooks/useDocument'
import { Brand } from '../common/Brand'
import { UploadZone } from './UploadZone'

interface UploadViewProps {
  file: File | null
  phase: AppPhase
  error: string
  isBusy: boolean
  onFileSelect: (file?: File) => void
  onUpload: () => void
}

const benefits = ['Quick setup', 'Simple and secure', 'Focused answers']

export function UploadView(props: UploadViewProps) {
  return (
    <div className="app-shell upload-shell">
      <header className="upload-header">
        <Brand />
        <span className="privacy-note">Your document stays private</span>
      </header>

      <main className="upload-main">
        <div className="upload-intro">
          <span className="eyebrow"><Sparkles /> AI-powered document chat</span>
          <h1>Chat with your <span>PDF</span></h1>
          <p>Upload a PDF and ask questions about its content.</p>
        </div>

        <UploadZone {...props} />

        <div className="trust-row" aria-label="Upload benefits">
          {benefits.map((benefit) => <span key={benefit}><i><Check /></i>{benefit}</span>)}
        </div>
      </main>
      <footer>Ask better questions. Find answers faster.</footer>
    </div>
  )
}
