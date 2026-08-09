import { FileText, Trash2 } from 'lucide-react'
import { Brand } from '../common/Brand'

interface DocumentHeaderProps {
  filename: string
  isDeleting: boolean
  onDelete: () => void
}

export function DocumentHeader({ filename, isDeleting, onDelete }: DocumentHeaderProps) {
  return (
    <header className="topbar">
      <Brand compactOnMobile />
      <div className="document-pill" title={filename}>
        <FileText />
        <span>{filename}</span>
      </div>
      <button className="change-button" type="button" onClick={onDelete} disabled={isDeleting}>
        <Trash2 />
        <span>{isDeleting ? 'Removing…' : 'Change PDF'}</span>
      </button>
    </header>
  )
}
