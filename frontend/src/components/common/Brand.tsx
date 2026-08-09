import { FileText } from 'lucide-react'

interface BrandProps {
  compactOnMobile?: boolean
}

export function Brand({ compactOnMobile = false }: BrandProps) {
  return (
    <div className={`brand${compactOnMobile ? ' compact-brand' : ''}`} aria-label="PDF Chat">
      <span className="brand-mark"><FileText /></span>
      <span>PDF Chat</span>
    </div>
  )
}
