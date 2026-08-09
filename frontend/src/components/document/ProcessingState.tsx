import type { AppPhase } from '../../hooks/useDocument'

interface ProcessingStateProps {
  phase: AppPhase
  hasFile: boolean
}

export function ProcessingState({ phase, hasFile }: ProcessingStateProps) {
  const isBusy = phase === 'uploading' || phase === 'processing'
  const label = phase === 'uploading'
    ? 'Uploading PDF…'
    : phase === 'processing'
      ? 'Preparing your document…'
      : hasFile ? 'Start chatting' : 'Select PDF'

  return (
    <>
      {isBusy && <span className="spinner" />}
      {label}
    </>
  )
}
