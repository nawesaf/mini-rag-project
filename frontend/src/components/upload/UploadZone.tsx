import { useRef, useState, type ChangeEvent, type DragEvent, type KeyboardEvent } from 'react'
import { FileText, Upload } from 'lucide-react'
import type { AppPhase } from '../../hooks/useDocument'
import { ErrorMessage } from '../common/ErrorMessage'
import { ProcessingState } from '../document/ProcessingState'

interface UploadZoneProps {
  file: File | null
  phase: AppPhase
  error: string
  isBusy: boolean
  onFileSelect: (file?: File) => void
  onUpload: () => void
}

export function UploadZone({ file, phase, error, isBusy, onFileSelect, onUpload }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const openFileDialog = () => {
    if (!isBusy) fileInputRef.current?.click()
  }

  const handleFileInput = (event: ChangeEvent<HTMLInputElement>) => {
    onFileSelect(event.target.files?.[0])
    event.target.value = ''
  }

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(false)
    if (!isBusy) onFileSelect(event.dataTransfer.files?.[0])
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' || event.key === ' ') openFileDialog()
  }

  const handlePrimaryAction = () => {
    if (file) void onUpload()
    else openFileDialog()
  }

  return (
    <section className="upload-card">
      <div
        className={`dropzone ${isDragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
        onDragEnter={(event) => { event.preventDefault(); if (!isBusy) setIsDragging(true) }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setIsDragging(false)
        }}
        onDrop={handleDrop}
        onClick={openFileDialog}
        role="button"
        tabIndex={isBusy ? -1 : 0}
        onKeyDown={handleKeyDown}
        aria-label="Choose a PDF file or drag it here"
      >
        <input ref={fileInputRef} type="file" accept="application/pdf,.pdf" onChange={handleFileInput} hidden />
        {file ? (
          <>
            <span className="file-icon"><FileText /></span>
            <div className="selected-file">
              <strong>{file.name}</strong>
              <span>{(file.size / 1024 / 1024).toFixed(2)} MB · PDF</span>
            </div>
            {!isBusy && <span className="replace-link">Choose another</span>}
          </>
        ) : (
          <>
            <span className="upload-icon"><Upload /></span>
            <h2>{isDragging ? 'Drop your PDF here' : 'Choose a PDF file'}</h2>
            <p>or drag and drop it here</p>
            <span className="file-type">PDF only · One file at a time</span>
          </>
        )}
      </div>

      <ErrorMessage message={error} variant="upload" />
      <button className="primary-button" type="button" onClick={handlePrimaryAction} disabled={isBusy}>
        <ProcessingState phase={phase} hasFile={Boolean(file)} />
      </button>
      {isBusy && <p className="processing-note">This may take a moment. Please keep this page open.</p>}
    </section>
  )
}
