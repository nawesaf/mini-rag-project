import { useEffect, useState } from 'react'
import { deleteDocument, getDocument, uploadDocument } from '../services/api'

export type AppPhase = 'upload' | 'uploading' | 'processing' | 'chat'

const POLL_INTERVAL_MS = 2500

function getErrorMessage(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback
}

export function useDocument() {
  const [phase, setPhase] = useState<AppPhase>('upload')
  const [file, setFile] = useState<File | null>(null)
  const [documentId, setDocumentId] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [isDeleting, setIsDeleting] = useState(false)

  useEffect(() => {
    if (phase !== 'processing' || !documentId) return

    let cancelled = false
    let timeout: number | undefined

    const poll = async () => {
      try {
        const result = await getDocument(documentId)
        if (cancelled) return
        if (result.status === 'failed') {
          setError('The document could not be processed. Please try another PDF.')
          setPhase('upload')
          return
        }
        if (result.status === 'ready') {
          setPhase('chat')
          return
        }
        timeout = window.setTimeout(poll, POLL_INTERVAL_MS)
      } catch (pollError) {
        if (!cancelled) {
          setError(getErrorMessage(pollError, 'Could not check the document status.'))
          setPhase('upload')
        }
      }
    }

    timeout = window.setTimeout(poll, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      if (timeout) window.clearTimeout(timeout)
    }
  }, [documentId, phase])

  const selectFile = (candidate?: File) => {
    setError('')
    if (!candidate) return

    const isPdf = candidate.type === 'application/pdf' || candidate.name.toLowerCase().endsWith('.pdf')
    if (!isPdf) {
      setFile(null)
      setError('Please choose a PDF file.')
      return
    }
    setFile(candidate)
  }

  const upload = async () => {
    if (!file) return

    setError('')
    setPhase('uploading')
    try {
      const result = await uploadDocument(file)
      setDocumentId(result.document_id)
      if (result.status === 'failed') {
        setError('The document could not be processed. Please try another PDF.')
        setPhase('upload')
      } else {
        setPhase(result.status === 'ready' ? 'chat' : 'processing')
      }
    } catch (uploadError) {
      setError(getErrorMessage(uploadError, 'The PDF could not be uploaded.'))
      setPhase('upload')
    }
  }

  const remove = async () => {
    if (!documentId || isDeleting) return false

    setIsDeleting(true)
    setError('')
    try {
      await deleteDocument(documentId)
      setPhase('upload')
      setFile(null)
      setDocumentId(null)
      return true
    } catch (deleteError) {
      setError(getErrorMessage(deleteError, 'The document could not be removed.'))
      return false
    } finally {
      setIsDeleting(false)
    }
  }

  return {
    phase,
    file,
    documentId,
    error,
    isDeleting,
    isBusy: phase === 'uploading' || phase === 'processing',
    selectFile,
    upload,
    remove,
  }
}
