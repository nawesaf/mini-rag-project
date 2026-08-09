import type { DocumentResponse, QuestionResponse } from '../types/api'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, options)
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const payload = (await response.json()) as { detail?: string }
      if (payload.detail) message = payload.detail
    } catch {
      // Keep the status-based message when the response has no JSON body.
    }
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

export function uploadDocument(file: File): Promise<DocumentResponse> {
  const body = new FormData()
  body.append('file', file)
  return request<DocumentResponse>('/document', { method: 'POST', body })
}

export function getDocument(documentId: string): Promise<DocumentResponse> {
  return request<DocumentResponse>(`/document/${encodeURIComponent(documentId)}`)
}

export function askQuestion(documentId: string, question: string): Promise<QuestionResponse> {
  return request<QuestionResponse>(`/document/${encodeURIComponent(documentId)}/questions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
}

export function deleteDocument(documentId: string): Promise<DocumentResponse> {
  return request<DocumentResponse>(`/document/${encodeURIComponent(documentId)}`, {
    method: 'DELETE',
  })
}
