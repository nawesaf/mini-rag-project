export type DocumentStatus = 'processing' | 'ready' | 'failed'

export interface DocumentResponse {
  document_id: string
  status?: DocumentStatus
  filename?: string | null
}

export interface QuestionResponse {
  answer: string
}
