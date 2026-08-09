import { useRef, useState } from 'react'
import { askQuestion } from '../services/api'
import type { Message } from '../types/chat'

function getErrorMessage(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback
}

export function useChat(documentId: string | null) {
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [error, setError] = useState('')
  const [isAnswering, setIsAnswering] = useState(false)
  const nextMessageId = useRef(1)

  const submitQuestion = async () => {
    const trimmedQuestion = question.trim()
    if (!trimmedQuestion || !documentId || isAnswering) return

    setMessages((current) => [
      ...current,
      { id: nextMessageId.current++, role: 'user', content: trimmedQuestion },
    ])
    setQuestion('')
    setError('')
    setIsAnswering(true)
    try {
      const result = await askQuestion(documentId, trimmedQuestion)
      setMessages((current) => [
        ...current,
        { id: nextMessageId.current++, role: 'assistant', content: result.answer },
      ])
    } catch (questionError) {
      setError(getErrorMessage(questionError, 'Your question could not be answered.'))
    } finally {
      setIsAnswering(false)
    }
  }

  const resetChat = () => {
    setMessages([])
    setQuestion('')
    setError('')
    setIsAnswering(false)
    nextMessageId.current = 1
  }

  return {
    messages,
    question,
    error,
    isAnswering,
    setQuestion,
    submitQuestion,
    resetChat,
  }
}
