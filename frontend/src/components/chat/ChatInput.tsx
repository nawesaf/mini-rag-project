import type { FormEvent, KeyboardEvent } from 'react'
import { Send } from 'lucide-react'
import { ErrorMessage } from '../common/ErrorMessage'

interface ChatInputProps {
  question: string
  error: string
  isAnswering: boolean
  onQuestionChange: (question: string) => void
  onSubmit: () => void
}

export function ChatInput({ question, error, isAnswering, onQuestionChange, onSubmit }: ChatInputProps) {
  const handleSubmit = (event?: FormEvent) => {
    event?.preventDefault()
    void onSubmit()
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="composer-wrap">
      <ErrorMessage message={error} />
      <form className="composer" onSubmit={handleSubmit}>
        <textarea
          rows={1}
          value={question}
          onChange={(event) => onQuestionChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your PDF…"
          disabled={isAnswering}
          aria-label="Question"
        />
        <button
          type="submit"
          className="send-button"
          disabled={!question.trim() || isAnswering}
          aria-label="Send question"
        >
          <Send />
        </button>
      </form>
      <p className="composer-hint">Enter to send · Shift + Enter for a new line</p>
    </div>
  )
}
