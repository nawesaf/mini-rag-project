import { useEffect, useRef } from 'react'
import { Sparkles } from 'lucide-react'
import type { Message } from '../../types/chat'
import { ChatMessage } from './ChatMessage'

interface MessageListProps {
  filename: string
  messages: Message[]
  isAnswering: boolean
  onSuggestionSelect: (suggestion: string) => void
}

const suggestions = ['Summarize the key points', 'What are the main conclusions?']

export function MessageList({ filename, messages, isAnswering, onSuggestionSelect }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, isAnswering])

  return (
    <section className="conversation" aria-live="polite">
      {messages.length === 0 ? (
        <div className="empty-chat">
          <span className="empty-icon"><Sparkles /></span>
          <h1>Your PDF is ready</h1>
          <p>Ask anything about <strong>{filename}</strong>.</p>
          <div className="suggestions">
            {suggestions.map((suggestion) => (
              <button key={suggestion} type="button" onClick={() => onSuggestionSelect(suggestion)}>
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="messages">
          {messages.map((message) => <ChatMessage key={message.id} message={message} />)}
          {isAnswering && (
            <article className="message assistant loading-message">
              <span className="assistant-avatar"><Sparkles /></span>
              <div className="message-body">
                <span className="message-label">PDF Chat</span>
                <span className="typing" aria-label="Generating answer"><i /><i /><i /></span>
              </div>
            </article>
          )}
        </div>
      )}
      <div ref={messagesEndRef} />
    </section>
  )
}
