import { Sparkles } from 'lucide-react'
import type { Message } from '../../types/chat'

interface ChatMessageProps {
  message: Message
}

export function ChatMessage({ message }: ChatMessageProps) {
  return (
    <article className={`message ${message.role}`}>
      {message.role === 'assistant' && <span className="assistant-avatar"><Sparkles /></span>}
      <div className="message-body">
        <span className="message-label">{message.role === 'user' ? 'You' : 'PDF Chat'}</span>
        <p>{message.content}</p>
      </div>
    </article>
  )
}
