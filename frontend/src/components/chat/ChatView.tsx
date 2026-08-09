import type { Message } from '../../types/chat'
import { DocumentHeader } from '../document/DocumentHeader'
import { ChatInput } from './ChatInput'
import { MessageList } from './MessageList'

interface ChatViewProps {
  filename: string
  messages: Message[]
  question: string
  error: string
  isAnswering: boolean
  isDeleting: boolean
  onQuestionChange: (question: string) => void
  onSubmit: () => void
  onDelete: () => void
}

export function ChatView({
  filename,
  messages,
  question,
  error,
  isAnswering,
  isDeleting,
  onQuestionChange,
  onSubmit,
  onDelete,
}: ChatViewProps) {
  return (
    <div className="app-shell chat-shell">
      <DocumentHeader filename={filename} isDeleting={isDeleting} onDelete={onDelete} />
      <main className="chat-main">
        <MessageList
          filename={filename}
          messages={messages}
          isAnswering={isAnswering}
          onSuggestionSelect={onQuestionChange}
        />
        <ChatInput
          question={question}
          error={error}
          isAnswering={isAnswering}
          onQuestionChange={onQuestionChange}
          onSubmit={onSubmit}
        />
      </main>
    </div>
  )
}
