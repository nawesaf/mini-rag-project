import { ChatView } from './components/chat/ChatView'
import { UploadView } from './components/upload/UploadView'
import { useChat } from './hooks/useChat'
import { useDocument } from './hooks/useDocument'

function App() {
  const document = useDocument()
  const chat = useChat(document.documentId)

  const removeDocument = async () => {
    if (await document.remove()) chat.resetChat()
  }

  if (document.phase === 'chat') {
    return (
      <ChatView
        filename={document.file?.name ?? 'Document.pdf'}
        messages={chat.messages}
        question={chat.question}
        error={chat.error || document.error}
        isAnswering={chat.isAnswering}
        isDeleting={document.isDeleting}
        onQuestionChange={chat.setQuestion}
        onSubmit={chat.submitQuestion}
        onDelete={() => void removeDocument()}
      />
    )
  }

  return (
    <UploadView
      file={document.file}
      phase={document.phase}
      error={document.error}
      isBusy={document.isBusy}
      onFileSelect={document.selectFile}
      onUpload={document.upload}
    />
  )
}

export default App
