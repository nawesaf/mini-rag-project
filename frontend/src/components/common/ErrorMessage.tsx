interface ErrorMessageProps {
  message: string
  variant?: 'upload' | 'inline'
}

export function ErrorMessage({ message, variant = 'inline' }: ErrorMessageProps) {
  if (!message) return null
  return <div className={variant === 'upload' ? 'upload-error' : 'inline-error'} role="alert">{message}</div>
}
