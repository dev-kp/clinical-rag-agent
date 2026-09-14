import { useState } from 'react'
import type { ChatMessage } from './types'
import './ChatPane.css'

interface ChatPaneProps {
  messages: ChatMessage[]
  isStreaming: boolean
  onAsk: (question: string) => void
}

export function ChatPane({ messages, isStreaming, onAsk }: ChatPaneProps) {
  const [input, setInput] = useState('')

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const question = input.trim()
    if (!question || isStreaming) return
    onAsk(question)
    setInput('')
  }

  return (
    <main className="chat-pane">
      <div className="chat-disclaimer">
        Not medical advice. This is a portfolio demo answering only from the
        loaded clinical guideline corpus.
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <p className="chat-empty">
            Ask a question about the loaded CDC / Canadian Immunization Guide corpus.
          </p>
        )}
        {messages.map((m) => (
          <div key={m.id} className={`chat-message chat-message-${m.role}`}>
            <div className="chat-message-text">{m.text}</div>
            {m.citations && m.citations.length > 0 && (
              <div className="chat-message-citations">
                {m.citations.map((c) => (
                  <span key={c} className="chat-citation-badge">
                    {c}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <form className="chat-input-row" onSubmit={handleSubmit}>
        <input
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="e.g. what is the treatment for syphilis in pregnancy?"
          disabled={isStreaming}
        />
        <button className="chat-submit" type="submit" disabled={isStreaming || !input.trim()}>
          {isStreaming ? 'Asking…' : 'Ask'}
        </button>
      </form>
    </main>
  )
}
