import { useState } from 'react'
import { ChatPane } from './ChatPane'
import { TracePane } from './TracePane'
import { useAgentStream } from './useAgentStream'
import type { ChatMessage } from './types'
import './App.css'

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const { trace, isStreaming, error, ask } = useAgentStream()

  async function handleAsk(question: string) {
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      text: question,
    }
    setMessages((prev) => [...prev, userMessage])

    const finalTrace = await ask(question)

    const lastEvent = finalTrace[finalTrace.length - 1]
    if (lastEvent?.node !== 'verify_citations') return

    setMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: 'assistant',
        text: lastEvent.answer || 'No answer generated.',
        citations: lastEvent.citations,
      },
    ])
  }

  return (
    <div className="app-layout">
      <ChatPane messages={messages} isStreaming={isStreaming} onAsk={handleAsk} />
      <TracePane trace={trace} isStreaming={isStreaming} />
      {error && <div className="app-error">{error}</div>}
    </div>
  )
}

export default App
