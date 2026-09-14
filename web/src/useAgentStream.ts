import { useCallback, useState } from 'react'
import type { TraceEvent } from './types'

const API_BASE = 'http://localhost:8000'

/**
 * Streams one /ask call's node-by-node trace. Uses fetch + a stream
 * reader rather than the native EventSource API because EventSource
 * only supports GET, and /ask takes the question as a POST body.
 *
 * ask() returns the completed trace array directly (rather than relying
 * on the caller to read state right after awaiting, which wouldn't yet
 * reflect the final setTrace call) so the caller can pull the final
 * answer out immediately without a separate effect watching trace.
 */
export function useAgentStream() {
  const [trace, setTrace] = useState<TraceEvent[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const ask = useCallback(async (question: string): Promise<TraceEvent[]> => {
    setTrace([])
    setError(null)
    setIsStreaming(true)

    const collected: TraceEvent[] = []

    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })

      if (!response.ok || !response.body) {
        throw new Error(`Request failed: ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      for (;;) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const events = buffer.split('\n\n')
        buffer = events.pop() ?? ''

        for (const rawEvent of events) {
          if (rawEvent.startsWith('event: done')) continue
          const dataLine = rawEvent.split('\n').find((l) => l.startsWith('data: '))
          if (!dataLine) continue

          const event = JSON.parse(dataLine.slice('data: '.length)) as TraceEvent
          collected.push(event)
          setTrace((prev) => [...prev, event])
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setIsStreaming(false)
    }

    return collected
  }, [])

  return { trace, isStreaming, error, ask }
}
