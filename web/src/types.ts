export interface RetrievedChunk {
  chunk_id: string
  doc_id: string
  section_path: string
  source_url: string
  score: number
}

export interface TraceEvent {
  node: string
  iteration: number
  verdict: string
  sub_queries: string[]
  retrieved_count: number
  retrieved: RetrievedChunk[]
  answer: string
  citations: string[]
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  citations?: string[]
}
