import type { TraceEvent } from './types'
import './TracePane.css'

const NODE_LABELS: Record<string, string> = {
  plan_query: 'Planning query',
  retrieve: 'Retrieving chunks',
  grade_evidence: 'Grading evidence',
  generate: 'Generating answer',
  verify_citations: 'Verifying citations',
}

function NodeStep({ event, index }: { event: TraceEvent; index: number }) {
  const label = NODE_LABELS[event.node] ?? event.node

  return (
    <li className="trace-step">
      <div className="trace-step-header">
        <span className="trace-step-index">{index + 1}</span>
        <span className="trace-step-label">{label}</span>
        {event.iteration > 0 && (
          <span className="trace-step-iteration">iteration {event.iteration}</span>
        )}
      </div>

      {event.node === 'grade_evidence' && event.verdict && (
        <div className={`trace-verdict trace-verdict-${event.verdict}`}>
          verdict: {event.verdict}
        </div>
      )}

      {event.node === 'retrieve' && event.retrieved.length > 0 && (
        <ul className="trace-chunks">
          {event.retrieved.map((chunk) => (
            <li key={chunk.chunk_id} className="trace-chunk">
              <span className="trace-chunk-doc">{chunk.doc_id}</span>
              <span className="trace-chunk-section">{chunk.section_path}</span>
              <span className="trace-chunk-score">{chunk.score.toFixed(3)}</span>
            </li>
          ))}
        </ul>
      )}

      {event.node === 'verify_citations' && (
        <div className="trace-citations">
          {event.citations.length > 0
            ? `${event.citations.length} citation${event.citations.length === 1 ? '' : 's'} verified`
            : 'no citations to verify'}
        </div>
      )}
    </li>
  )
}

export function TracePane({ trace, isStreaming }: { trace: TraceEvent[]; isStreaming: boolean }) {
  return (
    <aside className="trace-pane">
      <h2 className="trace-pane-title">Agent trace</h2>
      {trace.length === 0 && !isStreaming && (
        <p className="trace-empty">Ask a question to see the agent's steps here.</p>
      )}
      <ol className="trace-list">
        {trace.map((event, i) => (
          <NodeStep key={i} event={event} index={i} />
        ))}
      </ol>
      {isStreaming && <div className="trace-pending">working…</div>}
    </aside>
  )
}
