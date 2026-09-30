import { useRef, useState } from "react";
import Message from "./Message.jsx";
import { ask, BackendError } from "../api.js";

export default function ChatPanel({ atlas }) {
  const [history, setHistory] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState(null);
  const threadRef = useRef(null);

  async function handleSubmit(e) {
    e.preventDefault();
    const q = question.trim();
    if (!q || loading) return;

    setHistory((h) => [...h, { role: "user", content: q }]);
    setQuestion("");
    setLoading(true);
    setToast(null);

    try {
      const result = await ask(q);
      setHistory((h) => [
        ...h,
        {
          role: "assistant",
          content: result.answer || "The model returned an empty answer.",
          animate: true,
          retrieved: result.retrieved || [],
          visualization: result.visualization || null,
          scopeTier: result.scope_tier || "grounded",
          citations: result.citations || [],
          liveLookupStatus: result.live_lookup_status || null,
          consistencyWarning: result.consistency_warning || null,
          consistencyRepaired: !!result.consistency_repaired,
        },
      ]);
      if (result.used_fallback) {
        setToast("The explanation model was unavailable — showing matched rules directly.");
      }
    } catch (err) {
      const msg =
        err instanceof BackendError
          ? `Something went wrong reaching the backend: ${err.message}`
          : `Unexpected error: ${err.message}`;
      setHistory((h) => [...h, { role: "assistant", content: msg, retrieved: [], visualization: null }]);
    } finally {
      setLoading(false);
      // Scroll to latest message once the DOM updates.
      setTimeout(() => {
        if (threadRef.current) {
          threadRef.current.scrollTop = threadRef.current.scrollHeight;
        }
      }, 0);
    }
  }

  return (
    <div className="main-panel">
      <h1>fNIRS Fuzzy Rule Explainer</h1>
      <p className="subtitle">
        Ask about the fuzzy rules behind the classifier — answers are grounded in the
        extracted rule set, with the channels involved shown alongside.
      </p>
      <hr />

      {toast && <div className="toast">{toast}</div>}

      <div className="chat-thread" ref={threadRef}>
        {history.length === 0 && (
          <p className="caption">
            Try: "What does AF7 tell us?", "How are AF7 and C6h related?", or "What's the
            capital of France?" (to see the out-of-scope guard).
          </p>
        )}
        {history.map((m, i) => (
          <Message
            key={i}
            role={m.role}
            content={m.content}
            animate={m.animate}
            retrieved={m.retrieved}
            visualization={m.visualization}
            atlas={atlas}
            scopeTier={m.scopeTier}
            citations={m.citations}
            liveLookupStatus={m.liveLookupStatus}
            consistencyWarning={m.consistencyWarning}
            consistencyRepaired={m.consistencyRepaired}
          />
        ))}
        {loading && <p className="caption">Thinking...</p>}
      </div>

      <form className="chat-input-row" onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about the model's rules..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
