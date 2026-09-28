import VisualizationPanel from "./VisualizationPanel.jsx";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function fmt(value, digits = 3, fallback = "n/a") {
  if (value === null || value === undefined || Number.isNaN(value)) return fallback;
  return Number(value).toFixed(digits);
}

export default function Message({
  role,
  content,
  retrieved,
  visualization,
  atlas,
  scopeTier,
  citations,
  consistencyWarning,
  consistencyRepaired,
}) {
  return (
    <div className={`chat-message ${role}`}>
      <div className="role">{role}</div>
      {role === "assistant" && scopeTier === "general" && (
        <div className="scope-badge scope-badge-general">General knowledge</div>
      )}
      {role === "assistant" && scopeTier === "out_of_scope" && (
        <div className="scope-badge scope-badge-refused">Out of scope</div>
      )}
      {role === "assistant" ? (
        <div className="markdown-body">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        </div>
      ) : (
        <div>{content}</div>
      )}

      {consistencyWarning && (
        <div className="consistency-warning" role="alert">{consistencyWarning}</div>
      )}
      {consistencyRepaired && !consistencyWarning && (
        <p className="caption">
          This answer was automatically corrected after a consistency check against the rules.
        </p>
      )}

      {citations && citations.length > 0 && (
        <div className="citations-block">
          <div className="citations-label">Sources</div>
          {citations.map((c) => (
            <p key={c.url} className="caption">
              <a href={c.url} target="_blank" rel="noreferrer">
                {c.citation}
              </a>
            </p>
          ))}
        </div>
      )}

      {visualization && (
        <div style={{ marginTop: "0.75rem" }}>
          <VisualizationPanel
            visualization={visualization}
            atlas={atlas}
            title="Channels behind this answer"
          />
        </div>
      )}

      {retrieved && retrieved.length > 0 && (
        <details style={{ marginTop: "0.5rem" }}>
          <summary>Rules used ({retrieved.length})</summary>
          {retrieved.map((r) => (
            <p key={r.rule_id} className="caption">
              Rule {r.rule_id}: {r.antecedent} &rarr; {r.consequent} (ACC={fmt(r.accuracy)})
            </p>
          ))}
        </details>
      )}
    </div>
  );
}
