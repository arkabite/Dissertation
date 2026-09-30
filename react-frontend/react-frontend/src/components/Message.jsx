import VisualizationPanel from "./VisualizationPanel.jsx";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useEffect, useState } from "react";

function fmt(value, digits = 3, fallback = "n/a") {
  if (value === null || value === undefined || Number.isNaN(value)) return fallback;
  return Number(value).toFixed(digits);
}

export default function Message({
  role,
  content,
  animate = false,
  retrieved,
  visualization,
  atlas,
  scopeTier,
  citations,
  liveLookupStatus,
  consistencyWarning,
  consistencyRepaired,
}) {
  const [visibleLength, setVisibleLength] = useState(animate ? 0 : content.length);

  useEffect(() => {
    if (!animate) {
      setVisibleLength(content.length);
      return;
    }

    const wordEnds = [...content.matchAll(/\S+\s*/g)].map((match) => match.index + match[0].length);
    let nextWord = 0;
    setVisibleLength(0);

    const timer = setInterval(() => {
      nextWord += 1;
      setVisibleLength(wordEnds[nextWord - 1] ?? content.length);
      if (nextWord >= wordEnds.length) clearInterval(timer);
    }, 28);

    return () => clearInterval(timer);
  }, [animate, content]);

  const displayedContent = animate ? content.slice(0, visibleLength) : content;

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
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{displayedContent}</ReactMarkdown>
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
              {!c.vetted && <strong>Unvetted live result: </strong>}
              <a href={c.url} target="_blank" rel="noreferrer">
                {c.citation}
              </a>
            </p>
          ))}
        </div>
      )}

      {role === "assistant" && scopeTier === "general" && (!citations || citations.length === 0) && liveLookupStatus && (
        <p className="caption">
          {liveLookupStatus === "provider_error"
            ? "Live source lookup failed for this answer."
            : liveLookupStatus === "no_qualifying_results"
              ? "Live source lookup found no matching papers."
              : "Live source lookup was not attempted for this answer."}
        </p>
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
