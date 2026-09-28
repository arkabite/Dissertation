import { useState } from "react";
import Brain3D from "./Brain3D.jsx";
import Topomap from "./Topomap.jsx";
import ChordDiagram from "./ChordDiagram.jsx";
import AboutPanel from "./AboutPanel.jsx";

function fmt(value, digits = 3, fallback = "n/a") {
  if (value === null || value === undefined || Number.isNaN(value)) return fallback;
  return Number(value).toFixed(digits);
}

export default function Sidebar({ data, atlas, overviewTopomap, overviewChord }) {
  const [view, setView] = useState("topomap");
  const rules = data?.rules || [];
  const cv = data?.cv_performance_estimate || null;
  const targetName = data?.target_name || "target";

  return (
    <aside className="sidebar">
      <AboutPanel />
      <h3>Model overview</h3>
      <p>
        <strong>{rules.length} rules</strong> over target <code>{targetName}</code>
      </p>

      {cv && (
        <>
          <h4>Performance</h4>
          <div className="metrics-row">
            <div className="metric-card">
              <div className="metric-label">Test MCC</div>
              <div className="metric-value">{fmt(cv.test_mcc_mean)}</div>
              <div className="metric-delta">± {fmt(cv.test_mcc_std)}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Test accuracy</div>
              <div className="metric-value">
                {cv.test_accuracy_mean != null
                  ? `${(cv.test_accuracy_mean * 100).toFixed(1)}%`
                  : "n/a"}
              </div>
              <div className="metric-delta">
                ± {cv.test_accuracy_std != null ? `${(cv.test_accuracy_std * 100).toFixed(1)}%` : "n/a"}
              </div>
            </div>
          </div>
          <p className="caption">
            {cv.n_folds || "n/a"}-fold subject-grouped cross-validation. MCC is shown first
            since accuracy alone sits close to the majority-class baseline here.
          </p>
        </>
      )}

      <h4>Rule set</h4>
      <details>
        <summary>View all rules ({rules.length})</summary>
        {rules.map((r) => (
          <div className="rule-entry" key={r.rule_id}>
            <div className="rule-head">Rule {r.rule_id}</div>
            <div>
              IF {r.antecedent}
              <br />
              THEN {targetName} = {r.consequent}
            </div>
            <div className="rule-stats">
              DS {fmt(r.dominance_score)} &middot; ACC {fmt(r.accuracy)}
            </div>
          </div>
        ))}
      </details>

      <h4>Visualizations</h4>
      <details open>
        <summary>Channel map / connectivity (all rules)</summary>
        <div className="view-toggle" role="group" aria-label="Overview visualization">
          <button
            className={view === "topomap" ? "active" : ""}
            onClick={() => setView("topomap")}
          >
            Channel map
          </button>
          <button className={view === "brain3d" ? "active" : ""} onClick={() => setView("brain3d")}>
            3D brain
          </button>
          <button className={view === "chord" ? "active" : ""} onClick={() => setView("chord")}>
            Connectivity
          </button>
        </div>

        {view === "topomap" ? (
          <>
            <p className="caption">
              Every channel referenced across the full rule set, coloured by activation
              level. Shape distinguishes HbO (circle) from HbR (diamond).
            </p>
            {overviewTopomap && overviewTopomap.length > 0 ? (
              <Topomap points={overviewTopomap} atlas={atlas} />
            ) : (
              <p className="caption">No channel data available.</p>
            )}
          </>
        ) : view === "brain3d" ? (
          <>
            <p className="caption">
              Sensor positions shown over the cortical surface. Drag to rotate and scroll to zoom.
            </p>
            {overviewTopomap && overviewTopomap.length > 0 ? (
              <Brain3D points={overviewTopomap} />
            ) : (
              <p className="caption">No channel data available.</p>
            )}
          </>
        ) : (
          <>
            <p className="caption">
              Which channels co-occur together within the same rule. A connection means
              the two channels appear ANDed in a rule's condition — not a measured
              physiological connectivity claim.
            </p>
            <ChordDiagram chord={overviewChord} />
          </>
        )}
      </details>
    </aside>
  );
}
