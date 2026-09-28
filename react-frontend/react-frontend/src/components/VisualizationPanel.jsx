import { useState } from "react";
import Brain3D from "./Brain3D.jsx";
import Topomap from "./Topomap.jsx";
import ChordDiagram from "./ChordDiagram.jsx";
import ChannelNeighbors from "./ChannelNeighbors.jsx";

/**
 * visualization: null | {type: "topomap"|"chord"|"neighbors", topomap, chord, neighbors}
 * matches ask_with_visualization()'s return shape exactly.
 */
export default function VisualizationPanel({ visualization, atlas, title }) {
  const [view, setView] = useState("2d");

  if (!visualization) return null;

  if (visualization.type === "topomap") {
    return (
      <>
        <div className="view-toggle" role="group" aria-label="Channel map view">
          <button className={view === "2d" ? "active" : ""} onClick={() => setView("2d")}>
            2D scalp map
          </button>
          <button className={view === "3d" ? "active" : ""} onClick={() => setView("3d")}>
            3D brain
          </button>
        </div>
        {view === "3d" ? (
          <Brain3D points={visualization.topomap} />
        ) : (
          <Topomap points={visualization.topomap} atlas={atlas} title={title} />
        )}
        <p className="caption">
          Shape = signal (HbO circle / HbR diamond). Color = activation level.
        </p>
      </>
    );
  }

  if (visualization.type === "chord") {
    return (
      <>
        <ChordDiagram chord={visualization.chord} />
        <p className="caption">Hover a ribbon for the rule(s) behind that connection.</p>
      </>
    );
  }

  if (visualization.type === "neighbors") {
    return (
      <>
        <ChannelNeighbors data={visualization.neighbors} atlas={atlas} />
        <p className="caption">
          Physical adjacency from this study's montage mesh, not a rule-derived
          relationship. Amber marks the queried channel; green marks its physical
          neighbors.
        </p>
      </>
    );
  }

  return null;
}
