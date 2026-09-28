import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

const WIDTH = 420;
const HEIGHT = 420;
const RADIUS = 160;

const HEMI_COLOR = {
  "Left hemisphere (by standard 10-10 naming convention)": "#4C78A8",
  "Right hemisphere (by standard 10-10 naming convention)": "#E45756",
};

/**
 * chord: {nodes: [{channel, functional_region, hemisphere}],
 *         edges: [{source, target, rule_id, consequent, dominance_score, accuracy}]}
 * "Connected" here means literal AND-co-occurrence within a rule's
 * antecedent, not a measured physiological connectivity claim — same
 * framing as the Streamlit caption.
 */
export default function ChordDiagram({ chord }) {
  const svgRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);

  useEffect(() => {
    if (!chord || !chord.nodes || chord.nodes.length === 0) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const cx = WIDTH / 2;
    const cy = HEIGHT / 2;
    const nodes = chord.nodes;
    const angle = d3.scalePoint(
      nodes.map((n) => n.channel),
      [0, 2 * Math.PI * (1 - 1 / nodes.length)]
    );

    const pos = {};
    nodes.forEach((n) => {
      const a = angle(n.channel) - Math.PI / 2;
      pos[n.channel] = { x: cx + RADIUS * Math.cos(a), y: cy + RADIUS * Math.sin(a) };
    });

    const g = svg.append("g");

    // Ribbons (edges) drawn first so nodes sit on top.
    (chord.edges || []).forEach((e) => {
      const s = pos[e.source];
      const t = pos[e.target];
      if (!s || !t) return;
      g.append("path")
        .attr(
          "d",
          d3.line().curve(d3.curveBasis)([
            [s.x, s.y],
            [cx, cy],
            [t.x, t.y],
          ])
        )
        .attr("fill", "none")
        .attr("stroke", "#B5761F")
        .attr("stroke-opacity", 0.35)
        .attr("stroke-width", 1.5)
        .style("cursor", "pointer")
        .on("mousemove", (event) => {
          setTooltip({
            x: event.clientX + 12,
            y: event.clientY + 12,
            text: `${e.source} \u2194 ${e.target}\nRule ${e.rule_id} \u2192 ${e.consequent}${
              e.accuracy != null ? `\nACC=${Number(e.accuracy).toFixed(3)}` : ""
            }${
              e.dominance_score != null
                ? `\nDS=${Number(e.dominance_score).toFixed(3)}`
                : ""
            }`,
          });
        })
        .on("mouseleave", () => setTooltip(null));
    });

    // Nodes.
    nodes.forEach((n) => {
      const p = pos[n.channel];
      const color = HEMI_COLOR[n.hemisphere] || "#7A6E60";
      g.append("circle")
        .attr("cx", p.x)
        .attr("cy", p.y)
        .attr("r", 7)
        .attr("fill", color)
        .attr("stroke", "#333")
        .attr("stroke-width", 0.75);
      const a = angle(n.channel) - Math.PI / 2;
      const lx = cx + (RADIUS + 16) * Math.cos(a);
      const ly = cy + (RADIUS + 16) * Math.sin(a);
      g.append("text")
        .attr("x", lx)
        .attr("y", ly)
        .attr("text-anchor", "middle")
        .attr("font-size", 9)
        .attr("fill", "#333")
        .text(n.channel);
    });
  }, [chord]);

  if (!chord || !chord.edges || chord.edges.length === 0) {
    return <p className="caption">No co-occurring channel pairs found.</p>;
  }

  return (
    <div>
      <svg
        ref={svgRef}
        className="viz-svg"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="fNIRS channel connectivity diagram"
      />
      {tooltip && (
        <div
          className="viz-tooltip"
          style={{ left: tooltip.x, top: tooltip.y, whiteSpace: "pre-line" }}
        >
          {tooltip.text}
        </div>
      )}
    </div>
  );
}
