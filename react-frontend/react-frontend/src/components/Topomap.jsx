import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

// Same palette as viz_components.py's LEVEL_COLORS, so a screenshot from
// this app and one from the old Streamlit version read as the same
// visual language.
const LEVEL_COLORS = { Low: "#4C78A8", Medium: "#B0B0B0", High: "#E45756" };
const WIDTH = 420;
const HEIGHT = 420;
const PAD = 30;

/**
 * points: array shaped like build_topomap_payload()'s output
 *   {feature, bare_channel, signal, level, x, y, functional_region,
 *    hemisphere, rule_id, consequent, accuracy}
 * atlas: {channels: {code: {x, y, functional_region, ...}}, mesh_edges: [{a,b}]}
 */
export default function Topomap({ points, atlas, title }) {
  const svgRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);

  useEffect(() => {
    if (!atlas) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const channels = atlas.channels || {};
    const xs = Object.values(channels).map((c) => c.x);
    const ys = Object.values(channels).map((c) => c.y);
    const x = d3
      .scaleLinear()
      .domain([d3.min(xs) - 0.15, d3.max(xs) + 0.15])
      .range([PAD, WIDTH - PAD]);
    const y = d3
      .scaleLinear()
      .domain([d3.min(ys) - 0.15, d3.max(ys) + 0.15])
      .range([HEIGHT - PAD, PAD]); // invert so +y is "up" on the scalp

    // Head outline (rough circle) for spatial context.
    svg
      .append("circle")
      .attr("cx", WIDTH / 2)
      .attr("cy", HEIGHT / 2)
      .attr("r", Math.min(WIDTH, HEIGHT) / 2 - PAD + 10)
      .attr("fill", "none")
      .attr("stroke", "#ccc")
      .attr("stroke-width", 1);

    // Static mesh edges between neighbouring electrodes.
    const g = svg.append("g");
    (atlas.mesh_edges || []).forEach((edge) => {
      const a = channels[edge.a];
      const b = channels[edge.b];
      if (!a || !b) return;
      g.append("line")
        .attr("x1", x(a.x))
        .attr("y1", y(a.y))
        .attr("x2", x(b.x))
        .attr("y2", y(b.y))
        .attr("stroke", "#e0e0e0")
        .attr("stroke-width", 1);
    });

    // Every atlas channel, faint, as background context.
    Object.entries(channels).forEach(([code, c]) => {
      svg
        .append("circle")
        .attr("cx", x(c.x))
        .attr("cy", y(c.y))
        .attr("r", 4.5)
        .attr("fill", "#d8d8d8")
        .attr("stroke", "#bbb")
        .attr("stroke-width", 0.5);
      svg
        .append("text")
        .attr("x", x(c.x))
        .attr("y", y(c.y) - 8)
        .attr("text-anchor", "middle")
        .attr("font-size", 7)
        .attr("fill", "#aaa")
        .text(code);
    });

    // Highlighted rule-derived points on top: circle = HbO, diamond = HbR.
    (points || []).forEach((p) => {
      const cx = x(p.x);
      const cy = y(p.y);
      const color = LEVEL_COLORS[p.level] || "#999";
      const sel =
        p.signal === "HbR"
          ? svg
              .append("rect")
              .attr("x", cx - 6)
              .attr("y", cy - 6)
              .attr("width", 12)
              .attr("height", 12)
              .attr("transform", `rotate(45 ${cx} ${cy})`)
          : svg.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 7);

      sel
        .attr("fill", color)
        .attr("stroke", "#333")
        .attr("stroke-width", 0.75)
        .style("cursor", "pointer")
        .on("mousemove", (event) => {
          setTooltip({
            x: event.clientX + 12,
            y: event.clientY + 12,
            text: `${p.bare_channel} (${p.signal}): ${p.level}\nRule ${p.rule_id} → ${p.consequent}${
              p.accuracy != null ? `\nACC=${Number(p.accuracy).toFixed(3)}` : ""
            }`,
          });
        })
        .on("mouseleave", () => setTooltip(null));

      svg
        .append("text")
        .attr("x", cx)
        .attr("y", cy - 12)
        .attr("text-anchor", "middle")
        .attr("font-size", 9)
        .attr("fill", "#333")
        .text(p.bare_channel);
    });
  }, [points, atlas]);

  return (
    <div>
      {title && <div className="caption">{title}</div>}
      <svg
        ref={svgRef}
        className="viz-svg"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={title || "fNIRS channel topomap"}
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
