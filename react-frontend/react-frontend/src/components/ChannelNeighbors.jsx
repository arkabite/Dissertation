import { useEffect, useRef } from "react";
import * as d3 from "d3";

const WIDTH = 420;
const HEIGHT = 420;
const PAD = 30;

const QUERY_COLOR = "#B5761F"; // accent - the channel that was asked about
const NEIGHBOR_COLOR = "#4C9F70"; // distinct green - purely spatial, never a rule level

/**
 * Renders the result of the show_channel_neighbors tool: one queried
 * channel plus its REAL physical neighbors from the study's own montage
 * graph (atlas.mesh_edges) - not a rule-derived relationship, so this
 * deliberately does NOT reuse Topomap's Low/Medium/High/HbO/HbR styling,
 * which would misleadingly imply these points came from a rule.
 *
 * data: {channel, found, position: {x,y,...}, neighbors: [{channel,x,y,...}]}
 * atlas: {channels: {...}, mesh_edges: [...]}
 */
export default function ChannelNeighbors({ data, atlas }) {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!atlas || !data) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    if (!data.found) {
      return; // nothing to draw - the caption below explains why
    }

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
      .range([HEIGHT - PAD, PAD]);

    svg
      .append("circle")
      .attr("cx", WIDTH / 2)
      .attr("cy", HEIGHT / 2)
      .attr("r", Math.min(WIDTH, HEIGHT) / 2 - PAD + 10)
      .attr("fill", "none")
      .attr("stroke", "#ccc")
      .attr("stroke-width", 1);

    // Every atlas channel, faint, for spatial context.
    Object.entries(channels).forEach(([code, c]) => {
      svg
        .append("circle")
        .attr("cx", x(c.x))
        .attr("cy", y(c.y))
        .attr("r", 4)
        .attr("fill", "#e2e2e2")
        .attr("stroke", "#ccc")
        .attr("stroke-width", 0.5);
    });

    const qPos = { x: x(data.position.x), y: y(data.position.y) };

    // Draw a real mesh edge line from the queried channel to each actual
    // neighbor, so the connection itself (not just the two dots) is visible.
    (data.neighbors || []).forEach((n) => {
      svg
        .append("line")
        .attr("x1", qPos.x)
        .attr("y1", qPos.y)
        .attr("x2", x(n.x))
        .attr("y2", y(n.y))
        .attr("stroke", NEIGHBOR_COLOR)
        .attr("stroke-width", 2)
        .attr("stroke-opacity", 0.6);
    });

    // Neighbor points + labels.
    (data.neighbors || []).forEach((n) => {
      const cx = x(n.x);
      const cy = y(n.y);
      svg
        .append("circle")
        .attr("cx", cx)
        .attr("cy", cy)
        .attr("r", 7)
        .attr("fill", NEIGHBOR_COLOR)
        .attr("stroke", "#333")
        .attr("stroke-width", 0.75);
      svg
        .append("text")
        .attr("x", cx)
        .attr("y", cy - 12)
        .attr("text-anchor", "middle")
        .attr("font-size", 9)
        .attr("fill", "#333")
        .text(n.channel);
    });

    // Queried channel on top, larger, distinct color.
    svg
      .append("circle")
      .attr("cx", qPos.x)
      .attr("cy", qPos.y)
      .attr("r", 9)
      .attr("fill", QUERY_COLOR)
      .attr("stroke", "#333")
      .attr("stroke-width", 1);
    svg
      .append("text")
      .attr("x", qPos.x)
      .attr("y", qPos.y - 14)
      .attr("text-anchor", "middle")
      .attr("font-size", 10)
      .attr("font-weight", "bold")
      .attr("fill", "#333")
      .text(data.channel);
  }, [data, atlas]);

  if (!data || !data.found) {
    return (
      <p className="caption">
        {data?.channel
          ? `'${data.channel}' isn't a channel in this study's montage.`
          : "No matching channel was found."}
      </p>
    );
  }

  return (
    <svg
      ref={svgRef}
      className="viz-svg"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="img"
      aria-label={`Physical neighbors of ${data.channel}`}
    />
  );
}
