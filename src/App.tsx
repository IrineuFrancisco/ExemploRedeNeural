import React, { useState, useEffect, useRef, useCallback } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────
interface Inputs {
  yellow: number;
  red: number;
  rough: number;
}

interface NodeWeights {
  yellow: number;
  red: number;
  rough: number;
}

interface OutputResult {
  apple: number;
  orange: number;
}

interface Connection {
  from: keyof typeof SVG_PTS;
  to: keyof typeof SVG_PTS;
  weight: number;
  inputKey: keyof Inputs;
}

interface SliderProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  trackColor: string;
}

interface OutputNodeProps {
  label: string;
  value: number;
  glowColor: string;
  emoji: string;
}

interface DisplayState {
  inputs: Inputs;
  outputs: OutputResult;
}

// ─── Constants ────────────────────────────────────────────────────────────────
const SVG_PTS = {
  yellow: { x: 90,  y: 80  },
  red:    { x: 90,  y: 200 },
  rough:  { x: 90,  y: 320 },
  apple:  { x: 510, y: 140 },
  orange: { x: 510, y: 260 },
} as const;

const APPLE_WEIGHTS: NodeWeights  = { yellow: -0.5, red:  1.2, rough: -0.8 };
const ORANGE_WEIGHTS: NodeWeights = { yellow:  1.1, red: -0.6, rough:  0.9 };

const CONNECTIONS: Connection[] = [
  { from: "yellow", to: "apple",  weight: APPLE_WEIGHTS.yellow,  inputKey: "yellow" },
  { from: "red",    to: "apple",  weight: APPLE_WEIGHTS.red,     inputKey: "red"    },
  { from: "rough",  to: "apple",  weight: APPLE_WEIGHTS.rough,   inputKey: "rough"  },
  { from: "yellow", to: "orange", weight: ORANGE_WEIGHTS.yellow, inputKey: "yellow" },
  { from: "red",    to: "orange", weight: ORANGE_WEIGHTS.red,    inputKey: "red"    },
  { from: "rough",  to: "orange", weight: ORANGE_WEIGHTS.rough,  inputKey: "rough"  },
];

const WEIGHT_ROWS: Array<{ label: string; weights: NodeWeights }> = [
  { label: "Maçã",   weights: APPLE_WEIGHTS  },
  { label: "Laranja", weights: ORANGE_WEIGHTS },
];

// ─── Math helpers ─────────────────────────────────────────────────────────────
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function computeOutput(inputs: Inputs, weights: NodeWeights): number {
  const raw =
    inputs.yellow * weights.yellow +
    inputs.red    * weights.red    +
    inputs.rough  * weights.rough;
  return Math.max(0, Math.min(1, sigmoid(raw - 0.3)));
}

function toHex2(n: number): string {
  return Math.round(Math.max(0, Math.min(255, n)))
    .toString(16)
    .padStart(2, "0");
}

// ─── SVG Connections ──────────────────────────────────────────────────────────
function SvgConnections({ disp }: { disp: DisplayState }) {
  return (
    <svg
      viewBox="0 0 600 400"
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
    >
      {CONNECTIONS.map((c, i) => {
        const f      = SVG_PTS[c.from];
        const t      = SVG_PTS[c.to];
        const signal = c.weight * disp.inputs[c.inputKey];
        const abs    = Math.abs(signal);
        const alpha  = Math.round(lerp(30, 220, abs) );
        const width  = lerp(0.5, 5, abs);
        const color  = c.weight > 0 ? `#4aaeff${toHex2(alpha)}` : `#ff5544${toHex2(alpha)}`;
        return (
          <line key={i} x1={f.x} y1={f.y} x2={t.x} y2={t.y}
            stroke={color} strokeWidth={width} strokeLinecap="round" />
        );
      })}
    </svg>
  );
}

// ─── Slider ───────────────────────────────────────────────────────────────────
function Slider({ label, value, onChange, trackColor }: SliderProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
      <div style={{
        width: 64, height: 64, borderRadius: "50%",
        background: `conic-gradient(${trackColor} ${value * 360}deg, #2a2a2a ${value * 360}deg)`,
        display: "flex", alignItems: "center", justifyContent: "center",
        boxShadow: `0 0 ${14 * value}px ${trackColor}88`,
        transition: "box-shadow 0.2s",
      }}>
        <div style={{
          width: 48, height: 48, borderRadius: "50%",
          background: "#111",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 13, fontWeight: 700, color: trackColor,
          fontFamily: "'Space Mono', monospace",
        }}>
          {value.toFixed(2)}
        </div>
      </div>
      <span style={{ fontSize: 10, color: "#888", letterSpacing: 1 }}>{label.toUpperCase()}</span>
      <input
        type="range" min={0} max={1} step={0.01} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: 80, accentColor: trackColor, cursor: "pointer" }}
      />
    </div>
  );
}

// ─── Output Node ──────────────────────────────────────────────────────────────
function OutputNode({ label, value, glowColor, emoji }: OutputNodeProps) {
  const active = value > 0.4;
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
      <div style={{
        width: 80, height: 80, borderRadius: "50%",
        background: active
          ? `radial-gradient(circle at 35% 30%, #fff8, ${glowColor}cc, ${glowColor}44)`
          : "radial-gradient(circle, #2a2a2a, #111)",
        border: `3px solid ${active ? glowColor : "#333"}`,
        boxShadow: active ? `0 0 ${Math.round(value * 40)}px ${glowColor}99` : "none",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: active ? 32 : 20,
        transition: "all 0.35s cubic-bezier(.34,1.56,.64,1)",
        transform: active ? `scale(${1 + value * 0.12})` : "scale(1)",
      }}>
        <span style={{ filter: active ? "none" : "grayscale(1) opacity(0.3)", transition: "filter 0.3s" }}>
          {emoji}
        </span>
      </div>
      <div style={{
        fontSize: 11, fontFamily: "'Space Mono', monospace", letterSpacing: 1,
        color: active ? glowColor : "#444", transition: "color 0.3s",
      }}>
        {label.toUpperCase()} — {(value * 100).toFixed(0)}%
      </div>
    </div>
  );
}

// ─── Weight Table ─────────────────────────────────────────────────────────────
function WeightTable() {
  const keys: Array<keyof NodeWeights> = ["yellow", "red", "rough"];
  const keyLabels: Record<keyof NodeWeights, string> = { yellow: "Amar.", red: "Verm.", rough: "Rugos." };
  return (
    <div style={{
      background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)",
      borderRadius: 10, padding: "10px 14px", fontSize: 10,
      fontFamily: "'Space Mono', monospace", color: "#555", lineHeight: 2,
      minWidth: 170,
    }}>
      {WEIGHT_ROWS.map(row => (
        <React.Fragment key={row.label}>
          <div style={{ color: "#666", letterSpacing: 1, borderBottom: "1px solid #222", paddingBottom: 2, marginBottom: 2 }}>
            PESOS → {row.label.toUpperCase()}
          </div>
          {keys.map(k => (
            <div key={k} style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
              <span style={{ color: "#555" }}>{keyLabels[k]}</span>
              <span style={{
                color: row.weights[k] > 0 ? "#4aaeff" : "#ff5544",
                fontWeight: 700,
              }}>
                {row.weights[k] > 0 ? "+" : ""}{row.weights[k].toFixed(1)}
              </span>
            </div>
          ))}
          <div style={{ marginBottom: 8 }} />
        </React.Fragment>
      ))}
    </div>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [inputs, setInputs] = useState<Inputs>({ yellow: 0, red: 0, rough: 0 });
  const [outputs, setOutputs] = useState<OutputResult>({ apple: 0, orange: 0 });

  // Smooth display state animated via rAF
  const dispRef = useRef<DisplayState>({ inputs: { ...inputs }, outputs: { ...outputs } });
  const rafRef  = useRef<number | null>(null);

  useEffect(() => {
    const target = {
      inputs:  { ...inputs },
      outputs: {
        apple:  computeOutput(inputs, APPLE_WEIGHTS),
        orange: computeOutput(inputs, ORANGE_WEIGHTS),
      },
    };

    const animate = () => {
      const d = dispRef.current;
      const speed = 0.12;
      (Object.keys(d.inputs) as Array<keyof Inputs>).forEach(k => {
        d.inputs[k] = lerp(d.inputs[k], target.inputs[k], speed);
      });
      (Object.keys(d.outputs) as Array<keyof OutputResult>).forEach(k => {
        d.outputs[k] = lerp(d.outputs[k], target.outputs[k], speed);
      });
      setOutputs({ ...d.outputs });
      rafRef.current = requestAnimationFrame(animate);
    };

    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(animate);
    return () => { if (rafRef.current !== null) cancelAnimationFrame(rafRef.current); };
  }, [inputs]);

  const setInput = useCallback((key: keyof Inputs, val: number) => {
    setInputs(prev => ({ ...prev, [key]: val }));
  }, []);

  const winner = outputs.apple > outputs.orange ? "apple" : "orange";
  const confidence = Math.max(outputs.apple, outputs.orange);

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #0c0c0c 0%, #161628 60%, #0c0c0c 100%)",
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", padding: "28px 12px",
      fontFamily: "'Space Mono', monospace",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        input[type=range] { height: 4px; border-radius: 2px; }
      `}</style>

      <h1 style={{
        fontSize: "clamp(16px, 3.5vw, 24px)", color: "#fff",
        letterSpacing: 3, marginBottom: 4, fontWeight: 700,
        textTransform: "uppercase",
        textShadow: "0 0 30px rgba(80,160,255,0.5)",
      }}>
        🧠 Rede Neural — Classificador de Frutas
      </h1>
      <p style={{ color: "#444", fontSize: 10, letterSpacing: 2, marginBottom: 28 }}>
        AJUSTE OS NEURÔNIOS E VEJA A REDE APRENDER
      </p>

      {/* Main card */}
      <div style={{
        position: "relative", width: "min(760px, 98vw)",
        background: "rgba(255,255,255,0.025)",
        borderRadius: 20, border: "1px solid rgba(255,255,255,0.07)",
        boxShadow: "0 24px 80px rgba(0,0,0,0.6)",
        padding: "32px 20px", overflow: "hidden",
      }}>
        {/* SVG connections layer */}
        <SvgConnections disp={{ inputs: dispRef.current.inputs, outputs }} />

        {/* Three-column layout */}
        <div style={{
          position: "relative", zIndex: 2,
          display: "flex", alignItems: "center",
          justifyContent: "space-between", gap: 12,
        }}>
          {/* INPUT LAYER */}
          <div style={{ display: "flex", flexDirection: "column", gap: 24, alignItems: "center" }}>
            <span style={{ fontSize: 9, color: "#333", letterSpacing: 2 }}>ENTRADA</span>
            <Slider label="Amarelo"   value={inputs.yellow} onChange={(v: number) => setInput("yellow", v)} trackColor="#f5d742" />
            <Slider label="Vermelho"  value={inputs.red}    onChange={(v: number) => setInput("red",    v)} trackColor="#ff4433" />
            <Slider label="Rugosidade" value={inputs.rough} onChange={(v: number) => setInput("rough",  v)} trackColor="#88bbcc" />
          </div>

          {/* CENTER — weight table */}
          <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
            <WeightTable />
          </div>

          {/* OUTPUT LAYER */}
          <div style={{ display: "flex", flexDirection: "column", gap: 36, alignItems: "center" }}>
            <span style={{ fontSize: 9, color: "#333", letterSpacing: 2 }}>SAÍDA</span>
            <OutputNode label="Maçã"   value={outputs.apple}  glowColor="#dd3333" emoji="🍎" />
            <OutputNode label="Laranja" value={outputs.orange} glowColor="#ff9900" emoji="🍊" />
          </div>
        </div>
      </div>

      {/* Prediction banner */}
      <div style={{
        marginTop: 22, padding: "12px 28px", borderRadius: 50,
        background: confidence > 0.5
          ? winner === "apple"
            ? "rgba(220,50,50,0.15)"
            : "rgba(255,150,0,0.15)"
          : "rgba(255,255,255,0.03)",
        border: `1px solid ${confidence > 0.5 ? (winner === "apple" ? "#dd333366" : "#ff990066") : "rgba(255,255,255,0.07)"}`,
        display: "flex", alignItems: "center", gap: 12,
        transition: "all 0.4s ease",
      }}>
        <span style={{ fontSize: 20 }}>
          {confidence < 0.35 ? "🤔" : winner === "apple" ? "🍎" : "🍊"}
        </span>
        <span style={{
          fontSize: 11, letterSpacing: 2, fontWeight: 700,
          color: confidence < 0.35 ? "#444"
            : winner === "apple" ? "#ff8888" : "#ffbb44",
        }}>
          {confidence < 0.3
            ? "AJUSTE OS NEURÔNIOS..."
            : confidence < 0.5
            ? "INCERTO..."
            : `PROVAVELMENTE ${winner === "apple" ? "MAÇÃ" : "LARANJA"} — ${(confidence * 100).toFixed(0)}%`}
        </span>
      </div>

      {/* Legend */}
      <div style={{ marginTop: 16, display: "flex", gap: 20, alignItems: "center" }}>
        {[["#4aaeff", "PESO POSITIVO"], ["#ff5544", "PESO NEGATIVO"]].map(([color, label]) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 24, height: 3, background: color, borderRadius: 2 }} />
            <span style={{ fontSize: 9, color: "#444", letterSpacing: 1 }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}