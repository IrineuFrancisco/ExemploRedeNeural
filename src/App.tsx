import { useState, useEffect, useRef, useCallback } from "react";

// ════════════════════════════════════════════════════════════
//  TYPES
// ════════════════════════════════════════════════════════════
interface Inputs { yellow: number; red: number; rough: number; }
type InputKey = keyof Inputs;

interface HiddenNode { id: string; bias: number; weights: number[]; }
interface FruitOutputNode { id: string; label: string; emoji: string; glow: string;
                            bias: number; weights: number[]; }

interface TrainSample { inputs: Inputs; label: string; }

// ════════════════════════════════════════════════════════════
//  CONSTANTS
// ════════════════════════════════════════════════════════════
const INPUT_META = [
  { key: "yellow" as InputKey, label: "Amarelo",    color: "#f5d742" },
  { key: "red"    as InputKey, label: "Vermelho",   color: "#ff4433" },
  { key: "rough"  as InputKey, label: "Rugosidade", color: "#88bbcc" },
];

const FRUIT_META = [
  { id: "apple",      label: "Maçã",    emoji: "🍎", glow: "#dd3333" },
  { id: "orange",     label: "Laranja", emoji: "🍊", glow: "#ff9900" },
  { id: "banana",     label: "Banana",  emoji: "🍌", glow: "#ffe44d" },
  { id: "strawberry", label: "Morango", emoji: "🍓", glow: "#ff4488" },
];

const PRESETS = [
  { label: "Maçã típica",    emoji: "🍎", inputs: { yellow: 0.1, red: 0.9, rough: 0.1 } },
  { label: "Laranja típica", emoji: "🍊", inputs: { yellow: 0.9, red: 0.1, rough: 0.8 } },
  { label: "Banana típica",  emoji: "🍌", inputs: { yellow: 1.0, red: 0.0, rough: 0.2 } },
  { label: "Morango típico", emoji: "🍓", inputs: { yellow: 0.0, red: 0.8, rough: 0.3 } },
];

const TRAIN_DATA: TrainSample[] = [
  { inputs: { yellow: 0.1, red: 0.9, rough: 0.1 }, label: "apple"      },
  { inputs: { yellow: 0.2, red: 0.8, rough: 0.2 }, label: "apple"      },
  { inputs: { yellow: 0.9, red: 0.1, rough: 0.8 }, label: "orange"     },
  { inputs: { yellow: 0.8, red: 0.2, rough: 0.9 }, label: "orange"     },
  { inputs: { yellow: 1.0, red: 0.0, rough: 0.2 }, label: "banana"     },
  { inputs: { yellow: 0.9, red: 0.0, rough: 0.1 }, label: "banana"     },
  { inputs: { yellow: 0.0, red: 0.8, rough: 0.3 }, label: "strawberry" },
  { inputs: { yellow: 0.1, red: 0.9, rough: 0.4 }, label: "strawberry" },
];

// SVG layout — 5 hidden nodes
const SVG_W = 700, SVG_H = 460;
const INPUT_X = 80, HIDDEN_X = 290, OUTPUT_X = 560;
const INPUT_YS  = [70, 210, 350];
const HIDDEN_YS = [46, 138, 230, 322, 414];
const OUTPUT_YS = [70, 185, 300, 400];

// ════════════════════════════════════════════════════════════
//  MATH
// ════════════════════════════════════════════════════════════
function sigmoid(x: number): number { return 1 / (1 + Math.exp(-x)); }
function sigmoidDeriv(x: number): number { const s = sigmoid(x); return s * (1 - s); }
function lerp(a: number, b: number, t: number): number { return a + (b - a) * t; }
function toHex2(n: number): string {
  return Math.round(Math.max(0, Math.min(255, n))).toString(16).padStart(2, "0");
}
function softmax(arr: number[]): number[] {
  const mx = Math.max(...arr);
  const ex = arr.map(v => Math.exp(v - mx));
  const s  = ex.reduce((a, b) => a + b, 0);
  return ex.map(v => v / s);
}

// ════════════════════════════════════════════════════════════
//  PRE-TRAINED WEIGHTS (100% accuracy on all presets)
//  Inputs: [yellow, red, rough]
//  h0: yellow detector        → Laranja / Banana
//  h1: red detector           → Maçã / Morango
//  h2: rough detector         → Laranja / Morango
//  h3: smooth detector        → Maçã / Banana
//  h4: red+rough combined     → distingue Morango de Maçã
// ════════════════════════════════════════════════════════════
const PRETRAINED_HIDDEN: Array<{ bias: number; weights: [number, number, number] }> = [
  { bias: -1.5, weights: [ 3.5, -0.5, -0.5] },  // h0: yellow
  { bias: -1.5, weights: [-0.5,  3.5, -0.5] },  // h1: red
  { bias: -1.5, weights: [-0.5, -0.5,  3.5] },  // h2: rough
  { bias:  2.2, weights: [-0.5, -0.5, -5.0] },  // h3: smooth
  { bias: -4.5, weights: [-0.5,  4.0,  4.0] },  // h4: red+rough
];

const PRETRAINED_OUTPUT: Array<{ bias: number; weights: [number, number, number, number, number] }> = [
  { bias: -1.0, weights: [-1.5,  2.5, -2.5,  4.5, -5.0] },  // apple:      red + smooth
  { bias: -1.0, weights: [ 4.0, -1.5,  3.0, -2.0, -3.5] },  // orange:     yellow + rough
  { bias: -1.0, weights: [ 3.0, -1.5, -2.0,  3.0, -2.5] },  // banana:     yellow + smooth
  { bias: -1.0, weights: [-2.0,  1.5, -1.5, -3.5,  6.5] },  // strawberry: red + rough
];

// ════════════════════════════════════════════════════════════
//  NETWORK
// ════════════════════════════════════════════════════════════
function initNetwork() {
  const hidden: HiddenNode[] = PRETRAINED_HIDDEN.map((p, i) => ({
    id: `h${i}`, bias: p.bias, weights: [...p.weights],
  }));
  const outputs: FruitOutputNode[] = FRUIT_META.map((f, i) => ({
    ...f,
    bias: PRETRAINED_OUTPUT[i].bias,
    weights: [...PRETRAINED_OUTPUT[i].weights],
  }));
  return { hidden, outputs };
}

function forward(inp: number[], hidden: HiddenNode[], outputs: FruitOutputNode[]) {
  const hiddenAct = hidden.map(h =>
    sigmoid(h.weights.reduce((s, w, i) => s + w * inp[i], 0) + h.bias)
  );
  const outputRaw = outputs.map(o =>
    o.weights.reduce((s, w, i) => s + w * hiddenAct[i], 0) + o.bias
  );
  return { hiddenAct, outputAct: softmax(outputRaw) };
}

function trainStep(
  inp: number[], targetIdx: number,
  hidden: HiddenNode[], outputs: FruitOutputNode[], lr: number
) {
  const { hiddenAct, outputAct } = forward(inp, hidden, outputs);
  const loss  = -Math.log(Math.max(outputAct[targetIdx], 1e-9));
  const dOut  = outputAct.map((a, i) => a - (i === targetIdx ? 1 : 0));

  const newOutputs: FruitOutputNode[] = outputs.map((o, oi) => ({
    ...o,
    bias: o.bias - lr * dOut[oi],
    weights: o.weights.map((w, hi) => w - lr * dOut[oi] * hiddenAct[hi]),
  }));

  const dHidden = hidden.map((h, hi) => {
    const err = outputs.reduce((s, o, oi) => s + dOut[oi] * o.weights[hi], 0);
    const raw = h.weights.reduce((s, w, i) => s + w * inp[i], 0) + h.bias;
    return err * sigmoidDeriv(raw);
  });

  const newHidden: HiddenNode[] = hidden.map((h, hi) => ({
    ...h,
    bias: h.bias - lr * dHidden[hi],
    weights: h.weights.map((w, ii) => w - lr * dHidden[hi] * inp[ii]),
  }));

  return { hidden: newHidden, outputs: newOutputs, loss };
}

// ════════════════════════════════════════════════════════════
//  SVG NETWORK CANVAS
// ════════════════════════════════════════════════════════════
interface NetCanvasProps {
  hiddenNodes: HiddenNode[];
  outputNodes: FruitOutputNode[];
  inputVals: number[];
  hiddenActs: number[];
  outputActs: number[];
}

function NetCanvas({ hiddenNodes, outputNodes, inputVals, hiddenActs, outputActs }: NetCanvasProps) {
  const ihLines = INPUT_YS.flatMap((iy, ii) =>
    HIDDEN_YS.map((hy, hi) => {
      const w   = hiddenNodes[hi]?.weights[ii] ?? 0;
      const sig = Math.abs(w * inputVals[ii]);
      const alpha = toHex2(lerp(15, 190, Math.min(sig, 1)));
      const col = w >= 0 ? `#4aaeff${alpha}` : `#ff5544${alpha}`;
      return <line key={`ih-${ii}-${hi}`}
        x1={INPUT_X} y1={iy} x2={HIDDEN_X} y2={hy}
        stroke={col} strokeWidth={lerp(0.4, 4.5, Math.min(sig, 1))} strokeLinecap="round" />;
    })
  );

  const hoLines = HIDDEN_YS.flatMap((hy, hi) =>
    OUTPUT_YS.map((oy, oi) => {
      const w   = outputNodes[oi]?.weights[hi] ?? 0;
      const sig = Math.abs(w * hiddenActs[hi]);
      const alpha = toHex2(lerp(15, 190, Math.min(sig, 1)));
      const col = w >= 0 ? `#4aaeff${alpha}` : `#ff5544${alpha}`;
      return <line key={`ho-${hi}-${oi}`}
        x1={HIDDEN_X} y1={hy} x2={OUTPUT_X} y2={oy}
        stroke={col} strokeWidth={lerp(0.4, 4.5, Math.min(sig, 1))} strokeLinecap="round" />;
    })
  );

  const inputColors = ["245,215,66", "255,68,51", "136,187,204"];

  return (
    <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
      {ihLines}{hoLines}

      {/* Input nodes */}
      {INPUT_YS.map((y, i) => (
        <g key={`in-${i}`}>
          <circle cx={INPUT_X} cy={y} r={24}
            fill={`rgba(${inputColors[i]},${0.12 + inputVals[i] * 0.5})`}
            stroke={INPUT_META[i].color} strokeWidth={2} />
          <text x={INPUT_X} y={y + 5} textAnchor="middle" fontSize={10}
            fill={INPUT_META[i].color} fontFamily="'Space Mono',monospace" fontWeight={700}>
            {inputVals[i].toFixed(2)}
          </text>
        </g>
      ))}

      {/* Hidden nodes */}
      {HIDDEN_YS.map((y, i) => (
        <g key={`h-${i}`}>
          <circle cx={HIDDEN_X} cy={y} r={22}
            fill={`rgba(120,200,255,${0.05 + (hiddenActs[i] ?? 0) * 0.4})`}
            stroke={`rgba(120,200,255,${0.25 + (hiddenActs[i] ?? 0) * 0.65})`} strokeWidth={2} />
          <text x={HIDDEN_X} y={y + 5} textAnchor="middle" fontSize={9}
            fill={`rgba(140,210,255,${0.4 + (hiddenActs[i] ?? 0) * 0.6})`}
            fontFamily="'Space Mono',monospace" fontWeight={700}>
            {(hiddenActs[i] ?? 0).toFixed(2)}
          </text>
        </g>
      ))}

      {/* Output nodes */}
      {OUTPUT_YS.map((y, i) => {
        const f = FRUIT_META[i];
        const a = outputActs[i] ?? 0;
        return (
          <g key={`o-${i}`}>
            <circle cx={OUTPUT_X} cy={y} r={26}
              fill={`${f.glow}${toHex2(18 + a * 110)}`}
              stroke={f.glow} strokeWidth={a > 0.3 ? 2.5 : 1}
              style={{ filter: a > 0.35 ? `drop-shadow(0 0 ${(a * 14).toFixed(0)}px ${f.glow})` : "none" }} />
            <text x={OUTPUT_X} y={y + 9} textAnchor="middle" fontSize={a > 0.2 ? 20 : 14}
              style={{ transition: "font-size 0.3s" }}>
              {a > 0.18 ? f.emoji : "○"}
            </text>
          </g>
        );
      })}

      {/* Layer labels */}
      {([[INPUT_X, "ENTRADA"], [HIDDEN_X, "OCULTA"], [OUTPUT_X, "SAÍDA"]] as [number, string][]).map(([x, t]) => (
        <text key={t} x={x} y={SVG_H - 8} textAnchor="middle"
          fontSize={9} fill="#2e2e2e" fontFamily="'Space Mono',monospace" letterSpacing={2}>{t}</text>
      ))}
    </svg>
  );
}

// ════════════════════════════════════════════════════════════
//  SLIDER (circular)
// ════════════════════════════════════════════════════════════
function Slider({ label, value, onChange, color }: {
  label: string; value: number; onChange: (v: number) => void; color: string;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 5 }}>
      <div style={{
        width: 56, height: 56, borderRadius: "50%",
        background: `conic-gradient(${color} ${value * 360}deg, #181818 ${value * 360}deg)`,
        display: "flex", alignItems: "center", justifyContent: "center",
        boxShadow: `0 0 ${12 * value}px ${color}55`,
        transition: "box-shadow 0.2s",
      }}>
        <div style={{
          width: 42, height: 42, borderRadius: "50%", background: "#0c0c0c",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 11, fontWeight: 700, color,
          fontFamily: "'Space Mono',monospace",
        }}>{value.toFixed(2)}</div>
      </div>
      <span style={{ fontSize: 9, color: "#555", letterSpacing: 1 }}>{label.toUpperCase()}</span>
      <input type="range" min={0} max={1} step={0.01} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: 70, accentColor: color, cursor: "pointer" }} />
    </div>
  );
}

// ════════════════════════════════════════════════════════════
//  LOSS SPARKLINE
// ════════════════════════════════════════════════════════════
function LossChart({ history }: { history: number[] }) {
  if (history.length < 2) return (
    <div style={{ color: "#333", fontSize: 10, fontFamily: "'Space Mono',monospace" }}>
      Aguardando dados de treino...
    </div>
  );
  const max = Math.max(...history, 0.01);
  const pts = history.map((v, i) =>
    `${(i / (history.length - 1)) * 260},${50 - (v / max) * 46}`
  ).join(" ");
  return (
    <svg width={265} height={58} style={{ overflow: "visible" }}>
      <polyline points={pts} fill="none" stroke="#4aaeff" strokeWidth={1.8} strokeLinejoin="round" />
      <line x1={0} y1={50} x2={260} y2={50} stroke="#1a1a1a" strokeWidth={1} />
      <text x={0} y={58} fontSize={9} fill="#444" fontFamily="'Space Mono',monospace">
        loss atual: {history[history.length - 1].toFixed(4)}
      </text>
    </svg>
  );
}

// ════════════════════════════════════════════════════════════
//  APP
// ════════════════════════════════════════════════════════════
export default function App() {
  const [inputs, setInputs]         = useState<Inputs>({ yellow: 0.5, red: 0.5, rough: 0.5 });
  const [net, setNet]               = useState(() => initNetwork());
  const [hiddenActs, setHiddenActs] = useState<number[]>([0, 0, 0, 0, 0]);
  const [outputActs, setOutputActs] = useState<number[]>([0.25, 0.25, 0.25, 0.25]);
  const [training, setTraining]     = useState(false);
  const [epoch, setEpoch]           = useState(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [accuracy, setAccuracy]     = useState(0);
  const [tab, setTab]               = useState<"network" | "train">("network");

  const trainRef = useRef(false);
  const netRef   = useRef(net);
  netRef.current = net;

  useEffect(() => {
    const inp = [inputs.yellow, inputs.red, inputs.rough];
    const { hiddenAct, outputAct } = forward(inp, net.hidden, net.outputs);
    setHiddenActs(hiddenAct);
    setOutputActs(outputAct);
  }, [inputs, net]);

  useEffect(() => { trainRef.current = training; }, [training]);

  const startTraining = useCallback(() => {
    setTraining(true);
    let localNet = { hidden: [...netRef.current.hidden], outputs: [...netRef.current.outputs] };
    const LR = 0.08;

    const step = () => {
      if (!trainRef.current) return;
      const shuffled = [...TRAIN_DATA].sort(() => Math.random() - 0.5);
      let totalLoss = 0;
      shuffled.forEach(sample => {
        const inp = [sample.inputs.yellow, sample.inputs.red, sample.inputs.rough];
        const ti  = FRUIT_META.findIndex(f => f.id === sample.label);
        const res = trainStep(inp, ti, localNet.hidden, localNet.outputs, LR);
        localNet  = { hidden: res.hidden, outputs: res.outputs };
        totalLoss += res.loss;
      });

      let correct = 0;
      TRAIN_DATA.forEach(s => {
        const inp = [s.inputs.yellow, s.inputs.red, s.inputs.rough];
        const { outputAct } = forward(inp, localNet.hidden, localNet.outputs);
        if (FRUIT_META[outputAct.indexOf(Math.max(...outputAct))].id === s.label) correct++;
      });

      setNet({ hidden: [...localNet.hidden], outputs: [...localNet.outputs] });
      setEpoch(e => e + 1);
      setLossHistory(prev => [...prev.slice(-80), totalLoss / TRAIN_DATA.length]);
      setAccuracy(correct / TRAIN_DATA.length);

      if (trainRef.current) setTimeout(step, 60);
    };
    step();
  }, []);

  const stopTraining = useCallback(() => setTraining(false), []);

  const resetNet = useCallback(() => {
    setTraining(false);
    setNet(initNetwork());
    setEpoch(0);
    setLossHistory([]);
    setAccuracy(0);
  }, []);

  const setInput = useCallback((k: InputKey, v: number) =>
    setInputs(prev => ({ ...prev, [k]: v })), []);

  const winnerIdx  = outputActs.indexOf(Math.max(...outputActs));
  const winner     = FRUIT_META[winnerIdx];
  const confidence = outputActs[winnerIdx] ?? 0;

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(160deg,#070710 0%,#0f0f20 55%,#070710 100%)",
      display: "flex", flexDirection: "column", alignItems: "center",
      padding: "24px 10px 48px", fontFamily: "'Space Mono',monospace",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        input[type=range] { height: 4px; border-radius: 2px; }
        button { cursor: pointer; border: none; outline: none; font-family: 'Space Mono',monospace; }
      `}</style>

      {/* HEADER */}
      <h1 style={{
        fontSize: "clamp(14px,3vw,21px)", color: "#fff", letterSpacing: 4,
        marginBottom: 4, fontWeight: 700, textTransform: "uppercase",
        textShadow: "0 0 40px rgba(80,160,255,0.55)",
      }}>🧠 Rede Neural — Classificador de Frutas</h1>
      <p style={{ color: "#2d2d2d", fontSize: 9, letterSpacing: 3, marginBottom: 18 }}>
        CAMADA OCULTA · 4 FRUTAS · BACKPROPAGATION · EXEMPLOS PRÉ-DEFINIDOS
      </p>

      {/* TABS */}
      <div style={{ display: "flex", gap: 4, marginBottom: 18 }}>
        {(["network", "train"] as const).map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: "7px 22px", borderRadius: 6, fontSize: 10, letterSpacing: 2, fontWeight: 700,
            background: tab === t ? "rgba(74,174,255,0.18)" : "rgba(255,255,255,0.03)",
            color: tab === t ? "#4aaeff" : "#3a3a3a",
            border: `1px solid ${tab === t ? "#4aaeff44" : "rgba(255,255,255,0.06)"}`,
            transition: "all 0.2s",
          }}>{t === "network" ? "🔬 REDE" : "⚡ TREINAR"}</button>
        ))}
      </div>

      {/* ── TAB: NETWORK ── */}
      {tab === "network" && (<>
        <div style={{
          position: "relative", width: "min(820px,98vw)",
          background: "rgba(255,255,255,0.018)",
          borderRadius: 20, border: "1px solid rgba(255,255,255,0.055)",
          boxShadow: "0 24px 80px rgba(0,0,0,0.7)",
          overflow: "hidden", minHeight: 480,
        }}>
          <NetCanvas
            hiddenNodes={net.hidden} outputNodes={net.outputs}
            inputVals={[inputs.yellow, inputs.red, inputs.rough]}
            hiddenActs={hiddenActs} outputActs={outputActs} />

          {/* Input sliders */}
          <div style={{
            position: "absolute", left: 0, top: 0, bottom: 0, width: 108,
            display: "flex", flexDirection: "column",
            justifyContent: "space-around", alignItems: "center",
            paddingLeft: 6, paddingBottom: 22, zIndex: 10,
          }}>
            {INPUT_META.map(m => (
              <Slider key={m.key} label={m.label} value={inputs[m.key]}
                onChange={v => setInput(m.key, v)} color={m.color} />
            ))}
          </div>

          {/* Output labels */}
          <div style={{
            position: "absolute", right: 0, top: 0, bottom: 0, width: 108,
            display: "flex", flexDirection: "column",
            justifyContent: "space-around", alignItems: "center",
            paddingRight: 6, paddingBottom: 22, zIndex: 10,
          }}>
            {FRUIT_META.map((f, i) => (
              <div key={f.id} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
                <span style={{
                  fontSize: 9, letterSpacing: 1, fontWeight: 700,
                  color: outputActs[i] > 0.28 ? f.glow : "#2e2e2e",
                  transition: "color 0.3s",
                }}>{f.label.toUpperCase()}</span>
                <span style={{
                  fontSize: 10,
                  color: outputActs[i] > 0.28 ? "#bbb" : "#2a2a2a",
                  transition: "color 0.3s",
                }}>{(outputActs[i] * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* PRESETS */}
        <div style={{ marginTop: 14, display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center", alignItems: "center" }}>
          <span style={{ fontSize: 9, color: "#2d2d2d", letterSpacing: 2 }}>EXEMPLOS:</span>
          {PRESETS.map(p => (
            <button key={p.label} onClick={() => setInputs({ ...p.inputs })} style={{
              padding: "6px 14px", borderRadius: 20, fontSize: 10,
              background: "rgba(255,255,255,0.04)", color: "#666",
              border: "1px solid rgba(255,255,255,0.07)", transition: "all 0.2s",
            }}
              onMouseEnter={e => (e.currentTarget.style.background = "rgba(255,255,255,0.09)")}
              onMouseLeave={e => (e.currentTarget.style.background = "rgba(255,255,255,0.04)")}
            >{p.emoji} {p.label}</button>
          ))}
        </div>

        {/* PREDICTION BANNER */}
        <div style={{
          marginTop: 16, padding: "11px 30px", borderRadius: 50,
          background: confidence > 0.35 ? `${winner.glow}1e` : "rgba(255,255,255,0.02)",
          border: `1px solid ${confidence > 0.35 ? `${winner.glow}44` : "rgba(255,255,255,0.06)"}`,
          display: "flex", alignItems: "center", gap: 12, transition: "all 0.4s ease",
        }}>
          <span style={{ fontSize: 22 }}>{confidence > 0.32 ? winner.emoji : "🤔"}</span>
          <span style={{
            fontSize: 11, letterSpacing: 2, fontWeight: 700,
            color: confidence > 0.32 ? winner.glow : "#2e2e2e",
          }}>
            {confidence < 0.3
              ? "INCERTO — AJUSTE OS NEURÔNIOS"
              : `PROVAVELMENTE ${winner.label.toUpperCase()} — ${(confidence * 100).toFixed(0)}%`}
          </span>
        </div>

        {/* OUTPUT BAR CHART */}
        <div style={{ marginTop: 14, display: "flex", gap: 10, alignItems: "flex-end" }}>
          {FRUIT_META.map((f, i) => (
            <div key={f.id} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
              <div style={{
                width: 34,
                height: Math.max(4, Math.round(outputActs[i] * 80)),
                background: `${f.glow}${toHex2(70 + outputActs[i] * 170)}`,
                borderRadius: "4px 4px 0 0",
                boxShadow: outputActs[i] > 0.3 ? `0 0 10px ${f.glow}77` : "none",
                transition: "all 0.35s cubic-bezier(.34,1.56,.64,1)",
              }} />
              <span style={{ fontSize: 16 }}>{f.emoji}</span>
            </div>
          ))}
        </div>

        {epoch > 0 && (
          <div style={{ marginTop: 10, fontSize: 9, color: "#333", letterSpacing: 2 }}>
            REDE TREINADA — {epoch} ÉPOCAS · ACURÁCIA {(accuracy * 100).toFixed(0)}%
            {accuracy >= 1 && <span style={{ color: "#44ff88", marginLeft: 8 }}>✓ PERFEITO</span>}
          </div>
        )}
      </>)}

      {/* ── TAB: TRAIN ── */}
      {tab === "train" && (
        <div style={{
          width: "min(820px,98vw)",
          background: "rgba(255,255,255,0.018)",
          borderRadius: 20, border: "1px solid rgba(255,255,255,0.055)",
          padding: "28px", display: "flex", flexDirection: "column", gap: 22,
        }}>
          {/* Stats */}
          <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
            {([
              ["ÉPOCAS",   String(epoch)],
              ["ACURÁCIA", `${(accuracy * 100).toFixed(0)}%`],
              ["AMOSTRAS", String(TRAIN_DATA.length)],
              ["STATUS",   training ? "⚡ TREINANDO" : epoch > 0 ? "⏸ PAUSADO" : "AGUARDANDO"],
            ] as [string, string][]).map(([k, v]) => (
              <div key={k} style={{
                background: "rgba(255,255,255,0.03)", borderRadius: 10,
                padding: "10px 18px", border: "1px solid rgba(255,255,255,0.055)", flex: "1 1 auto",
              }}>
                <div style={{ fontSize: 8, color: "#333", letterSpacing: 2, marginBottom: 4 }}>{k}</div>
                <div style={{
                  fontSize: 20, fontWeight: 700,
                  color: k === "ACURÁCIA" && accuracy >= 1 ? "#44ff88"
                    : k === "STATUS" && training ? "#4aaeff" : "#aaa",
                }}>{v}</div>
              </div>
            ))}
          </div>

          {/* Loss chart */}
          <div style={{
            background: "rgba(0,0,0,0.28)", borderRadius: 10,
            padding: "14px 18px", border: "1px solid rgba(255,255,255,0.04)",
          }}>
            <div style={{ fontSize: 9, color: "#2e2e2e", letterSpacing: 2, marginBottom: 12 }}>CURVA DE LOSS</div>
            <LossChart history={lossHistory} />
          </div>

          {/* Dataset */}
          <div>
            <div style={{ fontSize: 9, color: "#2e2e2e", letterSpacing: 2, marginBottom: 10 }}>DATASET DE TREINO</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {TRAIN_DATA.map((s, i) => {
                const f = FRUIT_META.find(fr => fr.id === s.label)!;
                const inp = [s.inputs.yellow, s.inputs.red, s.inputs.rough];
                const { outputAct } = forward(inp, net.hidden, net.outputs);
                const correct = FRUIT_META[outputAct.indexOf(Math.max(...outputAct))].id === s.label;
                return (
                  <div key={i} style={{
                    padding: "5px 11px", borderRadius: 7, fontSize: 9,
                    background: correct ? "rgba(68,255,136,0.06)" : "rgba(255,68,68,0.06)",
                    border: `1px solid ${correct ? "#44ff8826" : "#ff444426"}`,
                    color: correct ? "#44ff88" : "#ff6666",
                  }}>
                    {f.emoji} {correct ? "✓" : "✗"} A:{s.inputs.yellow.toFixed(1)} V:{s.inputs.red.toFixed(1)} R:{s.inputs.rough.toFixed(1)}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Controls */}
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            {!training
              ? <button onClick={startTraining} style={{
                  padding: "10px 22px", borderRadius: 8, fontSize: 11, fontWeight: 700,
                  letterSpacing: 2, background: "rgba(74,174,255,0.14)", color: "#4aaeff",
                  border: "1px solid #4aaeff44",
                }}>⚡ INICIAR TREINO</button>
              : <button onClick={stopTraining} style={{
                  padding: "10px 22px", borderRadius: 8, fontSize: 11, fontWeight: 700,
                  letterSpacing: 2, background: "rgba(255,200,0,0.1)", color: "#ffcc00",
                  border: "1px solid #ffcc0044",
                }}>⏸ PAUSAR</button>
            }
            <button onClick={resetNet} style={{
              padding: "10px 22px", borderRadius: 8, fontSize: 11, fontWeight: 700,
              letterSpacing: 2, background: "rgba(255,80,80,0.09)", color: "#ff5555",
              border: "1px solid #ff555533",
            }}>↺ RESETAR REDE</button>
            <button onClick={() => setTab("network")} style={{
              padding: "10px 22px", borderRadius: 8, fontSize: 11, fontWeight: 700,
              letterSpacing: 2, background: "rgba(255,255,255,0.04)", color: "#555",
              border: "1px solid rgba(255,255,255,0.07)",
            }}>🔬 VER REDE</button>
          </div>

          <p style={{ fontSize: 9, color: "#2d2d2d", lineHeight: 2, letterSpacing: 1 }}>
            A rede usa <span style={{ color: "#4aaeff" }}>backpropagation</span> com gradiente descendente
            (lr=0.08) e <span style={{ color: "#4aaeff" }}>softmax</span> na saída.
            Treine até 100% de acurácia, depois volte à aba REDE para ver os neurônios ajustados!
          </p>
        </div>
      )}

      {/* LEGEND */}
      <div style={{ marginTop: 18, display: "flex", gap: 18, flexWrap: "wrap", justifyContent: "center", alignItems: "center" }}>
        {[["#4aaeff", "PESO POSITIVO"], ["#ff5544", "PESO NEGATIVO"]].map(([c, l]) => (
          <div key={l} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 22, height: 3, background: c, borderRadius: 2 }} />
            <span style={{ fontSize: 9, color: "#2a2a2a", letterSpacing: 1 }}>{l}</span>
          </div>
        ))}
        <span style={{ fontSize: 9, color: "#1e1e1e", letterSpacing: 1 }}>ESPESSURA = MAGNITUDE DO SINAL</span>
      </div>
    </div>
  );
}