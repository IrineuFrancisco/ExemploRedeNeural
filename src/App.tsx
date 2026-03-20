import { useState, useEffect, useRef } from "react";

const WEIGHTS = {
  apple:  { yellow: -0.7, red:  0.6, rough: -0.4 },
  orange: { yellow:  0.6, red: -0.8, rough:  0.5 },
};

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function lerp(a, b, t) { return a + (b - a) * t; }

function computeOutput(inputs, weights) {
  const sum = inputs.yellow * weights.yellow + inputs.red * weights.red + inputs.rough * weights.rough;
  return sigmoid(sum * 3);
}

const APPLE_IMG  = "https://static.vecteezy.com/system/resources/previews/029/228/390/non_2x/apple-transparent-background-free-png.png";
const ORANGE_IMG = "https://static.vecteezy.com/system/resources/thumbnails/048/560/411/small/studio-shot-of-fresh-natural-orange-isolated-on-a-transparent-background-png.png";

export default function NeuralFruitNet() {
  const [inputs, setInputs] = useState({ yellow: 0, red: 0, rough: 0 });
  const [disp, setDisp]     = useState({ yellow: 0, red: 0, rough: 0, apple: 0.5, orange: 0.5 });
  const rafRef = useRef(null);

  const appleScore  = computeOutput(inputs, WEIGHTS.apple);
  const orangeScore = computeOutput(inputs, WEIGHTS.orange);

  useEffect(() => {
    const animate = () => {
      setDisp(prev => ({
        yellow: lerp(prev.yellow, inputs.yellow, 0.12),
        red:    lerp(prev.red,    inputs.red,    0.12),
        rough:  lerp(prev.rough,  inputs.rough,  0.12),
        apple:  lerp(prev.apple,  appleScore,    0.10),
        orange: lerp(prev.orange, orangeScore,   0.10),
      }));
      rafRef.current = requestAnimationFrame(animate);
    };
    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [inputs, appleScore, orangeScore]);

  const setInput = (key, val) => setInputs(p => ({ ...p, [key]: parseFloat(val) }));

  const yellowHue = lerp(210, 55, disp.yellow);
  const yellowSat = lerp(0, 95, disp.yellow);
  const yellowLit = lerp(80, 58, disp.yellow);
  const yellowColor = `hsl(${yellowHue},${yellowSat}%,${yellowLit}%)`;

  const redHue  = lerp(210, 2, disp.red);
  const redSat  = lerp(0, 88, disp.red);
  const redLit  = lerp(80, 42, disp.red);
  const redColor = `hsl(${redHue},${redSat}%,${redLit}%)`;

  const roughLit = lerp(85, 55, disp.rough);
  const roughColor = `hsl(215, ${disp.rough * 18}%, ${roughLit}%)`;

  // SVG connection lines
  // Layout: input nodes at x=100, output nodes at x=700
  // Yellow y=100, Red y=250, Rough y=400 (in a 500h SVG)
  // Apple y=150, Orange y=350
  const svgPts = {
    yellow: { x: 100, y: 100 },
    red:    { x: 100, y: 250 },
    rough:  { x: 100, y: 400 },
    apple:  { x: 700, y: 150 },
    orange: { x: 700, y: 350 },
  };

  const connections = [
    { from: "yellow", to: "apple",  w: WEIGHTS.apple.yellow  },
    { from: "yellow", to: "orange", w: WEIGHTS.orange.yellow },
    { from: "red",    to: "apple",  w: WEIGHTS.apple.red     },
    { from: "red",    to: "orange", w: WEIGHTS.orange.red    },
    { from: "rough",  to: "apple",  w: WEIGHTS.apple.rough   },
    { from: "rough",  to: "orange", w: WEIGHTS.orange.rough  },
  ];

  function SvgConnections() {
    return (
      <svg viewBox="0 0 800 500" style={{
        position: "absolute", inset: 0, width: "100%", height: "100%",
        pointerEvents: "none",
      }}>
        <defs>
          <marker id="ap" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 Z" fill="#4aaeff" />
          </marker>
          <marker id="an" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 Z" fill="#ff5544" />
          </marker>
        </defs>
        {connections.map((c, i) => {
          const f = svgPts[c.from];
          const t = svgPts[c.to];
          const inVal = disp[c.from];
          const pos = c.w > 0;
          const color = pos ? "#4aaeff" : "#ff5544";
          const thick = 1 + Math.abs(c.w) * inVal * 7;
          const alpha = 0.08 + inVal * 0.92;
          // midpoint label offset
          const mx = (f.x + 40 + t.x - 40) / 2;
          const my = (f.y + t.y) / 2;
          return (
            <g key={i}>
              <line
                x1={f.x + 42} y1={f.y}
                x2={t.x - 42} y2={t.y}
                stroke={color} strokeWidth={thick} opacity={alpha}
                strokeLinecap="round"
                markerEnd={pos ? "url(#ap)" : "url(#an)"}
              />
              <text x={mx} y={my - 7} fill={color} fontSize={10}
                textAnchor="middle" opacity={0.35 + inVal * 0.65}
                fontFamily="monospace">{c.w.toFixed(1)}</text>
            </g>
          );
        })}
      </svg>
    );
  }

  function Slider({ label, value, onChange, trackColor }) {
    return (
      <div style={{ width: "100%", padding: "0 4px" }}>
        <div style={{ display:"flex", justifyContent:"space-between", marginBottom: 3 }}>
          <span style={{ fontSize: 10, color: "#7ab", letterSpacing: 1 }}>{label}</span>
          <span style={{ fontSize: 10, color: "#cde", fontWeight: 700 }}>{value.toFixed(2)}</span>
        </div>
        <input type="range" min={0} max={1} step={0.01} value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          style={{
            width: "100%", height: 4, appearance: "none", outline: "none",
            borderRadius: 2, cursor: "pointer", border: "none",
            background: `linear-gradient(to right, ${trackColor} ${value*100}%, #1e3040 ${value*100}%)`,
          }}
        />
      </div>
    );
  }

  function OutputNode({ label, value, imgSrc, glowColor, emoji }) {
    return (
      <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap: 8 }}>
        <div style={{
          width: 90, height: 90, borderRadius: "50%",
          overflow: "hidden", position: "relative",
          border: `3px solid ${glowColor}${Math.round(40 + value * 200).toString(16).padStart(2,"0")}`,
          boxShadow: `0 0 ${6 + value * 55}px ${glowColor}${Math.round(value * 180).toString(16).padStart(2,"0")}`,
          transition: "box-shadow 0.25s, border-color 0.25s",
        }}>
          <img src={imgSrc} alt={label} style={{
            width: "100%", height: "100%", objectFit: "cover",
            opacity: 0.08 + value * 0.92,
            filter: `saturate(${0.1 + value * 1.9}) brightness(${0.5 + value * 0.65})`,
            transition: "opacity 0.25s, filter 0.25s",
          }} />
          <div style={{
            position: "absolute", inset: 0, display: "flex",
            alignItems: "center", justifyContent: "center",
          }}>
            <span style={{
              fontFamily: "monospace", fontSize: 15, fontWeight: 700,
              color: "#fff", textShadow: "0 1px 8px #0008",
            }}>{value.toFixed(2)}</span>
          </div>
        </div>
        <span style={{ fontSize: 11, color: glowColor, letterSpacing: 1 }}>{emoji} {label}</span>
        {/* Progress bar */}
        <div style={{ width: 80, height: 5, borderRadius: 3, background: "#0e1e2e", overflow:"hidden" }}>
          <div style={{
            height: "100%", borderRadius: 3,
            width: `${value * 100}%`,
            background: `linear-gradient(to right, ${glowColor}88, ${glowColor})`,
            transition: "width 0.1s",
            boxShadow: `0 0 6px ${glowColor}`,
          }} />
        </div>
      </div>
    );
  }

  const winner = disp.apple > 0.6 || disp.orange > 0.6
    ? (disp.apple > disp.orange ? { label: "MAÇÃ", emoji: "🍎", color: "#ff5544" }
                                 : { label: "LARANJA", emoji: "🍊", color: "#ff9922" })
    : null;

  return (
    <div style={{
      minHeight: "100vh",
      background: "radial-gradient(ellipse at 30% 20%, #0a2033 0%, #050d18 70%)",
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      padding: "24px 16px",
      fontFamily: "monospace",
    }}>
      <style>{`
        input[type=range]::-webkit-slider-thumb{appearance:none;width:15px;height:15px;border-radius:50%;background:#cde;cursor:pointer;box-shadow:0 0 8px #4af8;}
        input[type=range]::-moz-range-thumb{width:15px;height:15px;border-radius:50%;background:#cde;cursor:pointer;border:none;}
      `}</style>

      {/* Header */}
      <h1 style={{
        fontSize: "clamp(16px,3vw,24px)", fontWeight: 700, letterSpacing: 3,
        color: "#ddeeff", marginBottom: 4, textAlign:"center",
      }}>REDE NEURAL · CLASSIFICADOR DE FRUTAS</h1>
      <p style={{ fontSize: 10, color: "#3a5a7a", letterSpacing: 2, marginBottom: 28, textAlign:"center" }}>
        AJUSTE OS NEURÔNIOS DE ENTRADA
      </p>

      {/* Network canvas */}
      <div style={{
        position: "relative",
        width: "min(820px, 95vw)", height: "clamp(300px, 50vw, 510px)",
        background: "rgba(255,255,255,0.025)",
        borderRadius: 20, border: "1px solid rgba(255,255,255,0.07)",
        boxShadow: "0 4px 60px #00000088",
      }}>
        <SvgConnections />

        {/* Winner badge */}
        {winner && (
          <div style={{
            position:"absolute", top: 14, left:"50%", transform:"translateX(-50%)",
            background: `linear-gradient(135deg, ${winner.color}44, ${winner.color}22)`,
            border: `1px solid ${winner.color}66`,
            borderRadius: 20, padding: "4px 20px",
            fontSize: 11, color: winner.color, letterSpacing: 2, fontWeight: 700,
            backdropFilter: "blur(8px)",
          }}>
            {winner.emoji} {winner.label}
          </div>
        )}

        {/* Input layer */}
        <div style={{
          position:"absolute", left: 12, top: 0, bottom: 0,
          width: 175, display:"flex", flexDirection:"column",
          justifyContent:"space-around", padding:"20px 0",
        }}>
          {/* YELLOW */}
          <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:8 }}>
            <div style={{
              width: 76, height: 76, borderRadius:"50%",
              background: `radial-gradient(circle at 38% 33%, #ffffe0, ${yellowColor})`,
              display:"flex", alignItems:"center", justifyContent:"center",
              boxShadow: `0 0 ${8 + disp.yellow*36}px ${yellowColor}cc`,
              border:"3px solid rgba(255,255,255,0.18)",
            }}>
              <span style={{ fontSize:13, fontWeight:700, color:"#222" }}>{disp.yellow.toFixed(2)}</span>
            </div>
            <Slider label="Amarelo" value={inputs.yellow} onChange={v=>setInput("yellow",v)} trackColor="#f5d742" />
          </div>

          {/* RED */}
          <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:8 }}>
            <div style={{
              width: 76, height: 76, borderRadius:"50%",
              background: `radial-gradient(circle at 38% 33%, #ffcdd2, ${redColor})`,
              display:"flex", alignItems:"center", justifyContent:"center",
              boxShadow: `0 0 ${8 + disp.red*36}px ${redColor}cc`,
              border:"3px solid rgba(255,255,255,0.18)",
            }}>
              <span style={{ fontSize:13, fontWeight:700, color:"#fff" }}>{disp.red.toFixed(2)}</span>
            </div>
            <Slider label="Vermelho" value={inputs.red} onChange={v=>setInput("red",v)} trackColor="#ff4433" />
          </div>

          {/* ROUGH */}
          <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:8 }}>
            <div style={{
              width: 76, height: 76, borderRadius:"50%",
              background: `radial-gradient(circle at 38% 33%, #eceff1, ${roughColor})`,
              display:"flex", alignItems:"center", justifyContent:"center",
              boxShadow: `0 0 ${8 + disp.rough*28}px rgba(150,190,220,0.5)`,
              border:"3px solid rgba(255,255,255,0.18)",
            }}>
              <span style={{ fontSize:13, fontWeight:700, color:"#333" }}>{disp.rough.toFixed(2)}</span>
            </div>
            <Slider label="Rugosidade" value={inputs.rough} onChange={v=>setInput("rough",v)} trackColor="#88bbcc" />
          </div>
        </div>

        {/* Layer labels */}
        <div style={{ position:"absolute", left:40, bottom:8, fontSize:9, color:"#2a4a6a", letterSpacing:2 }}>ENTRADA</div>
        <div style={{ position:"absolute", right:30, bottom:8, fontSize:9, color:"#2a4a6a", letterSpacing:2 }}>SAÍDA</div>

        {/* Output layer */}
        <div style={{
          position:"absolute", right:14, top:0, bottom:0,
          width:155, display:"flex", flexDirection:"column",
          justifyContent:"space-around", alignItems:"center", padding:"30px 0",
        }}>
          <OutputNode label="Maçã" value={disp.apple} imgSrc={APPLE_IMG}
            glowColor="#ff5544" emoji="🍎" />
          <OutputNode label="Laranja" value={disp.orange} imgSrc={ORANGE_IMG}
            glowColor="#ff9922" emoji="🍊" />
        </div>
      </div>

      {/* Weight table */}
      <div style={{
        marginTop: 22, display:"grid",
        gridTemplateColumns:"120px repeat(3, 80px)",
        gap:"5px 16px",
        background:"rgba(255,255,255,0.03)",
        border:"1px solid rgba(255,255,255,0.06)",
        borderRadius:12, padding:"14px 24px",
      }}>
        {["", "Amarelo","Vermelho","Rugosidade"].map((h,i)=>(
          <span key={i} style={{ fontSize:9, color:"#3a5570", letterSpacing:1, textAlign:"center" }}>{h}</span>
        ))}
        {[
          { label:"→ Maçã",   w: WEIGHTS.apple,  color:"#ff7766" },
          { label:"→ Laranja",w: WEIGHTS.orange, color:"#ffaa55" },
        ].map((row,ri) => (
          <><span key={`l${ri}`} style={{ fontSize:10, color:row.color }}>{row.label}</span>
          {["yellow","red","rough"].map(k=>(
            <span key={k} style={{
              fontSize:12, fontWeight:700, textAlign:"center",
              color: row.w[k]>0 ? "#4aaeff" : "#ff5544",
            }}>{row.w[k].toFixed(1)}</span>
          ))}</>
        ))}
      </div>

      <p style={{ marginTop:14, fontSize:9, color:"#1e3040", letterSpacing:2 }}>
        PESOS FIXOS · SIGMOID · CAMADA ÚNICA
      </p>
    </div>
  );
}
