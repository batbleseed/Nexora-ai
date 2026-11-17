"use client";

import { memo, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Check,
  Code2,
  Copy,
  Cpu,
  Globe,
  Grid3X3,
  Layers,
  Loader2,
  Play,
  RotateCcw,
  StepForward,
} from "lucide-react";
import {
  runSingleNeuronDemo,
  XorMlpTrainer,
  type ComputationGraphSnapshot,
} from "../lib/microgradEngine";
import {
  matMulWithTrace,
  runEducationalTransformerForward,
  type Matrix,
} from "../lib/transformerEngine";
import {
  CORPUS_PRESETS,
  DeepNeuralLanguageModel,
  DEFAULT_NEXORA_TRAINING_CORPUS,
  ensureSharedNeuralModelReady,
  getSharedNexoraNeuralModel,
  harvestAndFeedNeuralEngine,
  type OptimizerKind,
} from "../lib/neuralLanguageEngine";

export type NeuralLabTab = "pipeline" | "micrograd" | "attention" | "matmul";

type NeuralEngineLabProps = {
  initialTab?: NeuralLabTab;
};

const PURE_TS_SNIPPETS: Record<NeuralLabTab, string> = {
  pipeline: `// TensorFlow.js (@tensorflow/tfjs) Deep Neural Language Pipeline
import * as tf from "@tensorflow/tfjs";

const model = tf.sequential();
model.add(tf.layers.embedding({ inputDim: vocabSize, outputDim: 24, inputLength: 6 }));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({ units: 96, activation: "tanh" }));
model.add(tf.layers.dropout({ rate: 0.1 }));
model.add(tf.layers.dense({ units: 96, activation: "relu" }));
model.add(tf.layers.dense({ units: vocabSize, activation: "softmax" }));

model.compile({
  optimizer: optimizer === "sgd" ? tf.train.sgd(0.08) : tf.train.adam(0.008),
  loss: "sparseCategoricalCrossentropy",
  metrics: ["accuracy"],
});

await model.fit(xs, ys, { epochs: 16, batchSize: 64, shuffle: true });
const probs = model.predict(promptTensor) as tf.Tensor;`,
  micrograd: `// Pure TypeScript scalar autograd (zero external libraries)
class Value {
  grad = 0;
  _backward = () => {};
  constructor(public data: number, public _prev: Value[] = [], public op = "") {}

  mul(other: Value) {
    const out = new Value(this.data * other.data, [this, other], "×");
    out._backward = () => {
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    };
    return out;
  }

  tanh() {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], "tanh");
    out._backward = () => { this.grad += (1 - t * t) * out.grad; };
    return out;
  }

  backward() {
    const topo: Value[] = [], seen = new Set<Value>();
    const visit = (v: Value) => {
      if (seen.has(v)) return;
      seen.add(v);
      v._prev.forEach(visit);
      topo.push(v);
    };
    visit(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }
}`,
  attention: `// Pure TypeScript Scaled Dot-Product Attention: softmax((Q·Kᵀ) / √d_k) · V
function scaledDotProductAttention(Q: number[][], K: number[][], V: number[][], causal: boolean) {
  const dk = K[0].length;
  const scale = 1 / Math.sqrt(dk);
  const scores = matMul(Q, transpose(K)).map((row, r) =>
    row.map((val, c) => (causal && c > r ? -Infinity : val * scale))
  );
  const weights = scores.map((row) => {
    const finite = row.filter(Number.isFinite);
    const max = finite.length ? Math.max(...finite) : 0;
    const exps = row.map((x) => (Number.isFinite(x) ? Math.exp(x - max) : 0));
    const sum = exps.reduce((a, b) => a + b, 0) || 1;
    return exps.map((e) => e / sum);
  });
  return { weights, output: matMul(weights, V) };
}`,
  matmul: `// Pure TypeScript Matrix Multiplication: (M × K) · (K × N) -> (M × N)
function matMul(A: number[][], B: number[][]): number[][] {
  const rows = A.length, cols = B[0].length, inner = B.length;
  const C = Array.from({ length: rows }, () => Array(cols).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      let sum = 0;
      for (let k = 0; k < inner; k++) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
  return C;
}`,
};

function fmt(n: number, digits = 3): string {
  if (!Number.isFinite(n)) return "-∞";
  const rounded = Math.abs(n) < 1e-6 ? 0 : n;
  return rounded.toFixed(digits);
}

function ComputationGraphView({ graph }: { graph: ComputationGraphSnapshot }) {
  const maxDepth = Math.max(0, ...graph.nodes.map((n) => n.depth));
  const colW = 148;
  const nodeW = 122;
  const nodeH = 54;
  const rowGap = 72;
  const padX = 18;
  const padY = 18;

  const laneCounts = new Map<number, number>();
  for (const node of graph.nodes) {
    laneCounts.set(node.depth, Math.max(laneCounts.get(node.depth) ?? 0, node.lane + 1));
  }
  const maxLanes = Math.max(1, ...Array.from(laneCounts.values()));
  const svgW = Math.max(580, (maxDepth + 1) * colW + padX * 2);
  const svgH = Math.max(210, maxLanes * rowGap + padY * 2);

  const posById = new Map<string, { x: number; y: number }>();
  for (const node of graph.nodes) {
    const count = laneCounts.get(node.depth) ?? 1;
    const totalH = (count - 1) * rowGap + nodeH;
    const startY = (svgH - totalH) / 2;
    posById.set(node.id, {
      x: padX + node.depth * colW,
      y: startY + node.lane * rowGap,
    });
  }

  return (
    <div className="overflow-x-auto rounded-[11px] border border-line bg-inset/60">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="block min-w-full"
        role="img"
        aria-label="Autograd computation graph"
      >
        {graph.edges.map((edge, idx) => {
          const from = posById.get(edge.fromId);
          const to = posById.get(edge.toId);
          if (!from || !to) return null;
          const x1 = from.x + nodeW;
          const y1 = from.y + nodeH / 2;
          const x2 = to.x;
          const y2 = to.y + nodeH / 2;
          const cx = (x1 + x2) / 2;
          return (
            <path
              key={`${edge.fromId}-${edge.toId}-${idx}`}
              d={`M ${x1} ${y1} C ${cx} ${y1}, ${cx} ${y2}, ${x2} ${y2}`}
              fill="none"
              stroke="var(--line-strong)"
              strokeWidth="1.5"
            />
          );
        })}

        {graph.nodes.map((node) => {
          const pos = posById.get(node.id);
          if (!pos) return null;
          const isLoss = node.label === "Loss";
          const hasGrad = Math.abs(node.grad) > 1e-4;
          return (
            <g key={node.id} transform={`translate(${pos.x}, ${pos.y})`}>
              <rect
                width={nodeW}
                height={nodeH}
                rx="9"
                fill="var(--surface)"
                stroke={isLoss ? "var(--accent)" : hasGrad ? "var(--line-strong)" : "var(--line)"}
                strokeWidth={isLoss ? "1.5" : "1"}
              />
              <text
                x="10"
                y="17"
                fontSize="11"
                fontWeight="600"
                fill="var(--ink)"
                fontFamily="ui-monospace, monospace"
              >
                {node.label}
                {node.op ? ` (${node.op})` : ""}
              </text>
              <text
                x="10"
                y="33"
                fontSize="10.5"
                fill="var(--ink-2)"
                fontFamily="ui-monospace, monospace"
              >
                data {fmt(node.data, 2)}
              </text>
              <text
                x="10"
                y="46"
                fontSize="10.5"
                fontWeight="600"
                fill={hasGrad ? "var(--accent-ink)" : "var(--ink-3)"}
                fontFamily="ui-monospace, monospace"
              >
                grad {fmt(node.grad, 2)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function MatrixGrid({
  title,
  subtitle,
  matrix,
  rowLabels,
  colLabels,
  highlightRow,
  highlightCol,
  activeCell,
  onHoverCell,
  heatMode = false,
}: {
  title: string;
  subtitle?: string;
  matrix: Matrix;
  rowLabels?: string[];
  colLabels?: string[];
  highlightRow?: number | null;
  highlightCol?: number | null;
  activeCell?: { row: number; col: number } | null;
  onHoverCell?: (row: number, col: number) => void;
  heatMode?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[12px] font-semibold text-ink">{title}</span>
        {subtitle && <span className="font-mono text-[11px] text-ink-3">{subtitle}</span>}
      </div>
      <div className="overflow-x-auto rounded-[10px] border border-line bg-inset/50 p-2">
        <table className="w-full border-collapse font-mono text-[11.5px] tabular-nums">
          {colLabels && (
            <thead>
              <tr>
                {rowLabels && <th className="px-1.5 py-1 text-left text-ink-3" />}
                {colLabels.map((c, j) => (
                  <th
                    key={`${c}-${j}`}
                    className={`px-2 py-1 text-center font-medium ${
                      highlightCol === j ? "text-accent-ink" : "text-ink-3"
                    }`}
                  >
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
          )}
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                {rowLabels && (
                  <td
                    className={`px-2 py-1 text-left font-medium ${
                      highlightRow === i ? "text-accent-ink" : "text-ink-3"
                    }`}
                  >
                    {rowLabels[i] ?? `r${i}`}
                  </td>
                )}
                {row.map((val, j) => {
                  const isActive = activeCell?.row === i && activeCell?.col === j;
                  const isHighlighted = highlightRow === i || highlightCol === j;
                  const heatOpacity = heatMode && Number.isFinite(val) ? Math.min(0.85, Math.max(0.04, val)) : 0;
                  return (
                    <td key={j} className="p-0.5">
                      <button
                        type="button"
                        onMouseEnter={() => onHoverCell?.(i, j)}
                        onClick={() => onHoverCell?.(i, j)}
                        style={
                          heatMode
                            ? {
                                backgroundColor: Number.isFinite(val)
                                  ? `color-mix(in srgb, var(--accent) ${Math.round(heatOpacity * 100)}%, var(--surface))`
                                  : "var(--inset)",
                              }
                            : undefined
                        }
                        className={`w-full rounded-[6px] px-2 py-1.5 text-center transition-colors ${
                          isActive
                            ? "bg-ink font-semibold text-surface ring-1 ring-ink"
                            : isHighlighted
                              ? "bg-hover-2 font-medium text-ink"
                              : heatMode
                                ? "text-ink"
                                : "bg-surface text-ink-2 hover:bg-hover"
                        }`}
                      >
                        {fmt(val, heatMode ? 2 : 3)}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function NeuralEngineLab({ initialTab = "micrograd" }: NeuralEngineLabProps) {
  const [tab, setTab] = useState<NeuralLabTab>(initialTab);
  const [showCode, setShowCode] = useState(false);
  const [copiedCode, setCopiedCode] = useState(false);

  // --- Tab 0: 4-Stage Deep AI Pipeline state (TensorFlow.js shared live Nexora model) ---
  const lmModelRef = useRef<DeepNeuralLanguageModel | null>(null);
  if (!lmModelRef.current) {
    lmModelRef.current = getSharedNexoraNeuralModel();
  }
  const [corpusPresetKey, setCorpusPresetKey] = useState<keyof typeof CORPUS_PRESETS>("ai_core");
  const [corpusText, setCorpusText] = useState(
    () => lmModelRef.current?.getCorpusPreview(14_000) || DEFAULT_NEXORA_TRAINING_CORPUS
  );
  const [inferPrompt, setInferPrompt] = useState("neural networks learn");
  const [optimizerKind, setOptimizerKind] = useState<OptimizerKind>("adam");
  const [isFittingModel, setIsFittingModel] = useState(false);
  const [webFeedQuery, setWebFeedQuery] = useState("");
  const [isHarvestingWeb, setIsHarvestingWeb] = useState(false);
  const [webHarvestStatus, setWebHarvestStatus] = useState<string | null>(null);
  const [webHarvestError, setWebHarvestError] = useState<string | null>(null);
  const [lmTick, setLmTick] = useState(0);

  useEffect(() => {
    let active = true;
    void ensureSharedNeuralModelReady().then(() => {
      if (!active) return;
      setCorpusText(lmModelRef.current?.getCorpusPreview(14_000) || DEFAULT_NEXORA_TRAINING_CORPUS);
      setLmTick((t) => t + 1);
    });
    return () => {
      active = false;
    };
  }, []);

  const lmSnapshot = useMemo(
    () => lmModelRef.current!.getSnapshot(inferPrompt),
    [lmTick, inferPrompt]
  );
  const lmInference = useMemo(
    () => lmModelRef.current!.predictAndGenerate(inferPrompt || "data", 16, 0.6, 5),
    [lmTick, inferPrompt]
  );

  const runTfFit = async (epochs: number, resetText?: string) => {
    if (isFittingModel) return;
    setIsFittingModel(true);
    try {
      const model = lmModelRef.current!;
      if (resetText !== undefined) {
        model.loadCorpus(resetText, 2026);
      }
      await model.fitEpochs(epochs, {
        optimizer: optimizerKind,
        onEpochProgress: () => setLmTick((t) => t + 1),
      });
      setLmTick((t) => t + 1);
    } finally {
      setIsFittingModel(false);
    }
  };

  const handleHarvestWebCorpus = async () => {
    if (isHarvestingWeb || isFittingModel) return;
    setIsHarvestingWeb(true);
    setWebHarvestError(null);
    setWebHarvestStatus("Crawling real websites & fitting TensorFlow.js model...");
    try {
      const trimmed = webFeedQuery.trim();
      const looksLikeUrl =
        /^https?:\/\//i.test(trimmed) ||
        /^[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:\/.*)?$/i.test(trimmed);
      const { harvest, after } = await harvestAndFeedNeuralEngine({
        url: looksLikeUrl ? trimmed : undefined,
        topic: !looksLikeUrl && trimmed.length >= 2 ? trimmed : undefined,
        maxPages: 12,
        epochs: 16,
        optimizer: optimizerKind,
      });
      setCorpusText(lmModelRef.current!.getCorpusPreview(14_000));
      setWebHarvestStatus(
        `TFJS (${after.backend}) trained ${after.paramCount.toLocaleString()} weights on ${harvest.totalWords.toLocaleString()} words (${harvest.sources.length} web pages) -> loss ${after.currentLoss.toFixed(3)}, acc ${(after.accuracy * 100).toFixed(1)}%`
      );
      setLmTick((t) => t + 1);
    } catch (err) {
      setWebHarvestError(
        err instanceof Error ? err.message : "Unable to fetch web corpus."
      );
      setWebHarvestStatus(null);
    } finally {
      setIsHarvestingWeb(false);
    }
  };

  // --- Tab 1: Micrograd state ---
  const [microgradMode, setMicrogradMode] = useState<"neuron" | "xor">("neuron");
  const [neuronParams, setNeuronParams] = useState({
    x1: 2.0,
    w1: -3.0,
    x2: 0.0,
    w2: 1.0,
    b: 6.8814,
    target: 1.0,
  });

  const neuronDemo = useMemo(() => runSingleNeuronDemo(neuronParams), [neuronParams]);

  const xorTrainerRef = useRef<XorMlpTrainer | null>(null);
  if (!xorTrainerRef.current) {
    xorTrainerRef.current = new XorMlpTrainer(1337);
  }
  const [xorSnapshot, setXorSnapshot] = useState(() => xorTrainerRef.current!.evaluate());

  const stepNeuronSgd = (lr = 0.2) => {
    setNeuronParams((prev) => ({
      ...prev,
      w1: Number((prev.w1 - lr * neuronDemo.grads.w1).toFixed(4)),
      w2: Number((prev.w2 - lr * neuronDemo.grads.w2).toFixed(4)),
      b: Number((prev.b - lr * neuronDemo.grads.b).toFixed(4)),
    }));
  };

  const resetNeuron = () => {
    setNeuronParams({
      x1: 2.0,
      w1: -3.0,
      x2: 0.0,
      w2: 1.0,
      b: 6.8814,
      target: 1.0,
    });
  };

  // --- Tab 2: Attention / Transformer state ---
  const [tokenInput, setTokenInput] = useState("The cat sat down");
  const [causalMask, setCausalMask] = useState(true);
  const [dkDim, setDkDim] = useState(3);
  const [selectedAttnMatrix, setSelectedAttnMatrix] = useState<
    "weights" | "raw" | "qkv" | "norm"
  >("weights");
  const [hoveredAttnCell, setHoveredAttnCell] = useState<{ row: number; col: number }>({
    row: 1,
    col: 0,
  });

  const tokens = useMemo(() => {
    const split = tokenInput
      .trim()
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 6);
    return split.length > 0 ? split : ["The", "cat", "sat"];
  }, [tokenInput]);

  const transformerTrace = useMemo(
    () =>
      runEducationalTransformerForward({
        tokens,
        causal: causalMask,
        dModel: 4,
        dk: dkDim,
      }),
    [tokens, causalMask, dkDim]
  );

  // --- Tab 3: Matrix Multiplication state ---
  const [matA, setMatA] = useState<Matrix>([
    [1, 2, -1],
    [0, 3, 2],
  ]);
  const [matB, setMatB] = useState<Matrix>([
    [2, 1],
    [-1, 4],
    [3, 0],
  ]);
  const [selectedMatCell, setSelectedMatCell] = useState<{ row: number; col: number }>({
    row: 0,
    col: 0,
  });

  const matMulTrace = useMemo(() => matMulWithTrace(matA, matB), [matA, matB]);
  const activeCellDetail =
    matMulTrace.cells[selectedMatCell.row]?.[selectedMatCell.col] ?? matMulTrace.cells[0][0];

  const handleCopySnippet = () => {
    const code = PURE_TS_SNIPPETS[tab];
    void navigator.clipboard.writeText(code).then(() => {
      setCopiedCode(true);
      window.setTimeout(() => setCopiedCode(false), 1500);
    });
  };

  return (
    <div className="my-2 w-full overflow-hidden rounded-card border border-line bg-surface shadow-card">
      {/* Lab Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-2.5">
        <div className="flex items-center gap-2">
          <span className="flex size-6 items-center justify-center rounded-[7px] bg-inset text-ink">
            <Cpu className="size-3.5" />
          </span>
          <div className="flex items-baseline gap-2">
            <span className="text-[13px] font-semibold text-ink">
              Pure TypeScript Engine Lab
            </span>
            <span className="hidden text-[11.5px] text-ink-3 sm:inline">
              Zero external ML libs
            </span>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          <div
            role="tablist"
            aria-label="Engine lab sections"
            className="inline-flex items-center rounded-[9px] border border-line bg-inset p-0.5"
          >
            {(
              [
                { id: "pipeline", label: "Deep AI Pipeline", icon: Activity },
                { id: "micrograd", label: "Micrograd", icon: Layers },
                { id: "attention", label: "Attention", icon: Cpu },
                { id: "matmul", label: "MatMul", icon: Grid3X3 },
              ] as const
            ).map((item) => {
              const Icon = item.icon;
              const active = tab === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  onClick={() => setTab(item.id)}
                  className={`flex h-7 items-center gap-1.5 rounded-[7px] px-2.5 text-[11.5px] font-medium transition-colors ${
                    active
                      ? "bg-surface text-ink shadow-xs"
                      : "text-ink-2 hover:text-ink"
                  }`}
                >
                  <Icon className="size-3" />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </div>

          <button
            type="button"
            onClick={() => setShowCode((v) => !v)}
            aria-pressed={showCode}
            className={`flex h-8 items-center gap-1 rounded-[8px] border px-2.5 text-[11.5px] font-medium transition-colors ${
              showCode
                ? "border-ink bg-ink text-surface"
                : "border-line-strong/80 bg-surface text-ink-2 hover:bg-hover hover:text-ink"
            }`}
          >
            <Code2 className="size-3.5" />
            <span className="hidden sm:inline">TS Source</span>
          </button>
        </div>
      </div>

      {/* Optional Pure TS Source Drawer */}
      {showCode && (
        <div className="border-b border-line bg-inset/70 px-4 py-3">
          <div className="mb-2 flex items-center justify-between">
            <span className="font-mono text-[11.5px] font-medium text-ink-2">
              {tab === "pipeline"
                ? "neuralLanguageEngine.ts — TensorFlow.js Sequential + model.fit() + Adam/SGD"
                : tab === "micrograd"
                  ? "microgradEngine.ts — Scalar Value Autograd"
                  : tab === "attention"
                    ? "transformerEngine.ts — Scaled Dot-Product Attention"
                    : "transformerEngine.ts — Pure TS Matrix Multiplication"}
            </span>
            <button
              type="button"
              onClick={handleCopySnippet}
              className="flex h-7 items-center gap-1 rounded-[6px] border border-line bg-surface px-2 text-[11px] font-medium text-ink-2 hover:text-ink"
            >
              {copiedCode ? <Check className="size-3 text-green" /> : <Copy className="size-3" />}
              <span>{copiedCode ? "Copied" : "Copy"}</span>
            </button>
          </div>
          <pre className="overflow-x-auto rounded-[9px] border border-line bg-surface p-3 font-mono text-[11.5px] leading-relaxed text-ink">
            <code>{PURE_TS_SNIPPETS[tab]}</code>
          </pre>
        </div>
      )}

      {/* TAB 0: 4-STAGE TENSORFLOW.JS DEEP AI PIPELINE */}
      {tab === "pipeline" && (
        <div className="flex flex-col gap-4 p-4">
          {/* Training Controls & Optimizer Selector */}
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-1.5">
              {(Object.keys(CORPUS_PRESETS) as Array<keyof typeof CORPUS_PRESETS>).map((key) => (
                <button
                  key={key}
                  type="button"
                  disabled={isFittingModel || isHarvestingWeb}
                  onClick={() => {
                    setCorpusPresetKey(key);
                    const nextText = CORPUS_PRESETS[key].text;
                    setCorpusText(nextText);
                    void runTfFit(12, nextText);
                  }}
                  className={`h-7 rounded-[8px] px-2.5 text-[11.5px] font-medium transition-colors disabled:opacity-50 ${
                    corpusPresetKey === key
                      ? "bg-hover-2 text-ink"
                      : "text-ink-2 hover:bg-hover hover:text-ink"
                  }`}
                >
                  {CORPUS_PRESETS[key].label}
                </button>
              ))}
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <div className="inline-flex items-center rounded-[8px] border border-line bg-inset p-0.5 text-[11px]">
                {(["adam", "sgd"] as const).map((opt) => (
                  <button
                    key={opt}
                    type="button"
                    disabled={isFittingModel}
                    onClick={() => {
                      setOptimizerKind(opt);
                      lmModelRef.current!.setOptimizer(opt);
                      setLmTick((t) => t + 1);
                    }}
                    className={`h-6 rounded-[6px] px-2 font-mono font-medium transition-colors ${
                      optimizerKind === opt
                        ? "bg-surface text-ink shadow-xs"
                        : "text-ink-2 hover:text-ink"
                    }`}
                  >
                    tf.train.{opt}
                  </button>
                ))}
              </div>

              <button
                type="button"
                disabled={isFittingModel || isHarvestingWeb}
                onClick={() => void runTfFit(5)}
                className="flex h-7 items-center gap-1 rounded-[8px] border border-line-strong/80 bg-surface px-2.5 text-[11.5px] font-medium text-ink hover:bg-hover disabled:opacity-50"
              >
                <StepForward className="size-3" />
                <span>+5 Epochs</span>
              </button>
              <button
                type="button"
                disabled={isFittingModel || isHarvestingWeb}
                onClick={() => void runTfFit(20)}
                className="flex h-7 items-center gap-1 rounded-[8px] bg-ink px-2.5 text-[11.5px] font-medium text-surface transition-opacity hover:opacity-90 disabled:opacity-50"
              >
                {isFittingModel ? (
                  <Loader2 className="size-3 animate-spin" />
                ) : (
                  <Play className="size-3" />
                )}
                <span>{isFittingModel ? "Fitting..." : "model.fit(20)"}</span>
              </button>
              <button
                type="button"
                disabled={isFittingModel || isHarvestingWeb}
                onClick={() => {
                  lmModelRef.current!.loadCorpus(corpusText, 2026);
                  setLmTick((t) => t + 1);
                }}
                className="flex h-7 items-center gap-1 rounded-[8px] border border-line-strong/80 bg-surface px-2.5 text-[11.5px] font-medium text-ink-2 hover:text-ink disabled:opacity-50"
              >
                <RotateCcw className="size-3" />
                <span>Reset Weights</span>
              </button>
            </div>
          </div>

          {/* 4 Live TensorFlow.js Neural Metrics */}
          <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
            <div className="rounded-[10px] border border-line bg-inset/40 px-3 py-2">
              <div className="text-[11px] text-ink-3">1. TFJS ({lmSnapshot.backend}) · Params</div>
              <div className="mt-0.5 font-mono text-[13px] font-semibold text-ink">
                {lmSnapshot.vocabSize} tok · {lmSnapshot.paramCount.toLocaleString()} w
              </div>
            </div>
            <div className="rounded-[10px] border border-line bg-inset/40 px-3 py-2">
              <div className="text-[11px] text-ink-3">2. model.fit (tf.train.{lmSnapshot.optimizer})</div>
              <div className="mt-0.5 font-mono text-[13px] font-semibold text-ink">
                {lmSnapshot.stepCount} epochs
              </div>
            </div>
            <div className="rounded-[10px] border border-line bg-inset/40 px-3 py-2">
              <div className="text-[11px] text-ink-3">3. Cross-Entropy Loss</div>
              <div className="mt-0.5 font-mono text-[13px] font-semibold text-orange">
                {fmt(lmSnapshot.currentLoss, 3)}{" "}
                <span className="text-[11px] font-normal text-ink-3">
                  (from {fmt(lmSnapshot.initialLoss, 2)})
                </span>
              </div>
            </div>
            <div className="rounded-[10px] border border-line bg-inset/40 px-3 py-2">
              <div className="text-[11px] text-ink-3">Accuracy · PPL</div>
              <div className="mt-0.5 font-mono text-[13px] font-semibold text-accent-ink">
                {(lmSnapshot.accuracy * 100).toFixed(1)}% · {fmt(lmSnapshot.perplexity, 1)}
              </div>
            </div>
          </div>

          {/* Step 1: Data & Input Tokenization + Step 2-3: Loss Curve */}
          <div className="flex flex-col gap-2 rounded-[10px] border border-line bg-inset/40 p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-1.5 text-[11.5px] font-medium text-ink">
                <Globe className="size-3.5 text-accent-ink" />
                <span>Harvest Live Website / Web Data → Train TFJS Model</span>
              </div>
              <span className="font-mono text-[11px] text-ink-3">
                {lmSnapshot.corpusWords.toLocaleString()} words ·{" "}
                {(lmSnapshot.corpusChars / 1024).toFixed(1)} KB
                {lmSnapshot.sourcesFed.length > 0
                  ? ` · ${lmSnapshot.sourcesFed.length} web pages`
                  : ""}
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <input
                type="text"
                value={webFeedQuery}
                onChange={(e) => setWebFeedQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    void handleHarvestWebCorpus();
                  }
                }}
                placeholder="Website URL (https://...) or topic (blank = broad multi-site harvest)"
                className="h-8 min-w-[220px] flex-1 rounded-[8px] border border-line-strong/80 bg-surface px-3 font-mono text-[11.5px] text-ink outline-none focus:border-ink"
              />
              <button
                type="button"
                disabled={isHarvestingWeb || isFittingModel}
                onClick={() => void handleHarvestWebCorpus()}
                className="flex h-8 items-center gap-1.5 rounded-[8px] bg-ink px-3 text-[11.5px] font-medium text-surface transition-opacity hover:opacity-90 disabled:opacity-60"
              >
                {isHarvestingWeb ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : (
                  <Globe className="size-3.5" />
                )}
                <span>{isHarvestingWeb ? "Harvesting & Fitting..." : "Fetch Web Data & Train"}</span>
              </button>
            </div>

            {webHarvestStatus && (
              <div className="font-mono text-[11px] text-green">{webHarvestStatus}</div>
            )}
            {webHarvestError && (
              <div className="font-mono text-[11px] text-orange">{webHarvestError}</div>
            )}

            {lmSnapshot.sourcesFed.length > 0 && (
              <div className="flex flex-wrap gap-1.5 pt-0.5">
                {lmSnapshot.sourcesFed.slice(0, 8).map((src) => (
                  <a
                    key={src.url}
                    href={src.url}
                    target="_blank"
                    rel="noreferrer"
                    className="rounded-[6px] border border-line bg-surface px-2 py-0.5 font-mono text-[10.5px] text-ink-2 hover:border-line-strong hover:text-ink"
                    title={src.title}
                  >
                    {src.title.slice(0, 34)} ({src.words.toLocaleString()}w)
                  </a>
                ))}
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-[11.5px]">
                <span className="font-medium text-ink">1. Data (The Fuel) — Active Training Corpus</span>
                <button
                  type="button"
                  disabled={isFittingModel || isHarvestingWeb}
                  onClick={() => void runTfFit(12, corpusText)}
                  className="font-medium text-accent-ink hover:underline disabled:opacity-50"
                >
                  Apply &amp; Fit
                </button>
              </div>
              <textarea
                value={corpusText}
                onChange={(e) => setCorpusText(e.target.value)}
                rows={4}
                className="w-full rounded-[10px] border border-line-strong/80 bg-inset p-2.5 font-mono text-[11.5px] leading-relaxed text-ink outline-none focus:border-ink"
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-[11.5px]">
                <span className="font-medium text-ink">2–3. Training &amp; Backprop Loss Descent</span>
                <span className="font-mono text-ink-3">
                  ΔW = {fmt(lmSnapshot.lastWeightDeltaNorm, 3)}
                </span>
              </div>
              <div className="flex flex-1 flex-col justify-between rounded-[10px] border border-line bg-inset/50 p-3">
                <svg viewBox="0 0 320 82" className="h-20 w-full" aria-label="Neural language model training loss">
                  {(() => {
                    const hist = lmSnapshot.lossHistory;
                    const maxL = Math.max(0.5, ...hist);
                    const pts = hist
                      .map((val, idx) => {
                        const x = hist.length <= 1 ? 160 : (idx / (hist.length - 1)) * 308 + 6;
                        const y = 74 - (val / maxL) * 64;
                        return `${x.toFixed(1)},${y.toFixed(1)}`;
                      })
                      .join(" ");
                    return (
                      <polyline
                        fill="none"
                        stroke="var(--accent)"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        points={pts}
                      />
                    );
                  })()}
                </svg>
                <div className="mt-1 flex flex-wrap items-center gap-1.5 font-mono text-[10.5px] text-ink-2">
                  <span className="text-ink-3">Tokens:</span>
                  {lmInference.promptTokens.map((tok, i) => (
                    <span
                      key={`${tok}-${i}`}
                      className="rounded-[5px] border border-line bg-surface px-1.5 py-0.5 text-ink"
                    >
                      {tok}→#{lmInference.promptTokenIds[i]}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Step 4: Inference (Next-Token Probabilities + Autoregressive Generation) */}
          <div className="flex flex-col gap-2.5 rounded-[10px] border border-line bg-inset/40 p-3">
            <div className="flex flex-wrap items-end justify-between gap-2">
              <label className="flex min-w-[210px] flex-1 flex-col gap-1">
                <span className="text-[11.5px] font-medium text-ink">
                  4. Inference — Prompt Starter (Trained Weights Predict &amp; Generate)
                </span>
                <input
                  type="text"
                  value={inferPrompt}
                  onChange={(e) => setInferPrompt(e.target.value)}
                  placeholder="Type words from your training data..."
                  className="h-8 rounded-[8px] border border-line-strong/80 bg-surface px-3 font-mono text-[12px] text-ink outline-none focus:border-ink"
                />
              </label>
              <div className="font-mono text-[11px] text-ink-3">
                ‖h₂‖ = {fmt(lmInference.hiddenActivationNorm, 2)}
              </div>
            </div>

            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              <div className="flex flex-col gap-1.5">
                <div className="text-[11px] font-medium text-ink-2">
                  Top-5 Next-Token Probabilities (Softmax)
                </div>
                <div className="flex flex-col gap-1">
                  {lmInference.topNextTokens.map((cand) => (
                    <div
                      key={cand.id}
                      className="flex items-center gap-2 font-mono text-[11.5px]"
                    >
                      <span className="w-24 truncate text-ink">{cand.token}</span>
                      <div className="h-2 flex-1 overflow-hidden rounded-full bg-inset">
                        <div
                          className="h-full rounded-full bg-accent"
                          style={{
                            width: `${Math.min(100, Math.max(3, cand.probability * 100))}%`,
                          }}
                        />
                      </div>
                      <span className="w-12 text-right text-ink-2">
                        {(cand.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex flex-col gap-1.5">
                <div className="text-[11px] font-medium text-ink-2">
                  Autoregressive Generated Continuation
                </div>
                <div className="flex-1 rounded-[8px] border border-line bg-surface p-2.5 text-[12.5px] leading-relaxed text-ink">
                  <span className="font-medium text-ink-2">{inferPrompt.trim()}</span>{" "}
                  <span className="text-accent-ink">
                    {lmInference.generatedContinuation || "..."}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 1: MICROGRAD & BACKPROP */}
      {tab === "micrograd" && (
        <div className="flex flex-col gap-4 p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => setMicrogradMode("neuron")}
                className={`h-7 rounded-[8px] px-2.5 text-[12px] font-medium transition-colors ${
                  microgradMode === "neuron"
                    ? "bg-hover-2 text-ink"
                    : "text-ink-2 hover:bg-hover hover:text-ink"
                }`}
              >
                1. Single Neuron Graph
              </button>
              <button
                type="button"
                onClick={() => setMicrogradMode("xor")}
                className={`h-7 rounded-[8px] px-2.5 text-[12px] font-medium transition-colors ${
                  microgradMode === "xor"
                    ? "bg-hover-2 text-ink"
                    : "text-ink-2 hover:bg-hover hover:text-ink"
                }`}
              >
                2. Train XOR MLP (2 → 4 → 1)
              </button>
            </div>

            {microgradMode === "neuron" ? (
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => stepNeuronSgd(0.25)}
                  className="flex h-7 items-center gap-1 rounded-[8px] bg-ink px-2.5 text-[11.5px] font-medium text-surface transition-opacity hover:opacity-90"
                >
                  <StepForward className="size-3" />
                  <span>Step SGD (-0.25·grad)</span>
                </button>
                <button
                  type="button"
                  onClick={resetNeuron}
                  className="flex h-7 items-center gap-1 rounded-[8px] border border-line-strong/80 bg-surface px-2.5 text-[11.5px] font-medium text-ink-2 hover:text-ink"
                >
                  <RotateCcw className="size-3" />
                  <span>Reset</span>
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setXorSnapshot(xorTrainerRef.current!.stepEpochs(1))}
                  className="flex h-7 items-center gap-1 rounded-[8px] border border-line-strong/80 bg-surface px-2.5 text-[11.5px] font-medium text-ink hover:bg-hover"
                >
                  <StepForward className="size-3" />
                  <span>+1 Epoch</span>
                </button>
                <button
                  type="button"
                  onClick={() => setXorSnapshot(xorTrainerRef.current!.stepEpochs(25))}
                  className="flex h-7 items-center gap-1 rounded-[8px] bg-ink px-2.5 text-[11.5px] font-medium text-surface transition-opacity hover:opacity-90"
                >
                  <Play className="size-3" />
                  <span>Train 25 Epochs</span>
                </button>
                <button
                  type="button"
                  onClick={() => {
                    xorTrainerRef.current = new XorMlpTrainer(1337);
                    setXorSnapshot(xorTrainerRef.current.evaluate());
                  }}
                  className="flex h-7 items-center gap-1 rounded-[8px] border border-line-strong/80 bg-surface px-2.5 text-[11.5px] font-medium text-ink-2 hover:text-ink"
                >
                  <RotateCcw className="size-3" />
                  <span>Reset</span>
                </button>
              </div>
            )}
          </div>

          {microgradMode === "neuron" ? (
            <>
              <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-6">
                {(
                  [
                    { key: "x1", label: "x1 (input)", min: -3, max: 3, step: 0.1 },
                    { key: "w1", label: "w1 (weight)", min: -4, max: 4, step: 0.1 },
                    { key: "x2", label: "x2 (input)", min: -3, max: 3, step: 0.1 },
                    { key: "w2", label: "w2 (weight)", min: -4, max: 4, step: 0.1 },
                    { key: "b", label: "b (bias)", min: -4, max: 8, step: 0.1 },
                    { key: "target", label: "target y", min: -1, max: 1, step: 0.1 },
                  ] as const
                ).map((ctrl) => (
                  <label
                    key={ctrl.key}
                    className="flex flex-col gap-1 rounded-[10px] border border-line bg-inset/50 px-2.5 py-2"
                  >
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="font-medium text-ink-2">{ctrl.label}</span>
                      <span className="font-mono font-semibold text-ink">
                        {fmt(neuronParams[ctrl.key], 2)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={ctrl.min}
                      max={ctrl.max}
                      step={ctrl.step}
                      value={neuronParams[ctrl.key]}
                      onChange={(e) =>
                        setNeuronParams((prev) => ({
                          ...prev,
                          [ctrl.key]: Number(e.target.value),
                        }))
                      }
                      className="h-1.5 w-full accent-ink"
                    />
                  </label>
                ))}
              </div>

              <div className="grid grid-cols-3 gap-2.5">
                <div className="rounded-[10px] border border-line bg-inset/40 px-3 py-2">
                  <div className="text-[11px] text-ink-3">Prediction ŷ = tanh(n)</div>
                  <div className="mt-0.5 font-mono text-[15px] font-semibold text-ink">
                    {fmt(neuronDemo.prediction, 4)}
                  </div>
                </div>
                <div className="rounded-[10px] border border-line bg-inset/40 px-3 py-2">
                  <div className="text-[11px] text-ink-3">MSE Loss = (ŷ − y)²</div>
                  <div className="mt-0.5 font-mono text-[15px] font-semibold text-orange">
                    {fmt(neuronDemo.loss, 4)}
                  </div>
                </div>
                <div className="rounded-[10px] border border-line bg-inset/40 px-3 py-2">
                  <div className="text-[11px] text-ink-3">Gradients ∂L/∂w1, ∂L/∂b</div>
                  <div className="mt-0.5 font-mono text-[12.5px] font-semibold text-accent-ink">
                    {fmt(neuronDemo.grads.w1, 3)}, {fmt(neuronDemo.grads.b, 3)}
                  </div>
                </div>
              </div>

              <ComputationGraphView graph={neuronDemo.graph} />

              <div className="text-[11.5px] leading-relaxed text-ink-2">
                <strong className="font-medium text-ink">Reverse topological trace:</strong>{" "}
                <span className="font-mono text-[11px]">
                  {[...neuronDemo.graph.topoOrder].reverse().join(" → ")}
                </span>
              </div>
            </>
          ) : (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="flex flex-col gap-2.5">
                <div className="flex items-center justify-between text-[12px]">
                  <span className="font-medium text-ink">
                    Epoch {xorSnapshot.epoch} · {xorSnapshot.paramCount} scalar Values
                  </span>
                  <span className="font-mono font-semibold text-orange">
                    Loss {fmt(xorSnapshot.loss, 4)}
                  </span>
                </div>

                {/* Live Loss Sparkline */}
                <div className="rounded-[10px] border border-line bg-inset/50 p-3">
                  <svg viewBox="0 0 320 90" className="h-24 w-full" aria-label="XOR training loss">
                    {(() => {
                      const hist = xorSnapshot.lossHistory;
                      const maxL = Math.max(0.1, ...hist);
                      const pts = hist
                        .map((val, idx) => {
                          const x = hist.length <= 1 ? 160 : (idx / (hist.length - 1)) * 308 + 6;
                          const y = 82 - (val / maxL) * 70;
                          return `${x.toFixed(1)},${y.toFixed(1)}`;
                        })
                        .join(" ");
                      return (
                        <polyline
                          fill="none"
                          stroke="var(--accent)"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          points={pts}
                        />
                      );
                    })()}
                  </svg>
                </div>
              </div>

              <div className="flex flex-col gap-2">
                <div className="text-[12px] font-medium text-ink">XOR Gate Truth Table</div>
                <div className="overflow-hidden rounded-[10px] border border-line">
                  <table className="w-full border-collapse text-left font-mono text-[12px]">
                    <thead className="border-b border-line bg-inset text-[11px] text-ink-2">
                      <tr>
                        <th className="px-3 py-2">Inputs</th>
                        <th className="px-3 py-2">Target</th>
                        <th className="px-3 py-2">MLP Out</th>
                        <th className="px-3 py-2">Match</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-line">
                      {xorSnapshot.predictions.map((row, idx) => {
                        const correct =
                          (row.target > 0 && row.output > 0) || (row.target < 0 && row.output < 0);
                        return (
                          <tr key={idx}>
                            <td className="px-3 py-1.5 text-ink">
                              [{row.inputs[0]}, {row.inputs[1]}]
                            </td>
                            <td className="px-3 py-1.5 text-ink-2">{row.target}</td>
                            <td className="px-3 py-1.5 font-semibold text-ink">
                              {fmt(row.output, 3)}
                            </td>
                            <td className="px-3 py-1.5">
                              <span
                                className={`font-sans text-[11px] font-medium ${
                                  correct ? "text-green" : "text-ink-3"
                                }`}
                              >
                                {correct ? "Correct" : "Learning"}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 2: ATTENTION & TRANSFORMER */}
      {tab === "attention" && (
        <div className="flex flex-col gap-4 p-4">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <label className="flex min-w-[200px] flex-1 flex-col gap-1">
              <span className="text-[11.5px] font-medium text-ink-2">
                Input tokens (space-separated, max 6)
              </span>
              <input
                type="text"
                value={tokenInput}
                onChange={(e) => setTokenInput(e.target.value)}
                className="h-8 rounded-[9px] border border-line-strong/80 bg-inset px-3 font-mono text-[12.5px] text-ink outline-none focus:border-ink"
              />
            </label>

            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => setCausalMask((v) => !v)}
                className={`h-8 rounded-[9px] border px-3 text-[12px] font-medium transition-colors ${
                  causalMask
                    ? "border-ink bg-ink text-surface"
                    : "border-line-strong/80 bg-surface text-ink-2 hover:text-ink"
                }`}
              >
                Causal Mask: {causalMask ? "ON (GPT)" : "OFF (BERT)"}
              </button>

              <select
                aria-label="Key dimension d_k"
                value={dkDim}
                onChange={(e) => setDkDim(Number(e.target.value))}
                className="h-8 rounded-[9px] border border-line-strong/80 bg-surface px-2.5 text-[12px] font-medium text-ink outline-none"
              >
                <option value={2}>d_k = 2</option>
                <option value={3}>d_k = 3</option>
                <option value={4}>d_k = 4</option>
              </select>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            {(
              [
                { id: "weights", label: "Attention Weights (Softmax)" },
                { id: "raw", label: "Scaled Scores (QKᵀ / √d_k)" },
                { id: "qkv", label: "Q, K, V Projections" },
                { id: "norm", label: "Post-Attn & FFN Out" },
              ] as const
            ).map((stage) => (
              <button
                key={stage.id}
                type="button"
                onClick={() => setSelectedAttnMatrix(stage.id)}
                className={`h-7 rounded-[8px] px-2.5 text-[11.5px] font-medium transition-colors ${
                  selectedAttnMatrix === stage.id
                    ? "bg-hover-2 text-ink"
                    : "text-ink-2 hover:bg-hover hover:text-ink"
                }`}
              >
                {stage.label}
              </button>
            ))}
          </div>

          {selectedAttnMatrix === "weights" && (
            <div className="flex flex-col gap-3">
              <MatrixGrid
                title="Attention Probability Matrix · softmax((Q·Kᵀ) / √d_k)"
                subtitle={`${tokens.length}×${tokens.length} · scale 1/√${dkDim} = ${fmt(
                  transformerTrace.attention.scaleFactor,
                  3
                )}`}
                matrix={transformerTrace.attention.weights}
                rowLabels={tokens}
                colLabels={tokens}
                activeCell={hoveredAttnCell}
                highlightRow={hoveredAttnCell.row}
                highlightCol={hoveredAttnCell.col}
                onHoverCell={(row, col) => setHoveredAttnCell({ row, col })}
                heatMode
              />
              <div className="rounded-[9px] border border-line bg-inset/50 px-3 py-2 font-mono text-[11.5px] text-ink-2">
                Query <strong className="text-ink">&ldquo;{tokens[hoveredAttnCell.row] ?? tokens[0]}&rdquo;</strong>{" "}
                attends to Key{" "}
                <strong className="text-ink">&ldquo;{tokens[hoveredAttnCell.col] ?? tokens[0]}&rdquo;</strong>{" "}
                with weight{" "}
                <strong className="text-accent-ink">
                  {fmt(
                    (transformerTrace.attention.weights[hoveredAttnCell.row]?.[
                      hoveredAttnCell.col
                    ] ?? 0) * 100,
                    1
                  )}
                  %
                </strong>{" "}
                (raw score:{" "}
                {fmt(
                  transformerTrace.attention.maskedScores[hoveredAttnCell.row]?.[
                    hoveredAttnCell.col
                  ] ?? 0,
                  3
                )}
                )
              </div>
            </div>
          )}

          {selectedAttnMatrix === "raw" && (
            <MatrixGrid
              title="Masked Scaled Dot-Product Scores"
              subtitle={`(Q · Kᵀ) / √${dkDim}`}
              matrix={transformerTrace.attention.maskedScores}
              rowLabels={tokens}
              colLabels={tokens}
            />
          )}

          {selectedAttnMatrix === "qkv" && (
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
              <MatrixGrid
                title="Q = X · W_Q"
                subtitle={`${tokens.length}×${dkDim}`}
                matrix={transformerTrace.attention.Q}
                rowLabels={tokens}
              />
              <MatrixGrid
                title="K = X · W_K"
                subtitle={`${tokens.length}×${dkDim}`}
                matrix={transformerTrace.attention.K}
                rowLabels={tokens}
              />
              <MatrixGrid
                title="V = X · W_V"
                subtitle={`${tokens.length}×${transformerTrace.dModel}`}
                matrix={transformerTrace.attention.V}
                rowLabels={tokens}
              />
            </div>
          )}

          {selectedAttnMatrix === "norm" && (
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
              <MatrixGrid
                title="LayerNorm(X + Attn(Q,K,V))"
                subtitle={`${tokens.length}×${transformerTrace.dModel}`}
                matrix={transformerTrace.postAttnNorm}
                rowLabels={tokens}
              />
              <MatrixGrid
                title="Transformer Block Output (+ GELU FFN)"
                subtitle={`${tokens.length}×${transformerTrace.dModel}`}
                matrix={transformerTrace.ffnOut}
                rowLabels={tokens}
              />
            </div>
          )}
        </div>
      )}

      {/* TAB 3: MATRIX MULTIPLICATION */}
      {tab === "matmul" && (
        <div className="flex flex-col gap-4 p-4">
          <div className="flex flex-wrap items-center justify-between gap-2 text-[12px] text-ink-2">
            <span>
              Hover or click any cell in <strong className="text-ink">C = A × B</strong> to inspect its dot product.
            </span>
            <button
              type="button"
              onClick={() => {
                setMatA([
                  [1, 2, -1],
                  [0, 3, 2],
                ]);
                setMatB([
                  [2, 1],
                  [-1, 4],
                  [3, 0],
                ]);
              }}
              className="flex h-7 items-center gap-1 rounded-[8px] border border-line-strong/80 bg-surface px-2.5 text-[11.5px] font-medium text-ink-2 hover:text-ink"
            >
              <RotateCcw className="size-3" />
              <span>Reset matrices</span>
            </button>
          </div>

          <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
            {/* Editable Matrix A */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-baseline justify-between">
                <span className="text-[12px] font-semibold text-ink">Matrix A</span>
                <span className="font-mono text-[11px] text-ink-3">2×3</span>
              </div>
              <div className="grid grid-cols-3 gap-1.5 rounded-[10px] border border-line bg-inset/50 p-2">
                {matA.map((row, r) =>
                  row.map((val, c) => (
                    <input
                      key={`a-${r}-${c}`}
                      type="number"
                      step="0.5"
                      value={val}
                      onChange={(e) => {
                        const num = Number(e.target.value) || 0;
                        setMatA((prev) =>
                          prev.map((pr, ri) =>
                            ri === r ? pr.map((pv, ci) => (ci === c ? num : pv)) : pr
                          )
                        );
                      }}
                      className={`h-8 w-full rounded-[6px] border px-1.5 text-center font-mono text-[12px] outline-none ${
                        selectedMatCell.row === r
                          ? "border-accent bg-surface font-semibold text-ink"
                          : "border-line bg-surface text-ink-2"
                      }`}
                    />
                  ))
                )}
              </div>
            </div>

            {/* Editable Matrix B */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-baseline justify-between">
                <span className="text-[12px] font-semibold text-ink">Matrix B</span>
                <span className="font-mono text-[11px] text-ink-3">3×2</span>
              </div>
              <div className="grid grid-cols-2 gap-1.5 rounded-[10px] border border-line bg-inset/50 p-2">
                {matB.map((row, r) =>
                  row.map((val, c) => (
                    <input
                      key={`b-${r}-${c}`}
                      type="number"
                      step="0.5"
                      value={val}
                      onChange={(e) => {
                        const num = Number(e.target.value) || 0;
                        setMatB((prev) =>
                          prev.map((pr, ri) =>
                            ri === r ? pr.map((pv, ci) => (ci === c ? num : pv)) : pr
                          )
                        );
                      }}
                      className={`h-8 w-full rounded-[6px] border px-1.5 text-center font-mono text-[12px] outline-none ${
                        selectedMatCell.col === c
                          ? "border-accent bg-surface font-semibold text-ink"
                          : "border-line bg-surface text-ink-2"
                      }`}
                    />
                  ))
                )}
              </div>
            </div>

            {/* Result Matrix C */}
            <MatrixGrid
              title="Output C = A × B"
              subtitle="2×2"
              matrix={matMulTrace.result}
              activeCell={selectedMatCell}
              highlightRow={selectedMatCell.row}
              highlightCol={selectedMatCell.col}
              onHoverCell={(row, col) => setSelectedMatCell({ row, col })}
            />
          </div>

          <div className="rounded-[10px] border border-line bg-inset/60 px-3.5 py-2.5 font-mono text-[12px] text-ink">
            <span className="text-ink-3">
              C[{selectedMatCell.row},{selectedMatCell.col}] ={" "}
            </span>
            {activeCellDetail.terms.map((t, idx) => (
              <span key={t.k}>
                {idx > 0 ? " + " : ""}({t.aVal} × {t.bVal})
              </span>
            ))}
            <span className="text-ink-3"> = </span>
            <span className="font-semibold text-accent-ink">{fmt(activeCellDetail.sum, 2)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default memo(NeuralEngineLab);
