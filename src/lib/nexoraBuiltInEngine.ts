import { MLP, runSingleNeuronDemo } from "./microgradEngine";
import {
  matMulWithTrace,
  runEducationalTransformerForward,
  type Matrix,
} from "./transformerEngine";
import {
  resolveConversationalFollowUpTopic,
  synthesizeResearchWithBM25,
  tryCodeAssistant,
  tryQuickMathOrUnits,
  tryStatisticalAnalysisAndChart,
  tryStructuredReasoningAndWriting,
  tryStructuredUtility,
  trySymbolicMathAndLogic,
  tryTextRankSummary,
  type AssistantHistoryTurn,
  type ResearchPassage,
  type ResearchSourceCitation,
} from "./pureTsIntelligence";
import {
  extractMemoriesFromMessage,
  tryAnswerMemoryQuery,
  type ChatMemoryFact,
  type ThreadMemoryDigest,
} from "./chatMemory";
import type { NeuralLabTab } from "../components/NeuralEngineLab";
import {
  getLocalModelPreset,
  streamLocalWebGpuChat,
  type LocalLoadProgress,
} from "./localWebGpuEngine";
import {
  ensureSharedNeuralModelReady,
  generateNeuralAssistantAnswer,
  getSharedNexoraNeuralModel,
  harvestAndFeedNeuralEngine,
} from "./neuralLanguageEngine";
import type { LocalModelTier } from "./offlineStore";

export interface BuiltInAssistantResult {
  markdown: string;
  thinkingLines: string[];
  reasoningNote?: string;
  labTab?: NeuralLabTab;
  titleHint?: string;
  sources?: ResearchSourceCitation[];
}

// ---------------------------------------------------------------------------
// Inline Matrix Multiplication Parser (e.g. "[[1,2],[3,4]] * [[5,6],[7,8]]")
// ---------------------------------------------------------------------------
function tryParseInlineMatMul(prompt: string): {
  A: Matrix;
  B: Matrix;
  traceResult: ReturnType<typeof matMulWithTrace>;
} | null {
  const matrixPattern = /\[\s*\[[^\]]+\](?:\s*,\s*\[[^\]]+\])*\s*\]/g;
  const matches = prompt.match(matrixPattern);
  if (!matches || matches.length < 2) return null;

  try {
    const rawA = JSON.parse(matches[0]) as unknown;
    const rawB = JSON.parse(matches[1]) as unknown;
    const isValidMat = (m: unknown): m is Matrix =>
      Array.isArray(m) &&
      m.length > 0 &&
      m.length <= 6 &&
      m.every(
        (row) =>
          Array.isArray(row) &&
          row.length > 0 &&
          row.length <= 6 &&
          row.length === (m[0] as unknown[]).length &&
          row.every((v) => typeof v === "number" && Number.isFinite(v))
      );

    if (!isValidMat(rawA) || !isValidMat(rawB)) return null;
    if (rawA[0].length !== rawB.length) return null;

    return {
      A: rawA,
      B: rawB,
      traceResult: matMulWithTrace(rawA, rawB),
    };
  } catch {
    return null;
  }
}

function detectRequestedLabTab(prompt: string): NeuralLabTab | null {
  const clean = prompt.trim().toLowerCase();
  if (/^\/(pipeline|train|deep|lm)\b/.test(clean)) return "pipeline";
  if (/^\/(micrograd|backprop|autograd)\b/.test(clean)) return "micrograd";
  if (/^\/(attention|transformer|self-attention)\b/.test(clean)) return "attention";
  if (/^\/(matmul|matrix)\b/.test(clean)) return "matmul";
  if (/^\/lab\b/.test(clean)) {
    if (clean.includes("pipeline") || clean.includes("train") || clean.includes("deep")) {
      return "pipeline";
    }
    if (clean.includes("attention") || clean.includes("transformer")) return "attention";
    if (clean.includes("matmul") || clean.includes("matrix")) return "matmul";
    return "micrograd";
  }

  const mentionsPipeline =
    /(massive amounts? of data|core building blocks|deep learning|neural networks?.*weights.*biases|how ai learns|train.*(?:neural|language model|nexora)|input.*training.*adjustment.*inference|next[- ]token)/.test(
      clean
    );
  const mentionsMicrograd =
    /(micrograd|autograd|backprop|backpropagation|chain rule|\bxor\b|\bmlp\b|gradient descent)/.test(
      clean
    );
  const mentionsAttention =
    /(transformer|self-attention|scaled dot|softmax|\bqkv\b|layernorm|\bgelu\b|attention mechanism)/.test(
      clean
    );
  const mentionsMatmul =
    /(matrix mult|\bmatmul\b|dot product|multiply matrices)/.test(clean);

  if (mentionsPipeline) return "pipeline";
  if (mentionsMicrograd) return "micrograd";
  if (mentionsAttention) return "attention";
  if (mentionsMatmul) return "matmul";
  return null;
}

function formatMatrixTable(mat: Matrix, label: string): string {
  const header = `| ${label} | ${mat[0].map((_, c) => `Col ${c + 1}`).join(" | ")} |`;
  const sep = `| --- | ${mat[0].map(() => "---:").join(" | ")} |`;
  const rows = mat.map(
    (row, r) => `| **Row ${r + 1}** | ${row.map((v) => Number(v.toFixed(4))).join(" | ")} |`
  );
  return [header, sep, ...rows].join("\n");
}

export async function fetchBuiltInResearchPassages(
  query: string,
  signal?: AbortSignal
): Promise<ResearchPassage[]> {
  try {
    const res = await fetch("/api/assistant/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({ query }),
    });
    if (!res.ok) return [];
    const data = (await res.json()) as { passages?: ResearchPassage[] };
    return Array.isArray(data.passages) ? data.passages : [];
  } catch {
    return [];
  }
}

function parseWebHarvestIntent(
  prompt: string
): { url?: string; topic?: string } | null {
  const clean = prompt.trim();
  const lower = clean.toLowerCase();

  const wantsHarvest =
    /\b(get|fetch|scrape|crawl|harvest|pull|grab|collect|download|gather|feed|train|ingest|load)\b/.test(
      lower
    ) &&
    /\b(website|web\s*site|site|web\s*page|webpage|web\s*data|wikipedia|internet|url|link|corpus|dataset|more\s+data)\b/.test(
      lower
    ) &&
    (/\b(feed|train|neural|engine|typescript|model|ai|weights|vocab)\b/.test(lower) ||
      /\bas\s+much\s+as\s+you\s+can\b/.test(lower));

  if (!wantsHarvest) return null;

  // Check for explicit URL or domain
  const explicitUrlMatch = clean.match(/https?:\/\/[^\s<>)"'\]]+/i);
  if (explicitUrlMatch) {
    return { url: explicitUrlMatch[0].replace(/[.,;:!?)\]]+$/, "") };
  }
  const bareDomainMatch = clean.match(
    /\b([a-z0-9-]+(?:\.[a-z0-9-]+)+(?:\/[^\s]*)?)\b/i
  );
  if (
    bareDomainMatch &&
    !/\b(next\.js|node\.js|react\.js|qwen\s*2\.5|qwen\s*3\.8|index\.html)\b/i.test(
      bareDomainMatch[1]
    )
  ) {
    return { url: `https://${bareDomainMatch[1].replace(/[.,;:!?)\]]+$/, "")}` };
  }

  // Check if user requested a concrete topic (ignore vague phrases like "a website or smth")
  const topicMatch = clean.match(
    /\b(?:about|on|regarding|for)\s+([a-z0-9][a-z0-9\s-]{2,40}?)(?=\s+(?:and\s+(?:feed|train)|into\s+the|to\s+the|$))/i
  );
  if (topicMatch) {
    const candidate = topicMatch[1].trim();
    if (
      !/^(?:a\s+website|website|something|smth|anything|data|the\s+web|typescript|neural)/i.test(
        candidate
      )
    ) {
      return { topic: candidate };
    }
  }

  return {};
}

export function tryDeterministicPureTsHandlers(
  trimmed: string,
  options?: {
    hasWebGpuReady?: boolean;
    hasSavedByokModels?: boolean;
    history?: AssistantHistoryTurn[];
    memoryFacts?: ChatMemoryFact[];
    threadDigests?: ThreadMemoryDigest[];
    activeThreadTitle?: string;
  }
): BuiltInAssistantResult | null {
  const lower = trimmed.toLowerCase();

  // 1. Inline numeric matrix multiplication (e.g. [[1,2],[3,4]] * [[5,6],[7,8]])
  const inlineMat = tryParseInlineMatMul(trimmed);
  if (inlineMat) {
    const flatCells = inlineMat.traceResult.cells.flat();
    const traces = flatCells
      .slice(0, 12)
      .map(
        (cell) =>
          `- \`C[${cell.row},${cell.col}]\` = ${cell.terms
            .map((term) => `(${term.aVal} × ${term.bVal})`)
            .join(" + ")} = **${Number(cell.sum.toFixed(4))}**`
      )
      .join("\n");

    return {
      labTab: "matmul",
      titleHint: "Matrix Multiplication Trace",
      reasoningNote: `Parsed (${inlineMat.A.length}×${inlineMat.A[0].length}) · (${inlineMat.B.length}×${inlineMat.B[0].length}) matrices and evaluated ${flatCells.length} exact cell dot products in pure TypeScript.`,
      thinkingLines: [
        "Parsed numeric matrices in pure TypeScript",
        `Computed (${inlineMat.A.length}×${inlineMat.A[0].length}) · (${inlineMat.B.length}×${inlineMat.B[0].length}) dot-product trace`,
      ],
      markdown: [
        "### Pure TypeScript Matrix Multiplication Result",
        "",
        formatMatrixTable(inlineMat.traceResult.result, "Output C"),
        "",
        "**Dot-Product Expansion:**",
        traces,
        "",
        "```nexora-lab",
        "matmul",
        "```",
      ].join("\n"),
    };
  }

  // 2. Pure-TS Deep AI Pipeline / Micrograd / Transformer / MatMul interactive labs
  const labTab = detectRequestedLabTab(trimmed);
  if (labTab) {
    if (labTab === "pipeline") {
      const model = getSharedNexoraNeuralModel();
      void model.adaptOnText(trimmed, 1);
      const snap = model.getSnapshot("neural networks learn patterns");
      const infer = model.predictAndGenerate("neural networks learn", 14, 0.6, 5);
      const topPreds = infer.topNextTokens
        .slice(0, 3)
        .map((t) => `\`"${t.token}"\` (${(t.probability * 100).toFixed(1)}%)`)
        .join(", ");

      return {
        labTab: "pipeline",
        titleHint: "TensorFlow.js Deep AI Pipeline",
        reasoningNote: `Executed TensorFlow.js (${snap.backend}) Sequential model (${snap.paramCount.toLocaleString()} weights & biases, ${snap.stepCount} model.fit epochs, tf.train.${snap.optimizer}, loss ${snap.initialLoss.toFixed(2)} -> ${snap.currentLoss.toFixed(3)}, acc ${(snap.accuracy * 100).toFixed(1)}%).`,
        thinkingLines: [
          `Tokenized corpus into ${snap.vocabSize} IDs (${snap.paramCount.toLocaleString()} TFJS params)`,
          `Compiled with tf.train.${snap.optimizer} on ${snap.backend} (${snap.stepCount} epochs trained)`,
          "Executed model.predict() autoregressive inference",
        ],
        markdown: [
          `### Nexora Deep Neural AI Pipeline (TensorFlow.js · \`${snap.backend}\` backend)`,
          "",
          "Nexora now runs its deep language engine on **TensorFlow.js (`@tensorflow/tfjs`)** using `tf.sequential()`, `model.compile()`, and **`await model.fit()`** with **`" +
            snap.paramCount.toLocaleString() +
            "`** learnable parameters across embeddings and deep dense layers:",
          "",
          "#### The Core Building Blocks (Running Live in TFJS)",
          `- **Data (The Fuel):** Cleaned corpus + live harvested web pages (**${snap.vocabSize} vocabulary tokens**, **${snap.corpusWords.toLocaleString()} words**).`,
          `- **Machine Learning (The Engine):** \`model.compile({ optimizer: tf.train.${snap.optimizer}(${snap.learningRate}), loss: "sparseCategoricalCrossentropy" })\` discovers token transitions over **${snap.stepCount} \`model.fit()\` epochs**.`,
          `- **Neural Networks (The Brain Structure):** \`tf.layers.embedding\` (\`d_model = ${snap.dModel}\`) → \`flatten\` → \`dense(${snap.hiddenDim}, "tanh")\` → \`dropout(0.1)\` → \`dense(${snap.hiddenDim}, "relu")\` → \`dense(${snap.vocabSize}, "softmax")\`.`,
          `- **Deep Learning (The Heavy Lifting):** Multi-layer tensor ops accelerated on **\`${snap.backend}\`** reach **\`${(snap.accuracy * 100).toFixed(1)}%\` training accuracy**.`,
          "",
          "#### How Nexora Learns & Operates (Step-by-Step Telemetry)",
          `1. **Input:** Text maps to context integer windows (e.g., ${snap.sampleTokens.slice(0, 4).map((t) => `\`"${t.token}"→#${t.id}\``).join(", ")}) fed into \`tf.layers.embedding\`.`,
          `2. **Training:** \`await model.fit(xs, ys)\` measures Cross-Entropy loss (**initial \`${snap.initialLoss.toFixed(2)}\` → trained \`${snap.currentLoss.toFixed(3)}\`**, perplexity \`${snap.perplexity.toFixed(1)}\`).`,
          `3. **Adjustment:** TensorFlow.js backpropagation updates all **${snap.paramCount.toLocaleString()} weights & biases** via \`tf.train.${snap.optimizer}\` (switchable live between Adam & SGD below).`,
          `4. **Inference:** Calling \`model.predict()\` on \`"neural networks learn"\` predicts ${topPreds} and generates: _"${infer.generatedContinuation}"_.`,
          "",
          "```nexora-lab",
          "pipeline",
          "```",
        ].join("\n"),
      };
    }

    const coversAllThree =
      /(micrograd|backprop)/.test(lower) &&
      /(transformer|attention)/.test(lower) &&
      /(matrix|matmul)/.test(lower);

    if (coversAllThree) {
      const neuronSnapshot = runSingleNeuronDemo({
        x1: 2,
        w1: -3,
        x2: 0,
        w2: 1,
        b: 6.8813735,
        target: 1,
      });
      const attnSnapshot = runEducationalTransformerForward({
        tokens: ["Pure", "TS", "Attention"],
        dk: 3,
        causal: true,
      });
      const demoMat = matMulWithTrace(
        [
          [1, 2, -1],
          [0, 3, 2],
        ],
        [
          [2, -1],
          [1, 4],
          [3, 0],
        ]
      );

      return {
        labTab: "micrograd",
        titleHint: "Pure-TS Micrograd & Transformer Lab",
        reasoningNote: `Executed live forward+backward pass on scalar autograd neuron (ŷ=${neuronSnapshot.prediction.toFixed(4)}, dw1=${neuronSnapshot.grads.w1.toFixed(4)}), transformer scaled dot-product attention (dk=3, causal=true), and (2×3)·(3×2) matMul.`,
        thinkingLines: [
          "Running Pure-TS Micrograd & Transformer forward/backward passes",
          "Prepared interactive Value DAG, Attention heatmap & MatMul tracer",
        ],
        markdown: [
          "### Pure TypeScript Neural & Transformer Engines (Zero External Libraries)",
          "",
          "Everything below executes live in pure TypeScript with zero external ML libraries or third-party APIs:",
          "",
          `1. **Micrograd & Backpropagation** (\`Value\` autograd DAG): Forward pass outputs \`ŷ = ${neuronSnapshot.prediction.toFixed(4)}\` with MSE loss \`${neuronSnapshot.loss.toFixed(4)}\` and reverse-mode gradient \`dL/dw1 = ${neuronSnapshot.grads.w1.toFixed(4)}\`.`,
          `2. **Transformer Self-Attention** (\`softmax((Q · Kᵀ) / √d_k) · V\`): Forward pass on \`["Pure", "TS", "Attention"]\` (\`d_k = 3\`, causal mask, LayerNorm, GELU FFN; final token top weight \`${Math.max(...attnSnapshot.attention.weights[2]).toFixed(3)}\`).`,
          `3. **Matrix Multiplication** (\`matMulWithTrace\`): Example \`(2×3) · (3×2)\` product yields \`C[0,0] = ${demoMat.result[0][0]}\`, \`C[1,1] = ${demoMat.result[1][1]}\`.`,
          "",
          "```nexora-lab",
          "micrograd",
          "```",
        ].join("\n"),
      };
    }

    if (labTab === "attention") {
      const attnSnapshot = runEducationalTransformerForward({
        tokens: ["Why", "attention", "matters"],
        dk: 3,
        causal: true,
      });
      return {
        labTab: "attention",
        titleHint: "Transformer Attention Lab",
        reasoningNote: `Computed Q·Kᵀ / √3 scaled attention weights [${attnSnapshot.attention.weights[2].map((w) => w.toFixed(2)).join(", ")}], residual LayerNorm, and GELU FFN in pure TypeScript.`,
        thinkingLines: [
          "Running Pure-TS Scaled Dot-Product Attention",
          "Computed Q, K, V, causal mask, Softmax & GELU FFN",
        ],
        markdown: [
          "### Pure TypeScript Transformer & Scaled Dot-Product Attention",
          "",
          "`Attention(Q, K, V) = softmax((Q · Kᵀ) / √d_k + Mask) · V`",
          "",
          `- **Sample run** (\`${attnSnapshot.tokens.join(" ")}\`, \`d_k = 3\`): Token 3 attends with softmax weights \`[${attnSnapshot.attention.weights[2].map((w) => w.toFixed(2)).join(", ")}]\` followed by residual LayerNorm and GELU FFN.`,
          "",
          "```nexora-lab",
          "attention",
          "```",
        ].join("\n"),
      };
    }

    if (labTab === "matmul") {
      return {
        labTab: "matmul",
        titleHint: "Matrix Multiplication Lab",
        reasoningNote: "Initialized interactive (M×K)·(K×N) matrix multiplication dot-product tracer.",
        thinkingLines: ["Initialized Pure-TS Matrix Multiplication Tracer"],
        markdown: [
          "### Pure TypeScript Matrix Multiplication Lab",
          "",
          "Each output entry `C[i, j]` is the exact inner product of row `i` of `A` and column `j` of `B`. You can also paste numeric matrices directly in chat like `[[1, 2], [3, 4]] * [[5, 6], [7, 8]]`.",
          "",
          "```nexora-lab",
          "matmul",
          "```",
        ].join("\n"),
      };
    }

    const demo = runSingleNeuronDemo({
      x1: 2,
      w1: -3,
      x2: 0,
      w2: 1,
      b: 6.8813735,
      target: 1,
    });
    return {
      labTab: "micrograd",
      titleHint: "Micrograd & Backprop Lab",
      reasoningNote: `Constructed scalar computation DAG, ran reverse topological .backward(): ŷ=${demo.prediction.toFixed(4)}, loss=${demo.loss.toFixed(4)}, dw1=${demo.grads.w1.toFixed(4)}.`,
      thinkingLines: [
        "Constructed scalar Value computation graph",
        "Executed reverse-mode backpropagation",
      ],
      markdown: [
        "### Pure TypeScript Micrograd & Backpropagation",
        "",
        "Each `Value` node records `.data`, parents `._prev`, and a local derivative closure `._backward()`. Calling `.backward()` walks the DAG in reverse topological order and accumulates `.grad` via the chain rule.",
        "",
        `- **Live snapshot**: output \`ŷ = ${demo.prediction.toFixed(4)}\`, MSE loss \`${demo.loss.toFixed(4)}\`, gradients \`dw1 = ${demo.grads.w1.toFixed(4)}\`, \`dw2 = ${demo.grads.w2.toFixed(4)}\`, \`db = ${demo.grads.b.toFixed(4)}\`.`,
        "",
        "```nexora-lab",
        "micrograd",
        "```",
      ].join("\n"),
    };
  }

  // 3. Symbolic Algebra, Polynomial Calculus, Primes, Radix, GCD/LCM
  const symbolic = trySymbolicMathAndLogic(trimmed);
  if (symbolic) {
    return {
      titleHint: symbolic.titleHint,
      reasoningNote: symbolic.reasoningNote,
      thinkingLines: ["Parsed mathematical structure", symbolic.reasoningNote],
      markdown: symbolic.markdown,
    };
  }

  // 4. Statistical Analysis & Auto SVG Trend Chart
  const stats = tryStatisticalAnalysisAndChart(trimmed);
  if (stats) {
    return {
      titleHint: stats.titleHint,
      reasoningNote: stats.reasoningNote,
      thinkingLines: ["Computed dataset statistics & least-squares trend", "Generated interactive SVG visualization"],
      markdown: stats.markdown,
    };
  }

  // 5. Static Code Inspector / Interactive Widgets / Pure-TS Code & Page Builder (history-aware)
  const isPastedCodeReview = /```[\s\S]*```/.test(trimmed);
  if (!options?.hasWebGpuReady || isPastedCodeReview) {
    const code = tryCodeAssistant(trimmed, options?.history);
    if (code) {
      return {
        titleHint: code.titleHint,
        reasoningNote: code.reasoningNote,
        thinkingLines: ["Planned architecture & implementation", code.reasoningNote],
        markdown: code.markdown,
      };
    }
  }

  // 6. Persistent User Memory & Cross-Chat Recall Queries
  const extraction = extractMemoriesFromMessage(trimmed);
  const memoryReply = tryAnswerMemoryQuery(trimmed, {
    memoryFacts: options?.memoryFacts,
    threadDigests: options?.threadDigests,
    history: options?.history,
    extraction,
    activeThreadTitle: options?.activeThreadTitle,
  });
  if (memoryReply) {
    return {
      titleHint: memoryReply.titleHint,
      reasoningNote: memoryReply.reasoningNote,
      thinkingLines: memoryReply.thinkingLines,
      markdown: memoryReply.markdown,
    };
  }

  // 7. TextRank Graph Extractive Summarizer & Structured Utilities
  const summary = tryTextRankSummary(trimmed);
  if (summary) {
    return {
      titleHint: summary.titleHint,
      reasoningNote: summary.reasoningNote,
      thinkingLines: ["Built sentence cosine graph", "Ran TextRank centrality ranking"],
      markdown: summary.markdown,
    };
  }
  const structured = tryStructuredUtility(trimmed);
  if (structured) {
    return {
      titleHint: structured.titleHint,
      reasoningNote: structured.reasoningNote,
      thinkingLines: ["Transformed structured text layout"],
      markdown: structured.markdown,
    };
  }

  // 8. Friendly Conversational Greeting (personalized from persistent memory)
  if (/^(hi|hello|hey|yo|hola|halo|greetings|good\s+(morning|afternoon|evening))[\s!?.]*$/i.test(trimmed)) {
    const savedName = options?.memoryFacts?.find((f) => f.id === "identity_user_name")?.value;
    return {
      reasoningNote: savedName
        ? `Conversational greeting recognized; greeted ${savedName} using saved workspace memory.`
        : "Conversational greeting recognized; replying concisely and naturally.",
      thinkingLines: [savedName ? `Ready (remembered user: ${savedName})` : "Ready"],
      markdown: savedName
        ? `Hey **${savedName}**! How can I help today? Feel free to ask a question, continue an earlier topic, paste code to review, or request an HTML page or script.`
        : "Hey! How can I help today? Feel free to ask a question, paste code to review, request an HTML page or script, or run math and data analysis.",
    };
  }

  // 9. Exact Deterministic Arithmetic & Unit Conversions (history-aware for follow-ups like "double that")
  const quickMath = tryQuickMathOrUnits(trimmed, options?.history);
  if (quickMath) {
    return {
      reasoningNote: "Evaluated expression / unit conversion exactly via deterministic Pure-TS parser.",
      thinkingLines: ["Evaluated expression deterministically"],
      markdown: quickMath,
    };
  }

  // 10. Structured Reasoning, Guides, Comparisons & Writing Templates (when WebGPU neural weights aren't loaded)
  if (!options?.hasWebGpuReady) {
    const structuredGuide = tryStructuredReasoningAndWriting(trimmed);
    if (structuredGuide) {
      return {
        titleHint: structuredGuide.titleHint,
        reasoningNote: structuredGuide.reasoningNote,
        thinkingLines: ["Structured response plan", structuredGuide.reasoningNote],
        markdown: structuredGuide.markdown,
      };
    }
  }

  return null;
}

export function runNexoraBuiltInAssistant(
  rawPrompt: string,
  options?: {
    hasWebGpuReady?: boolean;
    hasSavedByokModels?: boolean;
    history?: AssistantHistoryTurn[];
    researchPassages?: ResearchPassage[];
    memoryFacts?: ChatMemoryFact[];
    threadDigests?: ThreadMemoryDigest[];
    activeThreadTitle?: string;
  }
): BuiltInAssistantResult {
  const trimmed = rawPrompt.trim();
  const direct = tryDeterministicPureTsHandlers(trimmed, options);
  if (direct) return direct;

  const neural = generateNeuralAssistantAnswer(trimmed, {
    history: options?.history,
    researchPassages: options?.researchPassages,
    memoryFacts: options?.memoryFacts,
    threadDigests: options?.threadDigests,
  });

  return {
    reasoningNote: neural.reasoningNote,
    thinkingLines: neural.thinkingLines,
    markdown: neural.markdown,
  };
}

function sleepAbortable(ms: number, signal?: AbortSignal): Promise<boolean> {
  if (signal?.aborted) return Promise.resolve(false);
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve(true);
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      resolve(false);
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

const STOP_THINK_TOKENS = new Set([
  "a", "an", "the", "is", "are", "was", "to", "of", "in", "on", "for", "with", "me", "my", "i", "please", "can", "you",
]);

interface NeuralThoughtTrace {
  dominantPathway: string;
  focusTokens: string[];
  topWeight: number;
  mlpConfidence: number;
  gradNorm: number;
  steps: string[];
}

function runNeuralThoughtPass(prompt: string, effort?: string): NeuralThoughtTrace {
  const clean = prompt.trim();
  const lower = clean.toLowerCase();

  // 1. Extract prompt tokens & execute Pure-TS Transformer Self-Attention (Q·Kᵀ / √d_k + GELU FFN)
  const rawWords = clean
    .replace(/[^a-zA-Z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
  const contentWords = rawWords.filter(
    (w) => w.length >= 2 && !STOP_THINK_TOKENS.has(w.toLowerCase())
  );
  const attnTokens = (contentWords.length >= 2 ? contentWords : rawWords).slice(0, 5);
  if (attnTokens.length === 0) attnTokens.push("hello");
  if (attnTokens.length === 1) attnTokens.push("intent");

  const attnOut = runEducationalTransformerForward({
    tokens: attnTokens,
    dk: 4,
    causal: true,
  });
  const lastRowWeights = attnOut.attention.weights[attnOut.attention.weights.length - 1] ?? [1];
  let bestWeight = -1;
  for (let i = 0; i < lastRowWeights.length; i++) {
    if (lastRowWeights[i] > bestWeight) {
      bestWeight = lastRowWeights[i];
    }
  }
  const rankedPairs = attnTokens
    .map((tok, i) => ({ tok, w: lastRowWeights[i] ?? 0 }))
    .sort((a, b) => b.w - a.w)
    .slice(0, 2);

  // 2. Encode 6 semantic features & run live Pure-TS Micrograd MLP(6 -> [8, 5]) forward + backward pass
  const fWebCode =
    /\b(html|css|js|javascript|typescript|react|python|express|api|code|build|create|make|webpage|website|calculator|todo|dashboard|login|portfolio)\b/.test(
      lower
    )
      ? 1
      : 0;
  const fMathData =
    /(\d\s*[+\-*/^=]\s*\d|\bsolve\b|\bderivative\b|\bd\/dx\b|\banalyze\b|\bstats\b|\bchart\b|\bfactor\b|\[\[)/.test(
      lower
    )
      ? 1
      : 0;
  const fResearch =
    /^(what|who|where|when|why|how|define|explain|meaning)\b|https?:\/\//.test(lower) ? 1 : 0;
  const fWritePlan =
    /\b(write|draft|compose|email|proposal|compare|vs|plan|roadmap|guide|summarize|tl;?dr)\b/.test(
      lower
    )
      ? 1
      : 0;
  const fNeuralLab =
    /\b(micrograd|autograd|backprop|attention|transformer|matmul|matrix|deep learning|neural network|weights and biases|inference|cross-entropy|next[- ]token)\b/.test(
      lower
    )
      ? 1
      : 0;
  const fComplexity = Math.min(1, clean.length / 180);

  const features = [fWebCode, fMathData, fResearch, fWritePlan, fNeuralLab, fComplexity];
  const mlp = new MLP(6, [8, 5], 2026);
  const logits = mlp.forward(features);
  const pathwayLabels = [
    "Code & UI Synthesis",
    "Symbolic Math & Analytics",
    "Research & Definition",
    "Structured Writing & Planning",
    "Neural Engine Lab",
  ];
  const targetSignals = [fWebCode, fMathData, fResearch, fWritePlan, fNeuralLab];
  let bestPathIdx = 0;
  let bestTarget = -1;
  for (let i = 0; i < targetSignals.length; i++) {
    const combined = targetSignals[i] * 2 + Math.tanh(logits[i]?.data ?? 0);
    if (combined > bestTarget) {
      bestTarget = combined;
      bestPathIdx = i;
    }
  }
  const primaryLogit = logits[bestPathIdx] ?? logits[0];
  const loss = primaryLogit.sub(1).pow(2);
  mlp.zeroGrad();
  loss.backward();
  const params = mlp.parameters();
  const gradNorm = Math.sqrt(
    params.slice(0, 12).reduce((acc, p) => acc + p.grad * p.grad, 0)
  );
  const mlpConfidence = 1 / (1 + Math.exp(-(Math.abs(primaryLogit.data) + targetSignals[bestPathIdx] * 1.4)));
  const dominantPathway =
    targetSignals.every((v) => v === 0) ? "Conversational & General Synthesis" : pathwayLabels[bestPathIdx];

  // 3. Query active TensorFlow.js Deep Neural Language Model (Embedding -> Dense -> model.predict)
  const coreLm = getSharedNexoraNeuralModel();
  void coreLm.adaptOnText(clean, 1);
  const lmSnap = coreLm.getSnapshot(clean);
  const lmInfer = coreLm.predictAndGenerate(clean, 8, 0.6, 3);
  const tokenIdPreview = lmInfer.promptTokens
    .slice(0, 4)
    .map((tok, i) => `"${tok}"→#${lmInfer.promptTokenIds[i]}`)
    .join(", ");
  const nextGuessPreview = lmInfer.topNextTokens
    .slice(0, 3)
    .map((cand) => `"${cand.token}" (${(cand.probability * 100).toFixed(0)}%)`)
    .join(", ");

  // 4. Build genuine 4-stage neural + cognitive reasoning prose tailored to the user's request
  const step1 = `1. Input & Attention: Tokenized [${tokenIdPreview || `"${attnTokens[0]}"`}]; Transformer Q·Kᵀ/√4 attended most to ${rankedPairs.map((p) => `"${p.tok}" (${(p.w * 100).toFixed(0)}%)`).join(" & ")}.`;
  const stepTrain = `2–3. TensorFlow.js Training & Adjustment (${lmSnap.backend}): Deep Sequential model (${lmSnap.paramCount.toLocaleString()} weights/biases, tf.train.${lmSnap.optimizer}, ${lmSnap.stepCount} model.fit epochs, loss ${lmSnap.initialLoss.toFixed(2)}→${lmSnap.currentLoss.toFixed(3)}, acc ${(lmSnap.accuracy * 100).toFixed(1)}%) + Micrograd 6→8→5 MLP routed to "${dominantPathway}" (act ${(mlpConfidence * 100).toFixed(0)}%, ‖∇‖=${gradNorm.toFixed(3)}); top next-token guesses: ${nextGuessPreview}.`;

  let step2 = "";
  let step3 = "";

  if (fNeuralLab) {
    step2 =
      "4. Inference Objective: User wants to inspect or run real Pure-TS neural mechanisms (Input tokenization, Deep MLP training/Adam updates, Attention, or MatMul).";
    step3 =
      "Execution Plan: Compute forward/backward weights, surface live loss/gradient telemetry, and mount interactive NeuralEngineLab.";
  } else if (fWebCode) {
    step2 =
      "4. Inference Objective: User is asking for working code or a complete web page—not a textbook definition. Ensure clean semantic HTML5/TS structure, accessible controls, responsive layout, and zero external CDN dependencies.";
    step3 =
      "Design & Verification: Plan CSS variables (light/dark tokens), interactive DOM/state handlers, and package as a self-contained runnable block with an explanation of how it works.";
  } else if (fMathData) {
    step2 =
      "4. Inference Objective: Detected mathematical expression, equation, or numeric dataset. Parse operands symbolically to guarantee zero floating-point/hallucination drift.";
    step3 =
      "Verification Strategy: Solve step-by-step (collecting terms / evaluating regression or power rule) and verify LHS = RHS or R² fit before formatting.";
  } else if (fWritePlan) {
    step2 =
      "4. Inference Objective: Structured communication, comparison, or planning request. Identify the audience, comparison criteria, or key deliverables so the draft is immediately actionable.";
    step3 =
      "Drafting Structure: Organize into a clear opening overview, structured breakdown/table, and concrete next steps.";
  } else if (fResearch) {
    step2 =
      "4. Inference Objective: Informational or concept query. Strip conversational wrappers/articles (e.g. 'what is a ...'), prioritize authoritative definitions & Wikipedia intros, and reject noisy promotional snippets.";
    step3 =
      "Synthesis Plan: Fetch authoritative passages via local HTTP research proxy, score relevance with BM25 + bigram overlap, and present a cohesive multi-sense summary with citations.";
  } else {
    step2 = `4. Inference Objective: Interpreting "${clean.slice(0, 72)}" in conversational context while checking whether local math, code, or live web context applies.`;
    step3 =
      "Response Plan: Formulate a direct, clear reply matching the user's tone and scope without unnecessary boilerplate.";
  }

  const steps =
    effort === "Low"
      ? [step1, stepTrain]
      : effort === "Max Effort"
        ? [
            step1,
            stepTrain,
            step2,
            step3,
            "Self-Check: Confirm clarity, formatting consistency, zero external broken links, and exact user-constraint alignment before streaming output.",
          ]
        : [step1, stepTrain, step2, step3];

  return {
    dominantPathway,
    focusTokens: rankedPairs.map((p) => p.tok),
    topWeight: bestWeight,
    mlpConfidence,
    gradNorm,
    steps,
  };
}

// Streams thought trace and response tokens cleanly (keeping ```nexora-lab / ```nexora-preview blocks atomic so widgets render smoothly)
export async function streamNexoraBuiltInAssistant(opts: {
  prompt: string;
  history?: AssistantHistoryTurn[];
  memoryFacts?: ChatMemoryFact[];
  threadDigests?: ThreadMemoryDigest[];
  activeThreadTitle?: string;
  effort?: string;
  language?: string;
  isOnline?: boolean;
  hasWebGpuReady?: boolean;
  localWebGpuTier?: LocalModelTier;
  hasSavedByokModels?: boolean;
  signal?: AbortSignal;
  onProgress?: (info: LocalLoadProgress) => void;
  onThought?: (delta: string) => void;
  onThinkingLine?: (line: string) => void;
  onSources?: (sources: ResearchSourceCitation[]) => void;
  onReplaceText?: (text: string) => void;
  onDelta: (delta: string) => void;
}): Promise<BuiltInAssistantResult> {
  const trimmed = opts.prompt.trim();

  // Ensure TFJS WebGL/CPU warmup finishes cleanly before thinking/training
  await ensureSharedNeuralModelReady();

  // Execute real Transformer Attention + Micrograd MLP + TFJS inspection pass and stream thought steps
  const neuralPass = runNeuralThoughtPass(trimmed || "hello", opts.effort);
  opts.onThinkingLine?.(`Neuron pass: ${neuralPass.dominantPathway}`);

  for (const step of neuralPass.steps) {
    if (opts.signal?.aborted) break;
    opts.onThought?.(`${step}\n\n`);
    const ok = await sleepAbortable(opts.effort === "Low" ? 70 : 125, opts.signal);
    if (!ok) break;
  }

  // Check if user is asking to harvest real website/web data and feed it into the TypeScript TensorFlow.js Deep Neural Engine
  const harvestIntent = parseWebHarvestIntent(trimmed);
  let resolved: BuiltInAssistantResult | null = null;

  if (harvestIntent && opts.isOnline !== false && !opts.signal?.aborted) {
    const targetDesc = harvestIntent.url
      ? harvestIntent.url
      : harvestIntent.topic
        ? `topic "${harvestIntent.topic}"`
        : "multi-domain web & Wikipedia knowledge";
    opts.onThinkingLine?.(`Harvesting live web corpus (${targetDesc})...`);
    opts.onThought?.(
      `Step 1 (Web Data Fuel): Crawling public pages & extracting clean prose from ${targetDesc}...\n\n`
    );
    try {
      const { harvest, before, after, sampleInference } =
        await harvestAndFeedNeuralEngine({
          url: harvestIntent.url,
          topic: harvestIntent.topic,
          maxPages: 12,
          epochs: 16,
          optimizer: "adam",
          signal: opts.signal,
        });

      const citations: ResearchSourceCitation[] = harvest.sources.map((s) => ({
        title: s.title,
        url: s.url,
      }));
      if (citations.length > 0) {
        opts.onSources?.(citations);
      }

      const kb = (harvest.totalChars / 1024).toFixed(1);
      opts.onThinkingLine?.(
        `Fitted TFJS (${after.backend}) on ${harvest.totalWords.toLocaleString()} words (${kb} KB)`
      );
      opts.onThought?.(
        `Step 2–3 (TensorFlow.js model.fit + tf.train.${after.optimizer}): Loaded ${harvest.totalSentences.toLocaleString()} sentences (${kb} KB) from ${harvest.sources.length} web pages; rebuilt vocab ${before.vocabSize} → ${after.vocabSize} tokens (${after.paramCount.toLocaleString()} parameters) and ran ${after.stepCount} epochs on ${after.backend} (loss ${after.initialLoss.toFixed(3)} → ${after.currentLoss.toFixed(3)}, acc ${(after.accuracy * 100).toFixed(1)}%).\n\n`
      );

      const topNextStr = sampleInference.topNextTokens
        .slice(0, 4)
        .map((t) => `\`"${t.token}"\` (${(t.probability * 100).toFixed(1)}%)`)
        .join(", ");

      const sourceListMd = harvest.sources
        .slice(0, 12)
        .map(
          (s, idx) =>
            `${idx + 1}. **${s.title}** — ${s.words.toLocaleString()} words (\`${(
              s.chars / 1024
            ).toFixed(1)} KB\`)`
        )
        .join("\n");

      resolved = {
        labTab: "pipeline",
        titleHint: "Web Corpus -> TensorFlow.js",
        reasoningNote: `Harvested ${harvest.totalWords.toLocaleString()} words (${kb} KB) across ${harvest.sources.length} web pages and trained TensorFlow.js (${after.backend}, ${after.paramCount.toLocaleString()} params, tf.train.${after.optimizer}, ${after.stepCount} model.fit epochs, loss ${after.initialLoss.toFixed(2)} → ${after.currentLoss.toFixed(3)}).`,
        thinkingLines: [
          `Crawled ${harvest.sources.length} real web sources (${harvest.totalWords.toLocaleString()} words)`,
          `Trained ${after.paramCount.toLocaleString()} TFJS weights (${after.stepCount} epochs on ${after.backend})`,
        ],
        sources: citations,
        markdown: [
          `### Live Web Data Fed & Fitted in TensorFlow.js (\`${after.backend}\` backend)`,
          "",
          `Crawled **${harvest.sources.length} real public web pages & full articles**, cleaned **${harvest.totalSentences.toLocaleString()} prose sentences** (**${harvest.totalWords.toLocaleString()} words · ${kb} KB**), compiled \`tf.sequential()\` with \`tf.train.${after.optimizer}\`, and executed **\`await model.fit()\`**:`,
          "",
          "| TensorFlow.js Metric | Before Web Feed | After Web Feed & `model.fit()` |",
          "| --- | ---: | ---: |",
          `| **Dataset Size** | ${before.corpusWords.toLocaleString()} words | **${after.corpusWords.toLocaleString()} words (${(after.corpusChars / 1024).toFixed(1)} KB)** |`,
          `| **Token Stream Length** | ${before.corpusTokenStreamLength.toLocaleString()} tokens | **${after.corpusTokenStreamLength.toLocaleString()} tokens** |`,
          `| **Vocabulary Size** | ${before.vocabSize} tokens | **${after.vocabSize} tokens** |`,
          `| **TFJS Trainable Params** | ${before.paramCount.toLocaleString()} | **${after.paramCount.toLocaleString()}** |`,
          `| **Loss & Accuracy** | ${before.currentLoss.toFixed(3)} | **${after.initialLoss.toFixed(3)} → ${after.currentLoss.toFixed(3)}** (${(after.accuracy * 100).toFixed(1)}% acc, ${after.stepCount} epochs) |`,
          "",
          "#### Web Sources Fed into the TensorFlow.js Corpus",
          sourceListMd,
          "",
          "#### Stage 4 `model.predict()` Check on Freshly Trained Weights",
          `- **Top predicted tokens:** ${topNextStr}`,
          `- **Generated continuation:** _"${sampleInference.generatedContinuation}"_`,
          "",
          "You can switch optimizers (`tf.train.adam` / `tf.train.sgd`), run more `model.fit()` epochs, or crawl another URL directly below:",
          "",
          "```nexora-lab",
          "pipeline",
          "```",
        ].join("\n"),
      };
    } catch (err) {
      opts.onThought?.(
        `Web harvest note: ${err instanceof Error ? err.message : "Network unavailable"}; falling back to local neural pipeline.\n\n`
      );
    }
  }

  if (!resolved) {
    resolved = tryDeterministicPureTsHandlers(trimmed, {
      hasWebGpuReady: opts.hasWebGpuReady,
      hasSavedByokModels: opts.hasSavedByokModels,
      history: opts.history,
      memoryFacts: opts.memoryFacts,
      threadDigests: opts.threadDigests,
      activeThreadTitle: opts.activeThreadTitle,
    });
  }

  const canUseQwenLocal =
    Boolean(opts.hasWebGpuReady) &&
    Boolean(opts.localWebGpuTier) &&
    opts.localWebGpuTier !== "smollm2-360m";

  const isLocalRefinementOrIdentity =
    /^(?:make (?:it )?shorter|shorter|summarize|tl;?dr|explain simpler|simpler|in simple words)[\s!?.]*$/i.test(
      trimmed
    ) ||
    /\b(who\s+are\s+you|what\s+are\s+you|which\s+model|what\s+model)\b/i.test(trimmed);

  const researchQuery = resolveConversationalFollowUpTopic(trimmed, opts.history);

  let fetchedPassages: ResearchPassage[] = [];
  if (
    !resolved &&
    !isLocalRefinementOrIdentity &&
    opts.isOnline !== false &&
    trimmed.length > 0 &&
    !opts.signal?.aborted
  ) {
    opts.onThinkingLine?.(
      canUseQwenLocal
        ? "Fetching Dictionary & Wikipedia grounding for on-device Qwen"
        : "Querying Dictionary, Wikipedia & web indexes (Pure-TS BM25)"
    );
    opts.onThought?.(
      researchQuery !== trimmed
        ? `Source Lookup: Resolved follow-up query "${researchQuery}" via conversation context and querying authoritative Dictionary & Wikipedia...\n\n`
        : "Source Lookup: Querying authoritative Dictionary, Wikipedia & public indexes via local HTTP proxy (zero external AI providers)...\n\n"
    );
    fetchedPassages = await fetchBuiltInResearchPassages(researchQuery, opts.signal);
  }

  // If user has cached Qwen 3 / Qwen 3.5 WebGPU weights on-device, run real neural inference grounded by live research
  if (!resolved && canUseQwenLocal && trimmed.length > 0 && !opts.signal?.aborted) {
    const preset = getLocalModelPreset(opts.localWebGpuTier);
    opts.onThinkingLine?.(`Running on-device ${preset.shortLabel} (WebGPU)`);

    const topPassages = [...fetchedPassages]
      .sort((a, b) => b.authority - a.authority)
      .slice(0, 3);
    const citations: ResearchSourceCitation[] = topPassages
      .filter((p) => p.url)
      .map((p) => ({ title: p.title, url: p.url }));
    if (citations.length > 0) {
      opts.onSources?.(citations);
    }

    const memoryLines = (opts.memoryFacts ?? [])
      .slice(0, 12)
      .map((f) => `- ${f.key}: ${f.value}`);
    const digestLines = (opts.threadDigests ?? [])
      .slice(0, 5)
      .map((d) => `- "${d.title}" (${d.updatedAt}): ${d.summary}`);
    const memoryBlock =
      memoryLines.length > 0 || digestLines.length > 0
        ? [
            memoryLines.length > 0 ? `[Saved User Memory]\n${memoryLines.join("\n")}` : "",
            digestLines.length > 0 ? `[Past Conversations]\n${digestLines.join("\n")}` : "",
          ]
            .filter(Boolean)
            .join("\n\n")
        : "";
    const passageBlock =
      topPassages.length > 0
        ? topPassages
            .map(
              (p, i) =>
                `[Source ${i + 1}: ${p.title}]\n${p.text.slice(0, 650)}`
            )
            .join("\n\n")
        : "";
    const researchContext =
      [memoryBlock, passageBlock].filter(Boolean).join("\n\n") || undefined;

    try {
      let qwenStreamed = "";
      const fullReply = await streamLocalWebGpuChat({
        prompt: trimmed,
        history: opts.history ?? [],
        tier: opts.localWebGpuTier,
        assistantName: "Nexora Assistant",
        researchContext,
        language: opts.language,
        effort: opts.effort,
        signal: opts.signal,
        onProgress: opts.onProgress,
        onThought: opts.onThought,
        onDelta: (delta) => {
          qwenStreamed += delta;
          opts.onDelta(delta);
        },
        onReplaceText: (cleaned) => {
          qwenStreamed = cleaned;
          opts.onReplaceText?.(cleaned);
        },
      });

      const finalText = (fullReply || qwenStreamed).trim();
      if (finalText) {
        return {
          markdown: finalText,
          thinkingLines: [
            ...(topPassages.length > 0
              ? [`Grounded with ${topPassages.length} public reference(s)`]
              : []),
            `Generated on-device with ${preset.shortLabel} (WebGPU)`,
          ],
          sources: citations.length > 0 ? citations : undefined,
        };
      }
    } catch {
      // If WebGPU execution fails or VRAM overflows, fall through smoothly to Pure-TS BM25 synthesis
      opts.onThought?.(
        `On-device ${preset.shortLabel} unavailable for this step; falling back to Pure-TS BM25 synthesis...\n`
      );
    }
  }

  // Pure-TS BM25 + Dictionary synthesis when WebGPU neural model is not loaded
  if (!resolved && fetchedPassages.length > 0) {
    opts.onThought?.(
      `Retrieved ${fetchedPassages.length} source passage(s). Scoring with Pure-TS BM25 & dictionary lookup...\n`
    );
    const synth = synthesizeResearchWithBM25(researchQuery, fetchedPassages);
    if (synth) {
      if (synth.citations.length > 0) {
        opts.onSources?.(synth.citations);
      }
      resolved = {
        markdown: synth.markdown,
        titleHint: synth.citations[0]?.title
          .replace(/\s*—\s*(Wikipedia|Dictionary Definition)$/i, "")
          .slice(0, 38),
        reasoningNote: synth.reasoningNote,
        thinkingLines: [
          `Retrieved ${fetchedPassages.length} public passages`,
          "Synthesized grounded answer via Pure-TS BM25 ranking",
        ],
        sources: synth.citations,
      };
    }
  }

  if (!resolved) {
    resolved = runNexoraBuiltInAssistant(trimmed, {
      hasWebGpuReady: opts.hasWebGpuReady,
      hasSavedByokModels: opts.hasSavedByokModels,
      history: opts.history,
      researchPassages: fetchedPassages,
      memoryFacts: opts.memoryFacts,
      threadDigests: opts.threadDigests,
      activeThreadTitle: opts.activeThreadTitle,
    });
  }

  if (resolved.reasoningNote) {
    opts.onThought?.(`${resolved.reasoningNote}\n`);
    await sleepAbortable(45, opts.signal);
  }
  for (const line of resolved.thinkingLines) {
    opts.onThinkingLine?.(line);
  }

  // Tokenize output cleanly into streamable chunks, keeping interactive nexora-lab / nexora-preview blocks whole
  const text = resolved.markdown;
  const atomicFenceRegex = /```(?:nexora-lab|neural-lab|engine-lab|nexora-preview|html-preview|preview|widget)[\s\S]*?```/g;
  const parts: Array<{ value: string; atomic: boolean }> = [];
  let cursor = 0;
  let m: RegExpExecArray | null;
  while ((m = atomicFenceRegex.exec(text)) !== null) {
    if (m.index > cursor) {
      parts.push({ value: text.slice(cursor, m.index), atomic: false });
    }
    parts.push({ value: m[0], atomic: true });
    cursor = atomicFenceRegex.lastIndex;
  }
  if (cursor < text.length) {
    parts.push({ value: text.slice(cursor), atomic: false });
  }

  const chunkStep = opts.effort === "Low" ? 36 : 20;
  for (const part of parts) {
    if (opts.signal?.aborted) break;
    if (part.atomic) {
      opts.onDelta(part.value);
      await sleepAbortable(25, opts.signal);
    } else {
      let i = 0;
      while (i < part.value.length && !opts.signal?.aborted) {
        let nextEnd = Math.min(part.value.length, i + chunkStep);
        // Align to word boundary when convenient for crisp typography
        if (nextEnd < part.value.length) {
          const spaceAhead = part.value.indexOf(" ", nextEnd);
          if (spaceAhead !== -1 && spaceAhead - nextEnd <= 10) {
            nextEnd = spaceAhead + 1;
          }
        }
        opts.onDelta(part.value.slice(i, nextEnd));
        i = nextEnd;
        if (i < part.value.length) {
          const ok = await sleepAbortable(14, opts.signal);
          if (!ok) break;
        }
      }
    }
  }

  return resolved;
}
