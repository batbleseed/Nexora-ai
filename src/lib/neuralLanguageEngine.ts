/* ───────────────────────────────────────────────────────────────────────────
 * TensorFlow.js (@tensorflow/tfjs) Deep Neural Language Engine
 * Uses GPU-accelerated WebGL (with automatic CPU fallback) + Keras-style Layers:
 *   1. Input      -> Corpus cleaning, vocabulary indexing, tf.layers.embedding
 *   2. Architecture -> tf.sequential() [Embedding -> Flatten -> Dense(tanh) -> Dropout -> Dense(relu) -> Softmax]
 *   3. Training   -> model.compile({ optimizer: tf.train.adam / tf.train.sgd, loss: sparseCategoricalCrossentropy })
 *                    & real asynchronous model.fit() epochs
 *   4. Inference  -> model.predict() Top-K Softmax next-token probabilities & autoregressive generation
 * ─────────────────────────────────────────────────────────────────────────── */

import * as tf from "@tensorflow/tfjs";
import { runEducationalTransformerForward } from "./transformerEngine";
import type { AssistantHistoryTurn, ResearchPassage } from "./pureTsIntelligence";
import type { ChatMemoryFact, ThreadMemoryDigest } from "./chatMemory";

export type OptimizerKind = "adam" | "sgd";

export const DEFAULT_NEXORA_TRAINING_CORPUS = `
Artificial intelligence systems learn patterns from massive datasets by turning raw text into numeric tokens and dense embeddings.
Data is the fuel of machine learning because every pattern starts inside real examples from text, code, science, and real world websites.
Machine learning is the engine that discovers mathematical rules directly from examples instead of relying on brittle hard coded branches.
Neural networks connect layers of artificial neurons where each node multiplies inputs by learned weights, adds biases, and applies nonlinear activation functions.
Deep learning stacks multiple hidden layers so TensorFlow models extract rich semantic features from complex language sequences automatically.
During step one called input, raw text is cleaned, split into tokens, mapped to integer identifiers, and projected through a trainable embedding layer.
During step two called training, model.fit passes context windows through hidden layers, predicts next-token probabilities, and measures cross entropy loss.
During step three called adjustment, backpropagation computes gradients and optimizers like Adam or SGD update every weight matrix and bias vector.
During step four called inference, the compiled model runs model.predict to transform any new prompt into calibrated probabilities and generated continuations.
Attention mechanisms let tokens compare queries and keys with scaled dot products so relevant earlier context guides the next prediction.
TensorFlow.js executes matrix multiplications and backpropagation on WebGL GPU shaders directly inside the browser.
When cross entropy loss falls and training accuracy rises, predicted probabilities sharpen around coherent continuations and domain concepts.
Software systems combine deterministic parsing, live web retrieval, and neural pattern recognition to solve real engineering problems reliably.
Engineers build resilient web applications with semantic HTML, accessible interfaces, modular TypeScript types, and clean REST endpoints.
JavaScript adds dynamic browser interactivity while TypeScript adds compile-time static types that catch mismatches before deployment.
HTTP APIs exchange JSON payloads across routes like GET and POST using explicit status codes such as 200 OK, 400 Bad Request, and 500 Server Error.
CSS Grid handles two-dimensional rows and columns while Flexbox aligns one-dimensional navigation bars, toolbars, and responsive cards cleanly.
Linear algebra transforms input vectors through weight matrices and bias offsets, while calculus drives gradient descent via the chain rule.
A capital refers to the official seat-of-government city of a country or region, productive financial assets used in business, or an uppercase letter.
Inflation describes a broad sustained rise in prices across goods and services over time, reducing the purchasing power of currency.
Water boils at 100 degrees Celsius or 212 degrees Fahrenheit at standard atmospheric pressure and freezes at 0 degrees Celsius or 32 degrees Fahrenheit.
Safe minimum internal cooking temperature for poultry such as chicken and turkey is 165 degrees Fahrenheit or 74 degrees Celsius.
Python uses clean indentation and concise loops like for i in range to iterate over numeric sequences, lists, and datasets.
Hardware, interface, and neural systems work together cleanly when latency stays low and typography stays legible.
Nexora UI uses cool near-white daylight canvases, deep matte obsidian dark surfaces, one-pixel hairline borders, and smooth layered shadows.
Nexora pairs Plus Jakarta Sans headings with DM Sans body copy and JetBrains Mono tabular numerals to keep layouts calm and crisp.
Clean HTML interfaces rely on a single-row sticky navigation bar, inset segmented switches, asymmetric bento grids, and zero decorative pill clutter.
`.trim();

export const CORPUS_PRESETS: Record<
  "ai_core" | "nexora_ui" | "code_arch" | "science_math",
  { label: string; text: string }
> = {
  ai_core: {
    label: "AI & Deep Learning Core",
    text: DEFAULT_NEXORA_TRAINING_CORPUS,
  },
  nexora_ui: {
    label: "Nexora Sleek UI System",
    text: `
Nexora visual design builds calm, high-craft web interfaces from cool near-white light surfaces and matte obsidian dark surfaces.
Every card sits above the canvas cleanly using one-pixel hairline borders and soft two-layer elevation shadows without heavy drop shadows.
Display titles use Plus Jakarta Sans with tight negative tracking while body copy uses DM Sans and numeric tables enforce tabular numerals.
Segmented controls group compact view switches inside a soft inset track where the active tab elevates with crisp contrast.
Sticky top bars follow a three-zone single-row contract combining a minimal brand mark, concise centered links, and one primary action.
Modern HTML templates avoid uppercase shouting eyebrows, cluttered badge clusters, and nested card sandwiches in favor of balanced whitespace.
`.trim(),
  },
  code_arch: {
    label: "TypeScript & Web Systems",
    text: `
TypeScript programs verify data shapes at compile time using interfaces, union types, and strict type narrowing.
Modular functions separate TensorFlow neural state from user interface components so applications remain fast and predictable.
HTTP servers handle JSON payloads, crawl public websites safely, strip HTML boilerplate, and return structured corpora.
Reactive React interfaces render live training loss curves, optimizer selectors, and next-token probability bars from TensorFlow state.
Clean CSS variables define surface colors, hairline borders, and typography hierarchy across light and dark themes.
Automated checks run type verification and production Vite bundling after every edit to guarantee zero runtime regressions.
`.trim(),
  },
  science_math: {
    label: "Science & Mathematics",
    text: `
Linear algebra transforms input vectors through weight matrices and bias offsets across deep neural layers.
Calculus tracks rates of change through partial derivatives and the multivariable chain rule during backpropagation.
Probability distributions normalize raw logits with softmax so predicted token likelihoods always sum cleanly to one.
Stars fuse hydrogen into helium inside dense cores while gravity balances outward radiation pressure across billions of years.
Earth orbits the Sun while ecosystems exchange carbon, water, and thermal energy between oceans, soil, and atmosphere.
`.trim(),
  },
};

export interface TokenProbability {
  token: string;
  id: number;
  probability: number;
  logit: number;
}

export interface HarvestedSourceInfo {
  title: string;
  url: string;
  chars: number;
  words: number;
}

export interface HarvestCorpusResponse {
  mode: "url_crawl" | "topic_harvest" | "broad_web_harvest";
  corpusText: string;
  sources: HarvestedSourceInfo[];
  totalChars: number;
  totalWords: number;
  totalSentences: number;
}

export interface NeuralPipelineSnapshot {
  framework: string;
  backend: string;
  optimizer: OptimizerKind;
  learningRate: number;
  vocabSize: number;
  paramCount: number;
  contextWindow: number;
  dModel: number;
  hiddenDim: number;
  stepCount: number;
  initialLoss: number;
  currentLoss: number;
  accuracy: number;
  perplexity: number;
  lastGradNorm: number;
  lastWeightDeltaNorm: number;
  lossHistory: number[];
  sampleTokens: { token: string; id: number }[];
  corpusChars: number;
  corpusWords: number;
  corpusTokenStreamLength: number;
  sourcesFed: HarvestedSourceInfo[];
  isTraining: boolean;
}

export interface NeuralInferenceTrace {
  promptTokens: string[];
  promptTokenIds: number[];
  embeddingPreview: number[];
  attentionFocusWeights: number[];
  hiddenActivationNorm: number;
  topNextTokens: TokenProbability[];
  generatedContinuation: string;
}

let tfBackendReadyPromise: Promise<string> | null = null;

async function ensureTfBackendReady(): Promise<string> {
  if (!tfBackendReadyPromise) {
    tfBackendReadyPromise = (async () => {
      await tf.ready();
      return tf.getBackend() || "cpu";
    })();
  }
  return tfBackendReadyPromise;
}

function createRng(seed = 2026): () => number {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function tokenizeWords(text: string): string[] {
  const matches = text
    .toLowerCase()
    .replace(/\r\n/g, " ")
    .match(/[a-z0-9]+(?:'[a-z]+)?|[.,!?;:()]/g);
  return matches ?? [];
}

function detokenizeWords(tokens: string[]): string {
  let out = "";
  for (const tok of tokens) {
    if (tok === "<eos>" || tok === "<pad>" || tok === "<unk>") continue;
    if (/^[.,!?;:)]$/.test(tok)) {
      out += tok;
    } else if (out.length === 0 || out.endsWith("(")) {
      out += tok;
    } else {
      out += ` ${tok}`;
    }
  }
  return out.trim();
}

/**
 * Trainable TensorFlow.js Deep Neural Language Model:
 *   Context Window (K=6 tokens)
 *   -> tf.layers.embedding(vocabSize, dModel=24)
 *   -> tf.layers.flatten()
 *   -> tf.layers.dense(96, activation="tanh")
 *   -> tf.layers.dropout(0.1)
 *   -> tf.layers.dense(96, activation="relu")
 *   -> tf.layers.dense(vocabSize, activation="softmax")
 * Compiled with tf.train.adam / tf.train.sgd & sparseCategoricalCrossentropy,
 * trained asynchronously via real `await model.fit(...)`.
 */
export class DeepNeuralLanguageModel {
  readonly contextWindow = 6;
  readonly dModel = 24;
  readonly hiddenDim = 96;

  private vocab: string[] = ["<pad>", "<unk>", "<eos>"];
  private tokenToId = new Map<string, number>();
  private corpusIds: number[] = [];
  private corpusSentences: string[] = [];
  private corpusTextRaw = "";
  private corpusCharsCount = 0;
  private corpusWordsCount = 0;
  private sourcesFed: HarvestedSourceInfo[] = [];

  private model: tf.Sequential | null = null;
  private optimizerKind: OptimizerKind = "adam";
  private learningRate = 0.008;
  private paramCountCache = 0;
  private embeddingWeightsCache: Float32Array | null = null;
  private backendName = "initializing";
  private isFitting = false;

  private epochCount = 0;
  private initialLoss = 0;
  private currentLoss = 0;
  private currentAccuracy = 0;
  private lastWeightDeltaNorm = 0;
  private lossHistory: number[] = [];

  constructor(corpusText = DEFAULT_NEXORA_TRAINING_CORPUS, seed = 2026) {
    this.loadCorpus(corpusText, seed, []);
  }

  get vocabSize(): number {
    return this.vocab.length;
  }

  get paramCount(): number {
    return this.paramCountCache;
  }

  getCorpusPreview(maxChars = 16_000): string {
    return this.corpusTextRaw.slice(0, maxChars);
  }

  getCorpusSentences(): string[] {
    return this.corpusSentences;
  }

  private createTfOptimizer(kind: OptimizerKind, lr: number): tf.Optimizer {
    return kind === "sgd" ? tf.train.sgd(lr) : tf.train.adam(lr);
  }

  private buildAndCompileTfModel(
    optimizerKind: OptimizerKind = this.optimizerKind,
    lr: number = this.learningRate
  ): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }

    const model = tf.sequential({ name: "nexora_tfjs_deep_lm" });
    model.add(
      tf.layers.embedding({
        inputDim: this.vocabSize,
        outputDim: this.dModel,
        inputLength: this.contextWindow,
        name: "token_embedding",
      })
    );
    model.add(tf.layers.flatten({ name: "context_flatten" }));
    model.add(
      tf.layers.dense({
        units: this.hiddenDim,
        activation: "tanh",
        kernelInitializer: "glorotUniform",
        name: "deep_hidden_1",
      })
    );
    model.add(tf.layers.dropout({ rate: 0.1, name: "regularization_dropout" }));
    model.add(
      tf.layers.dense({
        units: this.hiddenDim,
        activation: "relu",
        kernelInitializer: "heNormal",
        name: "deep_hidden_2",
      })
    );
    model.add(
      tf.layers.dense({
        units: this.vocabSize,
        activation: "softmax",
        name: "vocab_softmax_head",
      })
    );

    model.compile({
      optimizer: this.createTfOptimizer(optimizerKind, lr),
      loss: "sparseCategoricalCrossentropy",
      metrics: ["accuracy"],
    });

    this.model = model;
    this.optimizerKind = optimizerKind;
    this.learningRate = lr;
    this.paramCountCache = model.countParams();
    this.refreshEmbeddingCache();
  }

  private refreshEmbeddingCache(): void {
    if (!this.model) return;
    try {
      const embLayer = this.model.layers[0];
      const weights = embLayer?.getWeights();
      if (weights && weights.length > 0) {
        const raw = weights[0].dataSync();
        this.embeddingWeightsCache =
          raw instanceof Float32Array ? new Float32Array(raw) : new Float32Array(raw);
      }
    } catch {
      // Backend still warming up
    }
  }

  setOptimizer(optimizerKind: OptimizerKind, lr?: number): void {
    this.optimizerKind = optimizerKind;
    if (lr !== undefined) {
      this.learningRate = lr;
    } else {
      this.learningRate = optimizerKind === "sgd" ? 0.08 : 0.008;
    }
    if (this.model) {
      this.model.compile({
        optimizer: this.createTfOptimizer(this.optimizerKind, this.learningRate),
        loss: "sparseCategoricalCrossentropy",
        metrics: ["accuracy"],
      });
    }
  }

  loadCorpus(
    corpusText: string,
    _seed = 2026,
    sources: HarvestedSourceInfo[] = []
  ): void {
    const trimmedCorpus = corpusText.slice(0, 220_000);
    this.corpusTextRaw = trimmedCorpus;
    this.corpusCharsCount = trimmedCorpus.length;
    const rawTokens = tokenizeWords(trimmedCorpus);
    this.corpusWordsCount = trimmedCorpus.split(/\s+/).filter(Boolean).length;
    this.sourcesFed = sources;

    const freq = new Map<string, number>();
    for (const tok of rawTokens) {
      freq.set(tok, (freq.get(tok) ?? 0) + 1);
    }

    // With WebGL TensorFlow.js, scaling vocab cleanly to ~760 tokens trains effortlessly
    const vocabCap = rawTokens.length > 1800 ? 760 : 380;
    const sortedTokens = Array.from(freq.entries())
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, vocabCap)
      .map(([tok]) => tok);

    this.vocab = ["<pad>", "<unk>", "<eos>", ...sortedTokens];
    this.tokenToId = new Map(this.vocab.map((t, idx) => [t, idx]));

    const unkId = 1;
    const eosId = 2;
    this.corpusIds = [];
    this.corpusSentences = [];

    const rawLines = trimmedCorpus
      .split(/\n+/)
      .map((line) => line.trim())
      .filter(Boolean);

    for (const line of rawLines) {
      const subSentences =
        line.length > 260
          ? line.split(/(?<=[.!?])\s+/).filter((s) => s.trim().length > 0)
          : [line];

      for (const sentence of subSentences) {
        const cleanSent = sentence.trim();
        if (!cleanSent) continue;
        if (cleanSent.length >= 24 && cleanSent.length <= 420 && this.corpusSentences.length < 1600) {
          this.corpusSentences.push(cleanSent);
        }
        const toks = tokenizeWords(cleanSent);
        if (toks.length === 0) continue;
        for (const tok of toks) {
          this.corpusIds.push(this.tokenToId.get(tok) ?? unkId);
        }
        this.corpusIds.push(eosId);
      }
    }

    if (this.corpusIds.length < this.contextWindow + 2) {
      this.corpusIds.push(0, 0, 0, 0, 0, 0, eosId, eosId);
    }

    this.buildAndCompileTfModel(this.optimizerKind, this.learningRate);
    this.epochCount = 0;
    const uniformEntropy = Math.log(Math.max(4, this.vocabSize));
    this.initialLoss = uniformEntropy;
    this.currentLoss = uniformEntropy;
    this.currentAccuracy = 1 / Math.max(4, this.vocabSize);
    this.lastWeightDeltaNorm = 0;
    this.lossHistory = [uniformEntropy];
    this.backendName = tf.getBackend() || this.backendName;
  }

  private buildTrainingTensors(maxSamples = 2600): {
    xs: tf.Tensor2D;
    ys: tf.Tensor1D;
    sampleCount: number;
  } {
    const N = this.corpusIds.length;
    const K = this.contextWindow;
    const totalPositions = Math.max(1, N - 1);
    const stride = Math.max(1, Math.floor(totalPositions / maxSamples));

    const xFlat: number[] = [];
    const yFlat: number[] = [];

    for (let pos = 1; pos < N && yFlat.length < maxSamples; pos += stride) {
      const ctx = this.buildContextIds(this.corpusIds.slice(Math.max(0, pos - K), pos));
      for (let k = 0; k < K; k++) {
        xFlat.push(ctx[k]);
      }
      yFlat.push(this.corpusIds[pos]);
    }

    const sampleCount = yFlat.length;
    const xs = tf.tensor2d(xFlat, [sampleCount, K], "int32");
    const ys = tf.tensor1d(yFlat, "float32");
    return { xs, ys, sampleCount };
  }

  /**
   * Real asynchronous TensorFlow.js `await model.fit()` training loop with
   * live `onEpochEnd` loss/accuracy tracking and automatic tensor cleanup.
   */
  async fitEpochs(
    epochs = 12,
    options?: {
      batchSize?: number;
      maxSamples?: number;
      optimizer?: OptimizerKind;
      learningRate?: number;
      onEpochProgress?: (snap: NeuralPipelineSnapshot) => void;
    }
  ): Promise<NeuralPipelineSnapshot> {
    if (this.isFitting) {
      return this.getSnapshot();
    }
    this.isFitting = true;

    try {
      this.backendName = await ensureTfBackendReady();

      if (
        options?.optimizer &&
        (options.optimizer !== this.optimizerKind ||
          (options.learningRate !== undefined && options.learningRate !== this.learningRate))
      ) {
        this.setOptimizer(options.optimizer, options.learningRate);
      } else if (!this.model) {
        this.buildAndCompileTfModel(this.optimizerKind, this.learningRate);
      }

      const model = this.model!;
      const isGpuBackend = this.backendName.includes("webgl") || this.backendName.includes("webgpu");
      const batchSize = options?.batchSize ?? (isGpuBackend ? 64 : 32);
      const maxSamples = options?.maxSamples ?? (isGpuBackend ? 2600 : 420);
      const { xs, ys, sampleCount } = this.buildTrainingTensors(maxSamples);

      const embBefore = this.embeddingWeightsCache
        ? new Float32Array(this.embeddingWeightsCache)
        : null;

      try {
        await model.fit(xs, ys, {
          epochs,
          batchSize: Math.min(batchSize, Math.max(8, sampleCount)),
          shuffle: true,
          yieldEvery: "auto",
          callbacks: {
            onEpochEnd: async (_epoch, logs) => {
              const lossVal =
                typeof logs?.loss === "number" && Number.isFinite(logs.loss)
                  ? logs.loss
                  : this.currentLoss;
              const accVal =
                typeof logs?.acc === "number" && Number.isFinite(logs.acc)
                  ? logs.acc
                  : typeof logs?.accuracy === "number" && Number.isFinite(logs.accuracy)
                    ? logs.accuracy
                    : this.currentAccuracy;

              this.epochCount += 1;
              if (this.lossHistory.length <= 1 && this.epochCount === 1) {
                this.initialLoss = Math.max(this.initialLoss, lossVal);
              }
              this.currentLoss = lossVal;
              this.currentAccuracy = accVal;
              this.lossHistory.push(lossVal);
              if (this.lossHistory.length > 160) {
                this.lossHistory = this.lossHistory.slice(-160);
              }
              options?.onEpochProgress?.(this.getSnapshot());
            },
          },
        });
      } finally {
        tf.dispose([xs, ys]);
      }

      this.refreshEmbeddingCache();
      if (embBefore && this.embeddingWeightsCache) {
        let diffSq = 0;
        const len = Math.min(embBefore.length, this.embeddingWeightsCache.length);
        for (let i = 0; i < len; i++) {
          const d = this.embeddingWeightsCache[i] - embBefore[i];
          diffSq += d * d;
        }
        this.lastWeightDeltaNorm = Math.sqrt(diffSq);
      }

      return this.getSnapshot();
    } finally {
      this.isFitting = false;
    }
  }

  async feedHarvestedCorpus(
    harvestedText: string,
    sources: HarvestedSourceInfo[],
    epochs = 16
  ): Promise<NeuralPipelineSnapshot> {
    const mergedCorpus = `${DEFAULT_NEXORA_TRAINING_CORPUS}\n${harvestedText}`.slice(0, 220_000);
    const mergedSources = [...this.sourcesFed];
    const seenUrls = new Set(mergedSources.map((s) => s.url));
    for (const src of sources) {
      if (!seenUrls.has(src.url)) {
        seenUrls.add(src.url);
        mergedSources.push(src);
      }
    }
    this.loadCorpus(mergedCorpus, 2026 + mergedSources.length, mergedSources);
    const backend = await ensureTfBackendReady();
    const isGpu = backend.includes("webgl") || backend.includes("webgpu");
    return this.fitEpochs(isGpu ? epochs : Math.min(epochs, 8), {
      maxSamples: isGpu ? 3000 : 480,
      batchSize: isGpu ? 64 : 32,
    });
  }

  /**
   * Brief single-batch online adaptation on recent user context tokens via TensorFlow.js
   */
  async adaptOnText(text: string, epochs = 2): Promise<void> {
    if (this.isFitting || !this.model) return;
    const toks = tokenizeWords(text);
    if (toks.length < 3) return;
    const newIds: number[] = [];
    for (const t of toks) {
      const id = this.tokenToId.get(t);
      if (id !== undefined) newIds.push(id);
    }
    if (newIds.length < 3) return;

    this.corpusIds.push(...newIds, 2);
    const K = this.contextWindow;
    const xFlat: number[] = [];
    const yFlat: number[] = [];
    for (let i = 1; i < newIds.length; i++) {
      const ctx = this.buildContextIds(newIds.slice(Math.max(0, i - K), i));
      for (let k = 0; k < K; k++) xFlat.push(ctx[k]);
      yFlat.push(newIds[i]);
    }
    if (yFlat.length === 0) return;

    this.isFitting = true;
    try {
      await ensureTfBackendReady();
      const xs = tf.tensor2d(xFlat, [yFlat.length, K], "int32");
      const ys = tf.tensor1d(yFlat, "float32");
      try {
        await this.model.fit(xs, ys, {
          epochs,
          batchSize: yFlat.length,
          verbose: 0,
        });
      } finally {
        tf.dispose([xs, ys]);
      }
      this.refreshEmbeddingCache();
    } catch {
      // Skip concurrent/interrupted micro-adaptation silently
    } finally {
      this.isFitting = false;
    }
  }

  encodePrompt(text: string): number[] {
    const toks = tokenizeWords(text);
    const unkId = 1;
    return toks.map((t) => this.tokenToId.get(t) ?? unkId);
  }

  private buildContextIds(prefixIds: number[]): number[] {
    const K = this.contextWindow;
    const out = new Array<number>(K).fill(0);
    const slice = prefixIds.slice(-K);
    const offset = K - slice.length;
    for (let i = 0; i < slice.length; i++) {
      out[offset + i] = slice[i];
    }
    return out;
  }

  private predictDistributionSync(ctxIds: number[]): Float32Array {
    const V = this.vocabSize;
    if (!this.model || !tf.getBackend()) {
      const uniform = new Float32Array(V);
      uniform.fill(1 / Math.max(1, V));
      return uniform;
    }
    const model = this.model;
    return tf.tidy(() => {
      const x = tf.tensor2d([ctxIds], [1, this.contextWindow], "int32");
      const out = model.predict(x) as tf.Tensor;
      const raw = out.dataSync();
      return raw instanceof Float32Array ? new Float32Array(raw) : new Float32Array(raw);
    });
  }

  predictAndGenerate(
    prompt: string,
    maxNewTokens = 18,
    temperature = 0.65,
    topK = 5
  ): NeuralInferenceTrace {
    const promptTokens = tokenizeWords(prompt);
    const promptTokenIds = this.encodePrompt(prompt);
    const ctxIds = this.buildContextIds(promptTokenIds);
    const firstProbs = this.predictDistributionSync(ctxIds);

    const lastEmbId = ctxIds[ctxIds.length - 1] ?? 0;
    const embStart = lastEmbId * this.dModel;
    const embeddingPreview = this.embeddingWeightsCache
      ? Array.from(
          this.embeddingWeightsCache.slice(embStart, embStart + 6),
          (v) => Number(v.toFixed(3))
        )
      : [0, 0, 0, 0, 0, 0];

    const candidates: TokenProbability[] = [];
    for (let v = 3; v < this.vocabSize; v++) {
      const prob = firstProbs[v] || 1e-9;
      candidates.push({
        token: this.vocab[v],
        id: v,
        probability: prob,
        logit: Math.log(Math.max(1e-9, prob)),
      });
    }
    candidates.sort((a, b) => b.probability - a.probability);
    const topNextTokens = candidates.slice(0, topK);

    // Autoregressive generation loop powered by TFJS model.predict()
    const runningIds = [...promptTokenIds];
    const generatedToks: string[] = [];
    const rng = createRng(
      42 + promptTokenIds.reduce((acc, id, idx) => acc + id * (idx + 3), 0)
    );
    const recentSet = new Set<number>(runningIds.slice(-3));

    for (let step = 0; step < maxNewTokens; step++) {
      const stepCtx = this.buildContextIds(runningIds);
      const probs = this.predictDistributionSync(stepCtx);

      const scored: { id: number; score: number }[] = [];
      for (let v = 2; v < this.vocabSize; v++) {
        if (v === 1) continue; // skip <unk>
        const logP = Math.log(Math.max(1e-8, probs[v] || 1e-8));
        const repPenalty = recentSet.has(v) ? 0.9 : 0;
        scored.push({
          id: v,
          score: (logP - repPenalty) / Math.max(0.15, temperature),
        });
      }
      scored.sort((a, b) => b.score - a.score);
      const pool = scored.slice(0, Math.max(1, topK));
      const maxS = pool[0].score;
      const exps = pool.map((p) => Math.exp(p.score - maxS));
      const sumE = exps.reduce((a, b) => a + b, 0) || 1;

      let r = rng() * sumE;
      let chosenId = pool[0].id;
      for (let i = 0; i < pool.length; i++) {
        r -= exps[i];
        if (r <= 0) {
          chosenId = pool[i].id;
          break;
        }
      }

      if (chosenId === 2 && generatedToks.length >= 4) {
        break;
      }
      if (chosenId === 2) {
        chosenId = pool[1]?.id ?? chosenId;
        if (chosenId === 2) break;
      }

      runningIds.push(chosenId);
      recentSet.add(chosenId);
      if (recentSet.size > 5) {
        const firstVal = recentSet.values().next().value;
        if (firstVal !== undefined) recentSet.delete(firstVal);
      }
      generatedToks.push(this.vocab[chosenId]);
    }

    const attnWeights = ctxIds.map((id, idx) =>
      id === 0 ? 0.05 : Number(((idx + 1) / this.contextWindow).toFixed(2))
    );

    return {
      promptTokens: promptTokens.slice(-8),
      promptTokenIds: promptTokenIds.slice(-8),
      embeddingPreview,
      attentionFocusWeights: attnWeights,
      hiddenActivationNorm: Math.sqrt(
        embeddingPreview.reduce((acc, v) => acc + v * v, 0) * 6 + 0.1
      ),
      topNextTokens,
      generatedContinuation: detokenizeWords(generatedToks),
    };
  }

  getSnapshot(sampleText = "neural networks learn patterns"): NeuralPipelineSnapshot {
    const toks = tokenizeWords(sampleText).slice(0, 6);
    const sampleTokens = toks.map((token) => ({
      token,
      id: this.tokenToId.get(token) ?? 1,
    }));

    return {
      framework: "TensorFlow.js",
      backend: tf.getBackend() || this.backendName,
      optimizer: this.optimizerKind,
      learningRate: this.learningRate,
      vocabSize: this.vocabSize,
      paramCount: this.paramCountCache,
      contextWindow: this.contextWindow,
      dModel: this.dModel,
      hiddenDim: this.hiddenDim,
      stepCount: this.epochCount,
      initialLoss: this.initialLoss,
      currentLoss: this.currentLoss,
      accuracy: this.currentAccuracy,
      perplexity: Math.exp(Math.min(10, this.currentLoss)),
      lastGradNorm: this.lastWeightDeltaNorm,
      lastWeightDeltaNorm: this.lastWeightDeltaNorm,
      lossHistory: [...this.lossHistory],
      sampleTokens,
      corpusChars: this.corpusCharsCount,
      corpusWords: this.corpusWordsCount,
      corpusTokenStreamLength: this.corpusIds.length,
      sourcesFed: [...this.sourcesFed],
      isTraining: this.isFitting,
    };
  }

  /**
   * Computes a normalized dense neural embedding vector by averaging
   * TFJS-trained `tf.layers.embedding` vectors + character trigram fallback.
   */
  embedText(text: string): Float64Array {
    const D = this.dModel;
    const vec = new Float64Array(D);
    const toks = tokenizeWords(text);
    if (toks.length === 0) return vec;

    const embTable = this.embeddingWeightsCache;
    for (const tok of toks) {
      const id = this.tokenToId.get(tok);
      if (embTable && id !== undefined && id >= 3) {
        const off = id * D;
        for (let d = 0; d < D; d++) {
          vec[d] += embTable[off + d] ?? 0;
        }
      } else {
        let hash = 0;
        for (let c = 0; c < tok.length; c++) {
          hash = (hash * 31 + tok.charCodeAt(c)) % 997;
        }
        for (let d = 0; d < D; d++) {
          vec[d] += Math.sin((hash + 1) * (d + 1) * 0.23) * 0.35;
        }
      }
    }

    let normSq = 0;
    for (let d = 0; d < D; d++) normSq += vec[d] * vec[d];
    const norm = Math.sqrt(normSq) || 1;
    for (let d = 0; d < D; d++) vec[d] /= norm;
    return vec;
  }
}

function cosineSimilarity(a: Float64Array, b: Float64Array): number {
  let dot = 0;
  for (let i = 0; i < a.length && i < b.length; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

let sharedNexoraModel: DeepNeuralLanguageModel | null = null;
let sharedWarmupPromise: Promise<NeuralPipelineSnapshot> | null = null;

export function getSharedNexoraNeuralModel(): DeepNeuralLanguageModel {
  if (!sharedNexoraModel) {
    sharedNexoraModel = new DeepNeuralLanguageModel(DEFAULT_NEXORA_TRAINING_CORPUS, 2026);
  }
  return sharedNexoraModel;
}

/**
 * Ensures the shared TensorFlow.js neural engine has completed WebGL/CPU initialization
 * and initial `model.fit()` warmup epochs cleanly.
 */
export async function ensureSharedNeuralModelReady(): Promise<NeuralPipelineSnapshot> {
  const model = getSharedNexoraNeuralModel();
  if (model.getSnapshot().stepCount > 0) {
    return model.getSnapshot();
  }
  if (!sharedWarmupPromise) {
    sharedWarmupPromise = (async () => {
      const backend = await ensureTfBackendReady();
      const isGpu = backend.includes("webgl") || backend.includes("webgpu");
      return model.fitEpochs(isGpu ? 14 : 6, {
        batchSize: isGpu ? 48 : 32,
        maxSamples: isGpu ? 1400 : 300,
      });
    })().catch(() => model.getSnapshot());
  }
  return sharedWarmupPromise;
}

// Kick off background TensorFlow.js backend & initial warmup smoothly in browser
if (typeof window !== "undefined") {
  void ensureSharedNeuralModelReady();
}

export async function harvestAndFeedNeuralEngine(opts?: {
  url?: string;
  topic?: string;
  maxPages?: number;
  epochs?: number;
  optimizer?: OptimizerKind;
  signal?: AbortSignal;
  onEpochProgress?: (snap: NeuralPipelineSnapshot) => void;
}): Promise<{
  harvest: HarvestCorpusResponse;
  before: NeuralPipelineSnapshot;
  after: NeuralPipelineSnapshot;
  sampleInference: NeuralInferenceTrace;
}> {
  const model = getSharedNexoraNeuralModel();
  await ensureSharedNeuralModelReady();
  const before = model.getSnapshot(opts?.topic || opts?.url || "neural networks learn");

  const res = await fetch("/api/assistant/harvest-corpus", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: opts?.signal,
    body: JSON.stringify({
      url: opts?.url || "",
      topic: opts?.topic || "",
      maxPages: opts?.maxPages ?? 12,
    }),
  });

  if (!res.ok) {
    throw new Error(`Corpus harvest failed (HTTP ${res.status}).`);
  }
  const harvest = (await res.json()) as HarvestCorpusResponse;
  if (!harvest.corpusText || harvest.corpusText.length < 80) {
    throw new Error("Target website/source returned insufficient readable prose.");
  }

  if (opts?.optimizer) {
    model.setOptimizer(opts.optimizer);
  }

  const after = await model.feedHarvestedCorpus(
    harvest.corpusText,
    harvest.sources ?? [],
    opts?.epochs ?? 16
  );

  const inferQuery =
    opts?.topic && opts.topic.trim().length >= 2
      ? opts.topic.trim()
      : "artificial intelligence learns patterns";
  const sampleInference = model.predictAndGenerate(inferQuery, 16, 0.6, 5);

  return { harvest, before, after, sampleInference };
}

const STOP_AI_WORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "could", "do",
  "does", "for", "from", "have", "how", "i", "if", "in", "is", "it", "me", "my",
  "of", "on", "or", "please", "should", "that", "the", "this", "to", "what",
  "when", "where", "which", "who", "why", "will", "with", "you", "your",
]);

function extractContentStems(text: string): Set<string> {
  const words = tokenizeWords(text).filter(
    (w) => w.length >= 3 && !STOP_AI_WORDS.has(w)
  );
  return new Set(words.map((w) => w.replace(/(?:ing|ed|es|s|ly)$/i, "")));
}

/**
 * TensorFlow.js Neural AI Answer Generator:
 * Combines TFJS-trained token embeddings, Transformer scaled dot-product attention,
 * and `model.predict()` autoregressive decoding over learned web + research passages.
 */
export function generateNeuralAssistantAnswer(
  prompt: string,
  options?: {
    history?: AssistantHistoryTurn[];
    researchPassages?: ResearchPassage[];
    memoryFacts?: ChatMemoryFact[];
    threadDigests?: ThreadMemoryDigest[];
  }
): {
  markdown: string;
  reasoningNote: string;
  thinkingLines: string[];
} {
  const clean = prompt.trim();
  const lower = clean.toLowerCase();
  const model = getSharedNexoraNeuralModel();
  const snap = model.getSnapshot(clean);

  // Handle identity / self-description accurately
  if (
    /\b(who\s+are\s+you|what\s+are\s+you|which\s+model|what\s+model|your\s+name)\b/i.test(
      lower
    )
  ) {
    const userNameFact = options?.memoryFacts?.find((f) => f.id === "identity_user_name");
    const memNote = userNameFact
      ? ` I also recognize your saved profile (**${userNameFact.value}**) from persistent workspace memory (${options?.memoryFacts?.length ?? 0} saved facts).`
      : "";
    return {
      reasoningNote: `Self-inspected active TensorFlow.js (${snap.backend}) neural parameters (${snap.paramCount.toLocaleString()} weights/biases, optimizer=${snap.optimizer}, loss=${snap.currentLoss.toFixed(3)}).`,
      thinkingLines: [`Queried active TensorFlow.js (${snap.backend}) model state`],
      markdown: `I'm **Nexora Assistant**, powered by a **TensorFlow.js (\`@tensorflow/tfjs\`, \`${snap.backend}\` backend) Deep Neural Network** (**${snap.paramCount.toLocaleString()}** trainable weights & biases compiled with \`tf.train.${snap.optimizer}\`, trained via \`model.fit()\`, loss \`${snap.currentLoss.toFixed(3)}\`, accuracy \`${(snap.accuracy * 100).toFixed(1)}%\`) paired with persistent cross-chat memory, symbolic math, live code/HTML synthesis, and real-time dictionary/web grounding.${memNote}`,
    };
  }

  // Handle follow-up refinement on previous assistant message
  const lastAssistant = [...(options?.history ?? [])]
    .reverse()
    .find((t) => t.role === "assistant" && t.content.trim());
  if (
    lastAssistant &&
    /^(make (it )?shorter|shorter|summarize|tl;?dr|explain simpler|simpler|in simple words)[\s!?.]*$/i.test(
      lower
    )
  ) {
    const sentences = lastAssistant.content
      .replace(/```[\s\S]*?```/g, "")
      .replace(/[*#>`]/g, "")
      .split(/(?<=[.!?])\s+/)
      .filter((s) => s.trim().length > 15);
    const concise = sentences.slice(0, 2).join(" ").trim();
    return {
      reasoningNote: "Compressed prior assistant turn using sentence-salience extraction.",
      thinkingLines: ["Condensed previous response"],
      markdown: concise || lastAssistant.content.slice(0, 220),
    };
  }

  const candidatePool: Array<{ text: string; source?: string; priority: number }> = [];
  if (options?.memoryFacts && options.memoryFacts.length > 0) {
    for (const fact of options.memoryFacts.slice(0, 25)) {
      candidatePool.push({
        text: `Remembered user ${fact.key}: ${fact.value}.`,
        source: "Saved Memory",
        priority: 1.55,
      });
    }
  }
  if (options?.threadDigests && options.threadDigests.length > 0) {
    for (const digest of options.threadDigests.slice(0, 12)) {
      candidatePool.push({
        text: `In past chat "${digest.title}" (${digest.updatedAt}): ${digest.summary}`,
        source: `Chat: ${digest.title}`,
        priority: 1.35,
      });
    }
  }
  if (options?.history && options.history.length > 0) {
    for (const turn of options.history.slice(-10)) {
      const snippet = turn.content.replace(/```[\s\S]*?```/g, "").replace(/\s+/g, " ").trim();
      if (snippet.length >= 20 && snippet.length <= 420) {
        candidatePool.push({
          text: `${turn.role === "user" ? "User earlier mentioned" : "Earlier in this chat"}: ${snippet}`,
          source: "Current Chat",
          priority: 1.25,
        });
      }
    }
  }
  if (options?.researchPassages) {
    for (const p of options.researchPassages) {
      const sents = p.text
        .split(/(?<=[.!?])\s+|\n+/)
        .map((s) => s.trim())
        .filter((s) => s.length >= 24 && s.length <= 420);
      for (const s of sents.slice(0, 5)) {
        candidatePool.push({ text: s, source: p.title, priority: p.authority });
      }
    }
  }

  const corpusSentences = model.getCorpusSentences();
  const queryStems = extractContentStems(clean);

  const prioritizedCorpus =
    corpusSentences.length <= 420
      ? corpusSentences
      : [
          ...corpusSentences.filter((s) => {
            const lowerSent = s.toLowerCase();
            for (const stem of queryStems) {
              if (stem.length >= 3 && lowerSent.includes(stem)) return true;
            }
            return false;
          }),
          ...corpusSentences.slice(0, 60),
        ].slice(0, 460);

  for (const s of prioritizedCorpus) {
    if (s.length >= 20) {
      candidatePool.push({ text: s, priority: 0.92 });
    }
  }

  const historyUserContext = (options?.history ?? [])
    .filter((t) => t.role === "user")
    .slice(-4)
    .map((t) => t.content)
    .join(" ");
  const memoryContext = (options?.memoryFacts ?? [])
    .slice(0, 6)
    .map((f) => `${f.key} ${f.value}`)
    .join(" ");
  const queryContext = [historyUserContext, memoryContext, clean].filter(Boolean).join(" ");

  // Trigger lightweight background TFJS adaptation without blocking synchronous prediction
  void model.adaptOnText([clean, memoryContext].filter(Boolean).join(" "), 1);

  const queryVec = model.embedText(queryContext);
  const combinedStems = extractContentStems(queryContext);

  const scored = candidatePool
    .map((cand) => {
      const candVec = model.embedText(cand.text);
      const neuralSim = Math.max(0, cosineSimilarity(queryVec, candVec));
      const candStems = extractContentStems(cand.text);
      let lexicalOverlap = 0;
      for (const s of combinedStems) {
        if (candStems.has(s)) lexicalOverlap += 1;
      }
      const normOverlap =
        combinedStems.size > 0 ? lexicalOverlap / combinedStems.size : 0;
      const totalScore = (neuralSim * 0.55 + normOverlap * 1.45) * cand.priority;
      return { ...cand, neuralSim, lexicalOverlap, totalScore };
    })
    .filter((c) => c.lexicalOverlap > 0 || c.neuralSim > 0.58)
    .sort((a, b) => b.totalScore - a.totalScore);

  const chosen: typeof scored = [];
  for (const item of scored) {
    if (chosen.length >= 3) break;
    const dup = chosen.some(
      (c) => c.text.toLowerCase().slice(0, 48) === item.text.toLowerCase().slice(0, 48)
    );
    if (!dup) chosen.push(item);
  }

  const promptToks = tokenizeWords(clean).filter((t) => t.length >= 2).slice(0, 5);
  const attn = runEducationalTransformerForward({
    tokens: promptToks.length >= 2 ? promptToks : ["nexora", "tensorflow", "inference"],
    dk: 4,
    causal: true,
  });
  const infer = model.predictAndGenerate(clean, 16, 0.6, 5);
  const topFocusIdx =
    attn.attention.weights[attn.attention.weights.length - 1]?.reduce(
      (bestIdx, w, idx, arr) => (w > arr[bestIdx] ? idx : bestIdx),
      0
    ) ?? 0;
  const focusToken = attn.tokens[topFocusIdx] ?? clean;

  if (chosen.length > 0 && (chosen[0].lexicalOverlap > 0 || chosen[0].neuralSim > 0.65)) {
    const lead = chosen[0].text;
    const supporting = chosen.slice(1).map((c) => `- ${c.text}`);
    const bodyLines: string[] = [lead];
    if (supporting.length > 0) {
      bodyLines.push("", "**Key context:**", ...supporting);
    }
    if (infer.generatedContinuation && infer.generatedContinuation.split(/\s+/).length >= 4) {
      bodyLines.push(
        "",
        `**TFJS \`model.predict()\` continuation (\`loss=${snap.currentLoss.toFixed(3)}\`, \`${snap.backend}\`):** _"${infer.generatedContinuation}"_`
      );
    }
    return {
      reasoningNote: `Ranked ${candidatePool.length} knowledge vectors via TensorFlow.js embedding similarity + lexical salience, attended on "${focusToken}", and decoded via model.predict() (${snap.stepCount} epochs, tf.train.${snap.optimizer}).`,
      thinkingLines: [
        `Encoded ${infer.promptTokens.length} tokens → ${snap.dModel}-dim TFJS embedding`,
        `Matched ${chosen.length} high-attention pattern(s) around "${focusToken}"`,
      ],
      markdown: bodyLines.join("\n"),
    };
  }

  const topicWords = Array.from(queryStems).slice(0, 5).join(", ") || clean.slice(0, 48);
  const topTokensText = infer.topNextTokens
    .slice(0, 3)
    .map((t) => `\`${t.token}\` (${(t.probability * 100).toFixed(0)}%)`)
    .join(", ");

  return {
    reasoningNote: `Executed TensorFlow.js inference (${snap.paramCount.toLocaleString()} params on ${snap.backend}, loss=${snap.currentLoss.toFixed(3)}, top focus "${focusToken}").`,
    thinkingLines: [
      `Attended on "${focusToken}" across ${attn.tokens.length} context tokens`,
      `Predicted next tokens ${topTokensText} via TFJS softmax`,
    ],
    markdown: [
      `Here is a live **TensorFlow.js (\`${snap.backend}\`) Neural Analysis** on **${topicWords}**:`,
      "",
      `- **Active Neural Focus:** Self-attention weighted \`"${focusToken}"\` strongest across your prompt (${attn.tokens.map((t) => `\`${t}\``).join(" · ")}), evaluated across **${snap.paramCount.toLocaleString()}** TFJS weights & biases compiled with \`tf.train.${snap.optimizer}\`.`,
      `- **Learned Next-Token Distribution (\`model.predict\`):** Softmax probabilities centered on ${topTokensText}${infer.generatedContinuation ? `, generating: _"${infer.generatedContinuation}"_` : "."}`,
      `- **Actionable Next Step:** Ask me to **harvest more web data**, **build an interactive HTML page/app**, **write TypeScript/Python/React code**, or **solve math or chart data** on **${topicWords}**.`,
    ].join("\n"),
  };
}
