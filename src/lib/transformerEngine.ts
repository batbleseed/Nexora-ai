/* ───────────────────────────────────────────────────────────────────────────
 * Pure TypeScript Matrix & Transformer Attention Engine (Zero External Libs)
 * Implements matrix multiplication, scaled dot-product attention, causal masking,
 * softmax, LayerNorm, GELU, and a transparent Transformer forward pass.
 * ─────────────────────────────────────────────────────────────────────────── */

export type Matrix = number[][];

export interface MatMulTerm {
  k: number;
  aVal: number;
  bVal: number;
  product: number;
}

export interface MatMulCellTrace {
  row: number;
  col: number;
  terms: MatMulTerm[];
  sum: number;
}

export interface MatMulTraceResult {
  result: Matrix;
  cells: MatMulCellTrace[][];
}

export function zeros(rows: number, cols: number): Matrix {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0));
}

export function transpose(m: Matrix): Matrix {
  const rows = m.length;
  const cols = m[0]?.length ?? 0;
  const out = zeros(cols, rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      out[c][r] = m[r][c];
    }
  }
  return out;
}

export function addMatrices(a: Matrix, b: Matrix): Matrix {
  return a.map((row, r) => row.map((val, c) => val + (b[r]?.[c] ?? 0)));
}

export function scaleMatrix(m: Matrix, scalar: number): Matrix {
  return m.map((row) => row.map((val) => val * scalar));
}

export function matMul(a: Matrix, b: Matrix): Matrix {
  return matMulWithTrace(a, b).result;
}

export function matMulWithTrace(a: Matrix, b: Matrix): MatMulTraceResult {
  const rowsA = a.length;
  const colsA = a[0]?.length ?? 0;
  const rowsB = b.length;
  const colsB = b[0]?.length ?? 0;
  if (colsA !== rowsB) {
    throw new Error(`Shape mismatch: (${rowsA}x${colsA}) cannot multiply (${rowsB}x${colsB})`);
  }

  const result = zeros(rowsA, colsB);
  const cells: MatMulCellTrace[][] = Array.from({ length: rowsA }, () => []);

  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      let sum = 0;
      const terms: MatMulTerm[] = [];
      for (let k = 0; k < colsA; k++) {
        const aVal = a[i][k];
        const bVal = b[k][j];
        const product = aVal * bVal;
        sum += product;
        terms.push({ k, aVal, bVal, product });
      }
      result[i][j] = sum;
      cells[i][j] = { row: i, col: j, terms, sum };
    }
  }

  return { result, cells };
}

export function softmaxRow(row: number[]): number[] {
  const finite = row.filter((v) => Number.isFinite(v));
  const maxVal = finite.length > 0 ? Math.max(...finite) : 0;
  const exps = row.map((v) => (Number.isFinite(v) ? Math.exp(v - maxVal) : 0));
  const total = exps.reduce((acc, v) => acc + v, 0);
  if (total <= 0) {
    return row.map(() => 0);
  }
  return exps.map((e) => e / total);
}

export function softmaxRows(m: Matrix): Matrix {
  return m.map((row) => softmaxRow(row));
}

export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}

export function layerNormRow(row: number[], eps = 1e-5): number[] {
  if (row.length === 0) return [];
  const mean = row.reduce((acc, v) => acc + v, 0) / row.length;
  const variance = row.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) / row.length;
  const invStd = 1 / Math.sqrt(variance + eps);
  return row.map((v) => (v - mean) * invStd);
}

export interface AttentionTrace {
  Q: Matrix;
  K: Matrix;
  V: Matrix;
  KT: Matrix;
  dk: number;
  scaleFactor: number;
  rawScores: Matrix;
  scaledScores: Matrix;
  maskedScores: Matrix;
  weights: Matrix;
  output: Matrix;
}

export function scaledDotProductAttention(params: {
  Q: Matrix;
  K: Matrix;
  V: Matrix;
  causal: boolean;
}): AttentionTrace {
  const { Q, K, V, causal } = params;
  const dk = K[0]?.length ?? 1;
  const scaleFactor = 1 / Math.sqrt(dk);
  const KT = transpose(K);
  const rawScores = matMul(Q, KT);
  const scaledScores = scaleMatrix(rawScores, scaleFactor);
  const maskedScores = scaledScores.map((row, r) =>
    row.map((val, c) => (causal && c > r ? -Infinity : val))
  );
  const weights = softmaxRows(maskedScores);
  const output = matMul(weights, V);

  return {
    Q,
    K,
    V,
    KT,
    dk,
    scaleFactor,
    rawScores,
    scaledScores,
    maskedScores,
    weights,
    output,
  };
}

function deterministicWeightMatrix(rows: number, cols: number, seedOffset: number): Matrix {
  return Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => {
      const angle = (r + 1) * 1.37 + (c + 1) * 2.11 + seedOffset * 3.73;
      return Number((Math.sin(angle) * 0.72).toFixed(3));
    })
  );
}

export function encodeTokensToEmbeddings(tokens: string[], dModel = 4): Matrix {
  return tokens.map((token, pos) => {
    let hash = 0;
    for (let i = 0; i < token.length; i++) {
      hash = (hash * 31 + token.charCodeAt(i)) % 997;
    }
    return Array.from({ length: dModel }, (_, d) => {
      const sem = Math.sin((hash + 1) * (d + 1) * 0.19) * 0.65;
      const posEnc =
        d % 2 === 0
          ? Math.sin(pos / Math.pow(10, (2 * d) / dModel)) * 0.35
          : Math.cos(pos / Math.pow(10, (2 * (d - 1)) / dModel)) * 0.35;
      return Number((sem + posEnc).toFixed(3));
    });
  });
}

export interface TransformerForwardResult {
  tokens: string[];
  dModel: number;
  dk: number;
  causal: boolean;
  X: Matrix;
  WQ: Matrix;
  WK: Matrix;
  WV: Matrix;
  attention: AttentionTrace;
  postAttnNorm: Matrix;
  ffnOut: Matrix;
}

export function runEducationalTransformerForward(params: {
  tokens: string[];
  causal: boolean;
  dModel?: number;
  dk?: number;
}): TransformerForwardResult {
  const tokens = params.tokens.length > 0 ? params.tokens : ["The", "cat", "sat"];
  const dModel = params.dModel ?? 4;
  const dk = params.dk ?? 3;

  const X = encodeTokensToEmbeddings(tokens, dModel);
  const WQ = deterministicWeightMatrix(dModel, dk, 1);
  const WK = deterministicWeightMatrix(dModel, dk, 2);
  const WV = deterministicWeightMatrix(dModel, dModel, 3);

  const Q = matMul(X, WQ);
  const K = matMul(X, WK);
  const V = matMul(X, WV);

  const attention = scaledDotProductAttention({ Q, K, V, causal: params.causal });
  const residual1 = addMatrices(X, attention.output);
  const postAttnNorm = residual1.map((row) => layerNormRow(row));

  // Tiny 2-layer FFN with GELU activation + residual
  const W1 = deterministicWeightMatrix(dModel, dModel * 2, 4);
  const W2 = deterministicWeightMatrix(dModel * 2, dModel, 5);
  const hidden = matMul(postAttnNorm, W1).map((row) => row.map(gelu));
  const ffnProjected = matMul(hidden, W2);
  const ffnOut = addMatrices(postAttnNorm, ffnProjected).map((row) => layerNormRow(row));

  return {
    tokens,
    dModel,
    dk,
    causal: params.causal,
    X,
    WQ,
    WK,
    WV,
    attention,
    postAttnNorm,
    ffnOut,
  };
}
