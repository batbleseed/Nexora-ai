/* ───────────────────────────────────────────────────────────────────────────
 * Pure TypeScript Micrograd Engine (Zero External Dependencies)
 * Scalar reverse-mode automatic differentiation + Neuron / Layer / MLP
 * ─────────────────────────────────────────────────────────────────────────── */

export interface GraphNodeSnapshot {
  id: string;
  label: string;
  op: string;
  data: number;
  grad: number;
  depth: number;
  lane: number;
}

export interface GraphEdgeSnapshot {
  fromId: string;
  toId: string;
}

export interface ComputationGraphSnapshot {
  nodes: GraphNodeSnapshot[];
  edges: GraphEdgeSnapshot[];
  topoOrder: string[];
}

let nextValueSeq = 1;

export class Value {
  readonly id: string;
  label: string;
  readonly op: string;
  data: number;
  grad: number;
  readonly _prev: Value[];
  _backward: () => void;

  constructor(data: number, children: Value[] = [], op = "", label = "") {
    this.id = `v_${nextValueSeq++}`;
    this.data = data;
    this.grad = 0;
    this._prev = children;
    this.op = op;
    this.label = label;
    this._backward = () => {};
  }

  private static wrap(other: Value | number, label?: string): Value {
    return other instanceof Value
      ? other
      : new Value(other, [], "", label ?? String(Number(other.toFixed(4))));
  }

  add(other: Value | number, label = ""): Value {
    const rhs = Value.wrap(other);
    const out = new Value(this.data + rhs.data, [this, rhs], "+", label);
    out._backward = () => {
      this.grad += 1 * out.grad;
      rhs.grad += 1 * out.grad;
    };
    return out;
  }

  mul(other: Value | number, label = ""): Value {
    const rhs = Value.wrap(other);
    const out = new Value(this.data * rhs.data, [this, rhs], "×", label);
    out._backward = () => {
      this.grad += rhs.data * out.grad;
      rhs.grad += this.data * out.grad;
    };
    return out;
  }

  pow(exponent: number, label = ""): Value {
    const out = new Value(Math.pow(this.data, exponent), [this], `^${exponent}`, label);
    out._backward = () => {
      this.grad += exponent * Math.pow(this.data, exponent - 1) * out.grad;
    };
    return out;
  }

  neg(label = ""): Value {
    return this.mul(-1, label);
  }

  sub(other: Value | number, label = ""): Value {
    const rhs = Value.wrap(other);
    const out = new Value(this.data - rhs.data, [this, rhs], "−", label);
    out._backward = () => {
      this.grad += 1 * out.grad;
      rhs.grad += -1 * out.grad;
    };
    return out;
  }

  div(other: Value | number, label = ""): Value {
    const rhs = Value.wrap(other);
    return this.mul(rhs.pow(-1), label);
  }

  tanh(label = ""): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], "tanh", label);
    out._backward = () => {
      this.grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  relu(label = ""): Value {
    const activated = this.data > 0 ? this.data : 0;
    const out = new Value(activated, [this], "ReLU", label);
    out._backward = () => {
      this.grad += (out.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  exp(label = ""): Value {
    const e = Math.exp(this.data);
    const out = new Value(e, [this], "exp", label);
    out._backward = () => {
      this.grad += e * out.grad;
    };
    return out;
  }

  sigmoid(label = ""): Value {
    const s = 1 / (1 + Math.exp(-this.data));
    const out = new Value(s, [this], "σ", label);
    out._backward = () => {
      this.grad += s * (1 - s) * out.grad;
    };
    return out;
  }

  /**
   * Reverse-mode automatic differentiation:
   * 1) Topological sort of computation graph
   * 2) Seed output gradient = 1
   * 3) Propagate local chain-rule derivatives backward
   */
  backward(): Value[] {
    const topo: Value[] = [];
    const visited = new Set<Value>();

    const buildTopo = (node: Value) => {
      if (visited.has(node)) return;
      visited.add(node);
      for (const child of node._prev) {
        buildTopo(child);
      }
      topo.push(node);
    };

    buildTopo(this);
    for (const node of topo) {
      node.grad = 0;
    }
    this.grad = 1;

    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
    return topo;
  }
}

export function createSeededRng(seed = 42): () => number {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

export class Neuron {
  readonly w: Value[];
  readonly b: Value;
  readonly nonLin: boolean;

  constructor(nin: number, nonLin = true, rng: () => number = Math.random) {
    this.w = Array.from({ length: nin }, () => new Value(rng() * 2 - 1));
    this.b = new Value(0);
    this.nonLin = nonLin;
  }

  forward(x: (Value | number)[]): Value {
    let act = this.b;
    for (let i = 0; i < this.w.length; i++) {
      const xi = x[i] instanceof Value ? (x[i] as Value) : new Value(x[i] as number);
      act = act.add(this.w[i].mul(xi));
    }
    return this.nonLin ? act.tanh() : act;
  }

  parameters(): Value[] {
    return [...this.w, this.b];
  }
}

export class Layer {
  readonly neurons: Neuron[];

  constructor(nin: number, nout: number, nonLin = true, rng: () => number = Math.random) {
    this.neurons = Array.from({ length: nout }, () => new Neuron(nin, nonLin, rng));
  }

  forward(x: (Value | number)[]): Value[] {
    return this.neurons.map((n) => n.forward(x));
  }

  parameters(): Value[] {
    return this.neurons.flatMap((n) => n.parameters());
  }
}

export class MLP {
  readonly layers: Layer[];

  constructor(nin: number, nouts: number[], seed = 1337) {
    const rng = createSeededRng(seed);
    const sizes = [nin, ...nouts];
    this.layers = nouts.map(
      (nout, i) => new Layer(sizes[i], nout, i !== nouts.length - 1, rng)
    );
  }

  forward(x: (Value | number)[]): Value[] {
    let out: (Value | number)[] = x;
    for (const layer of this.layers) {
      out = layer.forward(out);
    }
    return out as Value[];
  }

  parameters(): Value[] {
    return this.layers.flatMap((l) => l.parameters());
  }

  zeroGrad(): void {
    for (const p of this.parameters()) {
      p.grad = 0;
    }
  }

  step(learningRate: number): void {
    for (const p of this.parameters()) {
      p.data -= learningRate * p.grad;
    }
  }
}

export function extractComputationGraph(root: Value): ComputationGraphSnapshot {
  const topo: Value[] = [];
  const visited = new Set<Value>();
  const depthMap = new Map<Value, number>();

  const buildTopo = (node: Value): number => {
    if (visited.has(node)) {
      return depthMap.get(node) ?? 0;
    }
    visited.add(node);
    let maxChildDepth = -1;
    for (const child of node._prev) {
      maxChildDepth = Math.max(maxChildDepth, buildTopo(child));
    }
    const depth = node._prev.length === 0 ? 0 : maxChildDepth + 1;
    depthMap.set(node, depth);
    topo.push(node);
    return depth;
  };

  buildTopo(root);

  const laneCounters = new Map<number, number>();
  const nodes: GraphNodeSnapshot[] = topo.map((v) => {
    const depth = depthMap.get(v) ?? 0;
    const lane = laneCounters.get(depth) ?? 0;
    laneCounters.set(depth, lane + 1);
    return {
      id: v.id,
      label: v.label || (v.op ? v.op : "const"),
      op: v.op,
      data: v.data,
      grad: v.grad,
      depth,
      lane,
    };
  });

  const edges: GraphEdgeSnapshot[] = [];
  for (const v of topo) {
    for (const child of v._prev) {
      edges.push({ fromId: child.id, toId: v.id });
    }
  }

  return {
    nodes,
    edges,
    topoOrder: topo.map((n) => n.label || n.op || n.id),
  };
}

export interface NeuronDemoResult {
  rawSum: number;
  prediction: number;
  loss: number;
  grads: {
    x1: number;
    w1: number;
    x2: number;
    w2: number;
    b: number;
    rawSum: number;
    prediction: number;
  };
  graph: ComputationGraphSnapshot;
}

export function runSingleNeuronDemo(params: {
  x1: number;
  w1: number;
  x2: number;
  w2: number;
  b: number;
  target: number;
}): NeuronDemoResult {
  const x1 = new Value(params.x1, [], "", "x1");
  const w1 = new Value(params.w1, [], "", "w1");
  const x2 = new Value(params.x2, [], "", "x2");
  const w2 = new Value(params.w2, [], "", "w2");
  const b = new Value(params.b, [], "", "b");

  const x1w1 = x1.mul(w1, "x1·w1");
  const x2w2 = x2.mul(w2, "x2·w2");
  const sum = x1w1.add(x2w2, "x1w1+x2w2");
  const raw = sum.add(b, "n");
  const pred = raw.tanh("ŷ");
  const targetNode = new Value(params.target, [], "", "y");
  const diff = pred.sub(targetNode, "ŷ−y");
  const loss = diff.pow(2, "Loss");

  loss.backward();

  return {
    rawSum: raw.data,
    prediction: pred.data,
    loss: loss.data,
    grads: {
      x1: x1.grad,
      w1: w1.grad,
      x2: x2.grad,
      w2: w2.grad,
      b: b.grad,
      rawSum: raw.grad,
      prediction: pred.grad,
    },
    graph: extractComputationGraph(loss),
  };
}

export interface XorTrainingSnapshot {
  epoch: number;
  loss: number;
  lossHistory: number[];
  predictions: { inputs: [number, number]; target: number; output: number }[];
  paramCount: number;
}

export class XorMlpTrainer {
  private readonly mlp: MLP;
  private epoch = 0;
  private lossHistory: number[] = [];
  private readonly dataset: { inputs: [number, number]; target: number }[] = [
    { inputs: [0, 0], target: -1 },
    { inputs: [0, 1], target: 1 },
    { inputs: [1, 0], target: 1 },
    { inputs: [1, 1], target: -1 },
  ];

  constructor(seed = 1337) {
    this.mlp = new MLP(2, [4, 1], seed);
  }

  evaluate(): XorTrainingSnapshot {
    let totalLoss = 0;
    const predictions = this.dataset.map((row) => {
      const [pred] = this.mlp.forward(row.inputs);
      const err = pred.data - row.target;
      totalLoss += err * err;
      return {
        inputs: row.inputs,
        target: row.target,
        output: pred.data,
      };
    });
    const avgLoss = totalLoss / this.dataset.length;
    return {
      epoch: this.epoch,
      loss: avgLoss,
      lossHistory:
        this.lossHistory.length > 0 ? [...this.lossHistory] : [avgLoss],
      predictions,
      paramCount: this.mlp.parameters().length,
    };
  }

  stepEpochs(count = 1, learningRate = 0.14): XorTrainingSnapshot {
    for (let s = 0; s < count; s++) {
      let lossNode = new Value(0);
      for (const row of this.dataset) {
        const [pred] = this.mlp.forward(row.inputs);
        const diff = pred.sub(row.target);
        lossNode = lossNode.add(diff.pow(2));
      }
      const meanLoss = lossNode.mul(1 / this.dataset.length);
      this.mlp.zeroGrad();
      meanLoss.backward();
      this.mlp.step(learningRate);
      this.epoch += 1;
      this.lossHistory.push(meanLoss.data);
      if (this.lossHistory.length > 120) {
        this.lossHistory = this.lossHistory.slice(-120);
      }
    }
    return this.evaluate();
  }
}
