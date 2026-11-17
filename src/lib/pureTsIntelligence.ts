/* ───────────────────────────────────────────────────────────────────────────
 * Pure TypeScript Cognitive & Algorithmic Engine (Zero External Libraries)
 * Implements:
 *  - BM25 + bigram + Jaccard sentence ranker for multi-source web/URL synthesis
 *  - TextRank (cosine-graph power iteration) extractive text summarizer
 *  - Symbolic linear/quadratic equation solver & polynomial calculus (d/dx)
 *  - Dataset statistical analyzer (mean, median, stdDev, R² trend) + SVG charts
 *  - Code analyzer (structure, complexity, lint hints) & algorithm synthesizer
 * ─────────────────────────────────────────────────────────────────────────── */

export interface ResearchPassage {
  title: string;
  url: string;
  text: string;
  authority: number; // higher = higher trust/priority (e.g. direct URL / Wikipedia intro)
}

export interface ResearchSourceCitation {
  title: string;
  url: string;
}

const STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
  "could", "did", "do", "does", "doing", "done", "for", "from", "had", "has",
  "have", "having", "he", "her", "here", "hers", "him", "his", "how", "i",
  "if", "in", "into", "is", "it", "its", "just", "me", "might", "more", "most",
  "my", "no", "not", "of", "on", "or", "other", "our", "out", "over", "own",
  "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
  "them", "then", "there", "these", "they", "this", "those", "through", "to",
  "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
  "where", "which", "while", "who", "whom", "why", "will", "with", "would",
  "you", "your", "please", "explain", "tell", "about", "define", "give",
]);

function lightStem(word: string): string {
  if (word.length <= 3) return word;
  return word
    .replace(/(ations?|ization|ational)$/i, "")
    .replace(/(ing|ed|es|s|ly|ment|ness|ity|ies)$/i, (m) =>
      m === "ies" ? "y" : ""
    );
}

export function tokenizeKeywords(text: string): string[] {
  const words = text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length >= 2 && !STOPWORDS.has(w));
  return words.map(lightStem).filter((w) => w.length >= 2);
}

function splitSentences(text: string): string[] {
  return text
    .replace(/\r\n/g, "\n")
    .split(/(?<=[.!?])\s+|\n+/)
    .map((s) => s.replace(/\s+/g, " ").trim())
    .filter((s) => s.length >= 28 && s.length <= 520 && /[a-zA-Z]/.test(s));
}

function jaccardSimilarity(a: string[], b: string[]): number {
  if (a.length === 0 || b.length === 0) return 0;
  const setA = new Set(a);
  const setB = new Set(b);
  let inter = 0;
  for (const w of setA) {
    if (setB.has(w)) inter += 1;
  }
  const union = setA.size + setB.size - inter;
  return union > 0 ? inter / union : 0;
}

const PROMO_OR_BOILERPLATE_RE =
  /\b(cookie|privacy policy|terms of service|sign up|subscribe|javascript|all rights reserved|earliest backers|redefining luxury|branded residences|early-stage investing|high profile investments|founded by visionary|leading provider of|click here to|contact our sales)\b/i;

// ---------------------------------------------------------------------------
// 1. BM25 + Bigram Multi-Source Research Synthesizer
// ---------------------------------------------------------------------------
export function synthesizeResearchWithBM25(
  query: string,
  passages: ResearchPassage[]
): {
  markdown: string;
  citations: ResearchSourceCitation[];
  reasoningNote: string;
} | null {
  const queryTerms = tokenizeKeywords(query);
  if (passages.length === 0) return null;

  const isDefinitionQuery =
    /^(what\s+(?:is|are|does)|what's|define|definition\s+of|meaning\s+of)\b/i.test(
      query.trim()
    );

  // If a high-authority dictionary definition passage is present, lead directly with clean senses
  const dictPassage = passages.find(
    (p) => p.authority >= 1.8 && /dictionary definition/i.test(p.title)
  );
  if (dictPassage && (isDefinitionQuery || queryTerms.length <= 2)) {
    const senses = dictPassage.text
      .split("\n")
      .map((s) => s.trim())
      .filter((s) => s.length >= 12);

    if (senses.length > 0) {
      const displayTerm = dictPassage.title.replace(/\s*—\s*Dictionary Definition$/i, "");
      const formattedSenses = senses.map((s) => {
        const posMatch = /^\(([^)]+)\)\s*([\s\S]+)$/.exec(s);
        return posMatch
          ? `- **${posMatch[1]}:** ${posMatch[2]}`
          : `- ${s}`;
      });

      const wikiPassages = passages.filter(
        (p) => p !== dictPassage && /wikipedia/i.test(p.title) && p.authority >= 1.2
      );
      const citations: ResearchSourceCitation[] = [
        { title: dictPassage.title, url: dictPassage.url },
        ...wikiPassages.slice(0, 2).map((w) => ({ title: w.title, url: w.url })),
      ];

      const mdLines: string[] = [
        `### ${displayTerm}`,
        "",
        formattedSenses[0].replace(/^-\s*/, ""),
      ];
      if (formattedSenses.length > 1) {
        mdLines.push("", "**Other meanings & uses:**", ...formattedSenses.slice(1));
      }

      return {
        markdown: mdLines.join("\n"),
        citations,
        reasoningNote: `Resolved exact dictionary senses (${senses.length} meanings) via authoritative lexical lookup and filtered noisy web snippets.`,
      };
    }
  }

  interface CandidateSentence {
    sentence: string;
    tokens: string[];
    title: string;
    url: string;
    authority: number;
    score: number;
  }

  const candidates: CandidateSentence[] = [];
  for (const p of passages) {
    const sentences = splitSentences(p.text);
    for (let idx = 0; idx < sentences.length; idx++) {
      const s = sentences[idx];
      if (PROMO_OR_BOILERPLATE_RE.test(s)) {
        continue;
      }
      const tokens = tokenizeKeywords(s);
      if (tokens.length < 4) continue;
      // Earlier sentences on authoritative sources (Wikipedia lead / page intro) get position weight
      const posWeight = idx === 0 ? 1.35 : idx <= 2 ? 1.12 : 1.0;
      const defWeight =
        isDefinitionQuery && /\b(?:is|are|refers to|means|denotes)\b/i.test(s)
          ? 1.25
          : 1.0;
      candidates.push({
        sentence: s,
        tokens,
        title: p.title,
        url: p.url,
        authority: p.authority * posWeight * defWeight,
        score: 0,
      });
    }
  }

  if (candidates.length === 0) return null;

  const N = candidates.length;
  const avgLen =
    candidates.reduce((sum, c) => sum + c.tokens.length, 0) / Math.max(1, N);
  const df = new Map<string, number>();
  for (const c of candidates) {
    const unique = new Set(c.tokens);
    for (const term of unique) {
      df.set(term, (df.get(term) ?? 0) + 1);
    }
  }

  const k1 = 1.4;
  const b = 0.65;
  const effectiveQueryTerms =
    queryTerms.length > 0 ? queryTerms : tokenizeKeywords(passages[0].title);

  const queryBigrams = new Set<string>();
  for (let i = 0; i < effectiveQueryTerms.length - 1; i++) {
    queryBigrams.add(`${effectiveQueryTerms[i]}_${effectiveQueryTerms[i + 1]}`);
  }

  for (const c of candidates) {
    const tf = new Map<string, number>();
    for (const t of c.tokens) {
      tf.set(t, (tf.get(t) ?? 0) + 1);
    }
    let bm25 = 0;
    for (const q of effectiveQueryTerms) {
      const f = tf.get(q) ?? 0;
      if (f === 0) continue;
      const docFreq = df.get(q) ?? 0;
      const idf = Math.log(1 + (N - docFreq + 0.5) / (docFreq + 0.5));
      bm25 += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + b * (c.tokens.length / avgLen))));
    }

    let bigramBonus = 0;
    for (let i = 0; i < c.tokens.length - 1; i++) {
      if (queryBigrams.has(`${c.tokens[i]}_${c.tokens[i + 1]}`)) {
        bigramBonus += 0.9;
      }
    }

    // Penalize ultra-short fragments slightly
    const lenFactor = Math.min(1.15, Math.max(0.75, c.tokens.length / 12));
    c.score = (bm25 + bigramBonus + 0.15) * c.authority * lenFactor;
  }

  candidates.sort((a, b) => b.score - a.score);

  const selected: CandidateSentence[] = [];
  for (const cand of candidates) {
    if (selected.length >= 5) break;
    const isDuplicate = selected.some(
      (s) => jaccardSimilarity(s.tokens, cand.tokens) > 0.48
    );
    if (!isDuplicate) {
      selected.push(cand);
    }
  }

  if (selected.length === 0) return null;

  const citationMap = new Map<string, ResearchSourceCitation>();
  for (const s of selected) {
    if (s.url && !citationMap.has(s.url)) {
      citationMap.set(s.url, { title: s.title, url: s.url });
    }
  }
  for (const p of passages) {
    if (p.url && !citationMap.has(p.url) && citationMap.size < 5) {
      citationMap.set(p.url, { title: p.title, url: p.url });
    }
  }

  const lead = selected[0];
  const second = selected[1];
  const supporting = second && jaccardSimilarity(lead.tokens, second.tokens) < 0.35
    ? selected.slice(2)
    : selected.slice(1);
  const overviewText =
    second && jaccardSimilarity(lead.tokens, second.tokens) < 0.35
      ? `${lead.sentence} ${second.sentence}`
      : lead.sentence;
  const cleanHeading = (lead.title || "Overview")
    .replace(/\s*—\s*(Wikipedia|Dictionary Definition)$/i, "")
    .trim();
  const citations = Array.from(citationMap.values());

  const mdLines: string[] = [
    `### ${cleanHeading}`,
    "",
    overviewText,
  ];

  if (supporting.length > 0) {
    mdLines.push("", "**Key Context & Details:**");
    for (const item of supporting) {
      mdLines.push(`- ${item.sentence}`);
    }
  }

  return {
    markdown: mdLines.join("\n"),
    citations,
    reasoningNote: `Scored ${candidates.length} candidate sentences across ${passages.length} source(s) via Pure-TS BM25 (k1=${k1}, b=${b}) + bigram matching and selected ${selected.length} non-redundant facts.`,
  };
}

// ---------------------------------------------------------------------------
// 2. TextRank Graph Extractive Summarizer (Power Iteration Cosine Centrality)
// ---------------------------------------------------------------------------
export function tryTextRankSummary(prompt: string): {
  markdown: string;
  reasoningNote: string;
  titleHint: string;
} | null {
  const summarizeMatch =
    /^(?:summarize|summarise|tl;?dr|key\s+points?|extract\s+takeaways?|shorten|bullet\s+points?)\s*[:\-\n]+\s*([\s\S]+)$/i.exec(
      prompt.trim()
    );
  const body = summarizeMatch ? summarizeMatch[1].trim() : prompt.trim();
  const isExplicitSummary = Boolean(summarizeMatch);
  const hasUrl = /https?:\/\//i.test(body);
  if (hasUrl && body.length < 180) return null;

  if (!isExplicitSummary && body.length < 320) return null;
  if (
    !isExplicitSummary &&
    !/\b(summarize|summarise|key\s+points?|takeaways|tl;?dr)\b/i.test(prompt)
  ) {
    return null;
  }

  const sentences = splitSentences(body);
  if (sentences.length < 3) return null;

  const tokenized = sentences.map((s) => tokenizeKeywords(s));
  const n = sentences.length;
  const matrix: number[][] = Array.from({ length: n }, () => Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const sim = jaccardSimilarity(tokenized[i], tokenized[j]);
      matrix[i][j] = sim;
      matrix[j][i] = sim;
    }
  }

  // Normalize rows + 15 iterations of TextRank damping
  let scores = Array(n).fill(1 / n);
  const d = 0.85;
  for (let iter = 0; iter < 15; iter++) {
    const next = Array(n).fill((1 - d) / n);
    for (let i = 0; i < n; i++) {
      const rowSum = matrix[i].reduce((a, b) => a + b, 0);
      if (rowSum <= 0) continue;
      for (let j = 0; j < n; j++) {
        next[j] += d * scores[i] * (matrix[i][j] / rowSum);
      }
    }
    scores = next;
  }

  const keepCount = Math.min(4, Math.max(2, Math.ceil(n * 0.35)));
  const rankedIndices = scores
    .map((score, idx) => ({ score, idx }))
    .sort((a, b) => b.score - a.score)
    .slice(0, keepCount)
    .map((x) => x.idx)
    .sort((a, b) => a - b);

  const wordCount = body.split(/\s+/).filter(Boolean).length;
  const selectedSentences = rankedIndices.map((i) => sentences[i]);

  return {
    titleHint: "TextRank Extractive Summary",
    reasoningNote: `Built ${n}×${n} Jaccard sentence similarity graph and ran 15 TextRank power iterations to extract top ${keepCount} sentences from ${wordCount} words.`,
    markdown: [
      `### Key Takeaways (${wordCount} words condensed to ${keepCount} key points)`,
      "",
      ...selectedSentences.map((s) => `- ${s}`),
    ].join("\n"),
  };
}

// ---------------------------------------------------------------------------
// 3. Symbolic Algebra (Linear, Quadratic), Derivative & Number Theory Solver
// ---------------------------------------------------------------------------
function formatNumClean(n: number): string {
  if (!Number.isFinite(n)) return String(n);
  if (Math.abs(n) < 1e-10) return "0";
  if (Number.isInteger(n)) return String(n);
  return Number(n.toFixed(6)).toString();
}

function parseLinearSide(expr: string): { coeffX: number; constant: number } | null {
  const clean = expr.replace(/\s+/g, "").replace(/−/g, "-");
  if (!clean || /[^0-9xX.+\-*/()]/.test(clean)) return null;
  if (/[*/^]/.test(clean.replace(/x\^1/gi, "x"))) {
    // Allow simple parenthesized distribution like 2(x+3)
    const distMatch = /^([+-]?\d*\.?\d*)\(([^()]+)\)$/.exec(clean);
    if (!distMatch) return null;
    const factor =
      distMatch[1] === "" || distMatch[1] === "+"
        ? 1
        : distMatch[1] === "-"
          ? -1
          : Number(distMatch[1]);
    const inner = parseLinearSide(distMatch[2]);
    if (!Number.isFinite(factor) || !inner) return null;
    return {
      coeffX: factor * inner.coeffX,
      constant: factor * inner.constant,
    };
  }

  // Split signed terms
  const expanded = clean.replace(
    /([+-]?\d*\.?\d*)\(([^()]+)\)/g,
    (_m, fStr: string, inside: string) => {
      const f =
        fStr === "" || fStr === "+" ? 1 : fStr === "-" ? -1 : Number(fStr);
      const inner = parseLinearSide(inside);
      if (!Number.isFinite(f) || !inner) return "INVALID";
      const cx = f * inner.coeffX;
      const c = f * inner.constant;
      return `${cx >= 0 ? "+" : ""}${cx}x${c >= 0 ? "+" : ""}${c}`;
    }
  );
  if (expanded.includes("INVALID")) return null;

  const terms = expanded.match(/[+-]?[^+-]+/g);
  if (!terms) return null;

  let coeffX = 0;
  let constant = 0;
  for (const rawTerm of terms) {
    const t = rawTerm.toLowerCase();
    if (t.includes("x")) {
      const numPart = t.replace(/x$/, "");
      const c =
        numPart === "" || numPart === "+"
          ? 1
          : numPart === "-"
            ? -1
            : Number(numPart);
      if (!Number.isFinite(c)) return null;
      coeffX += c;
    } else {
      const v = Number(t);
      if (!Number.isFinite(v)) return null;
      constant += v;
    }
  }
  return { coeffX, constant };
}

function gcdNum(a: number, b: number): number {
  let x = Math.abs(Math.trunc(a));
  let y = Math.abs(Math.trunc(b));
  while (y !== 0) {
    const tmp = y;
    y = x % y;
    x = tmp;
  }
  return x || 1;
}

function primeFactors(n: number): number[] {
  let num = Math.abs(Math.trunc(n));
  const factors: number[] = [];
  if (num < 2) return factors;
  while (num % 2 === 0) {
    factors.push(2);
    num /= 2;
  }
  for (let d = 3; d * d <= num; d += 2) {
    while (num % d === 0) {
      factors.push(d);
      num /= d;
    }
  }
  if (num > 1) factors.push(num);
  return factors;
}

export function trySymbolicMathAndLogic(prompt: string): {
  markdown: string;
  reasoningNote: string;
  titleHint: string;
} | null {
  const clean = prompt.trim();

  // A. Quadratic equation: ax^2 + bx + c = 0 (or with RHS)
  const eqMatch =
    /(?:solve\s+(?:for\s+x\s*[:=]?\s*)?)?([0-9xX^+\-.\s*()]+)=([0-9xX^+\-.\s*()]+)/i.exec(
      clean
    );
  if (eqMatch && /x/i.test(clean) && !/==|=>|const |let |var /.test(clean)) {
    const lhsRaw = eqMatch[1].trim();
    const rhsRaw = eqMatch[2].trim();
    if (/x\^2/i.test(lhsRaw) || /x\^2/i.test(rhsRaw)) {
      const parseQuad = (side: string): [number, number, number] | null => {
        const compact = side.replace(/\s+/g, "").replace(/−/g, "-").toLowerCase();
        if (/[^0-9x^+\-.]/.test(compact)) return null;
        const terms = compact.match(/[+-]?[^+-]+/g);
        if (!terms) return null;
        let a = 0, b = 0, c = 0;
        for (const t of terms) {
          if (t.includes("x^2")) {
            const p = t.replace(/x\^2/, "");
            a += p === "" || p === "+" ? 1 : p === "-" ? -1 : Number(p);
          } else if (t.includes("x")) {
            const p = t.replace(/x/, "");
            b += p === "" || p === "+" ? 1 : p === "-" ? -1 : Number(p);
          } else {
            c += Number(t);
          }
        }
        if ([a, b, c].some((v) => !Number.isFinite(v))) return null;
        return [a, b, c];
      };

      const L = parseQuad(lhsRaw);
      const R = parseQuad(rhsRaw);
      if (L && R) {
        const a = L[0] - R[0];
        const b = L[1] - R[1];
        const c = L[2] - R[2];
        if (Math.abs(a) > 1e-9) {
          const disc = b * b - 4 * a * c;
          const steps = [
            `**Standard form:** \`${formatNumClean(a)}x² ${b >= 0 ? "+" : "-"} ${formatNumClean(Math.abs(b))}x ${c >= 0 ? "+" : "-"} ${formatNumClean(Math.abs(c))} = 0\``,
            `**Discriminant:** \`Δ = b² - 4ac = (${formatNumClean(b)})² - 4(${formatNumClean(a)})(${formatNumClean(c)}) = ${formatNumClean(disc)}\``,
          ];
          if (disc > 1e-10) {
            const r1 = (-b + Math.sqrt(disc)) / (2 * a);
            const r2 = (-b - Math.sqrt(disc)) / (2 * a);
            steps.push(
              `**Two real roots:** \`x₁ = ${formatNumClean(r1)}\`, \`x₂ = ${formatNumClean(r2)}\``
            );
          } else if (Math.abs(disc) <= 1e-10) {
            const r = -b / (2 * a);
            steps.push(`**Repeated real root:** \`x = ${formatNumClean(r)}\``);
          } else {
            const real = -b / (2 * a);
            const imag = Math.sqrt(-disc) / (2 * a);
            steps.push(
              `**Complex roots:** \`x = ${formatNumClean(real)} ± ${formatNumClean(Math.abs(imag))}i\``
            );
          }
          return {
            titleHint: `Quadratic: ${lhsRaw} = ${rhsRaw}`.slice(0, 32),
            reasoningNote: `Normalized quadratic to (${a}, ${b}, ${c}) and evaluated discriminant Δ=${formatNumClean(disc)}.`,
            markdown: `### Symbolic Quadratic Solution\n\n${steps.join("\n\n")}`,
          };
        }
      }
    } else {
      // Linear equation: ax + b = cx + d
      const L = parseLinearSide(lhsRaw);
      const R = parseLinearSide(rhsRaw);
      if (L && R) {
        const netX = L.coeffX - R.coeffX;
        const netConst = R.constant - L.constant;
        if (Math.abs(netX) < 1e-10) {
          return {
            titleHint: "Linear Equation Analysis",
            reasoningNote: "Reduced x coefficients on both sides; net x coefficient is 0.",
            markdown:
              Math.abs(netConst) < 1e-10
                ? `The equation \`${lhsRaw} = ${rhsRaw}\` simplifies to \`0 = 0\`, so **all real numbers** \`x\` are solutions.`
                : `The equation \`${lhsRaw} = ${rhsRaw}\` simplifies to \`0 = ${formatNumClean(netConst)}\`, which has **no solution**.`,
          };
        }
        const sol = netConst / netX;
        const checkL = L.coeffX * sol + L.constant;
        const checkR = R.coeffX * sol + R.constant;
        return {
          titleHint: `Solve: ${lhsRaw} = ${rhsRaw}`.slice(0, 32),
          reasoningNote: `Isolated linear terms: (${formatNumClean(netX)})x = ${formatNumClean(netConst)}, verified LHS = RHS.`,
          markdown: [
            "### Step-by-Step Linear Solution",
            "",
            `- **Original:** \`${lhsRaw} = ${rhsRaw}\``,
            `- **Collect terms:** \`${formatNumClean(netX)}x = ${formatNumClean(netConst)}\``,
            `- **Solution:** **\`x = ${formatNumClean(sol)}\`**`,
            `- **Verification:** LHS = \`${formatNumClean(checkL)}\`, RHS = \`${formatNumClean(checkR)}\` ✓`,
          ].join("\n"),
        };
      }
    }
  }

  // B. Polynomial Derivative: "derivative of 3x^3 - 4x^2 + 5x - 7"
  const derivMatch =
    /(?:derivative(?:\s+of)?|differentiate|d\/dx(?:\s*\(?))\s*([0-9xX^+\-.\s]+)\)?/i.exec(
      clean
    );
  if (derivMatch && /x/i.test(derivMatch[1])) {
    const rawPoly = derivMatch[1].replace(/\s+/g, "").replace(/−/g, "-").toLowerCase();
    const terms = rawPoly.match(/[+-]?[^+-]+/g);
    if (terms && !terms.some((t) => /[^0-9x^+\-.]/.test(t))) {
      const derivTerms: string[] = [];
      for (const term of terms) {
        if (!term.includes("x")) continue;
        const powMatch = /^([+-]?\d*\.?\d*)x(?:\^([+-]?\d+))?$/.exec(term);
        if (!powMatch) continue;
        const coeff =
          powMatch[1] === "" || powMatch[1] === "+"
            ? 1
            : powMatch[1] === "-"
              ? -1
              : Number(powMatch[1]);
        const exp = powMatch[2] ? Number(powMatch[2]) : 1;
        if (!Number.isFinite(coeff) || !Number.isFinite(exp)) continue;
        const newCoeff = coeff * exp;
        const newExp = exp - 1;
        if (newCoeff === 0) continue;
        const sign = newCoeff >= 0 ? (derivTerms.length > 0 ? "+ " : "") : "- ";
        const absCoeff = formatNumClean(Math.abs(newCoeff));
        if (newExp === 0) derivTerms.push(`${sign}${absCoeff}`);
        else if (newExp === 1) derivTerms.push(`${sign}${absCoeff === "1" ? "" : absCoeff}x`);
        else derivTerms.push(`${sign}${absCoeff === "1" ? "" : absCoeff}x^${newExp}`);
      }
      if (derivTerms.length > 0) {
        const resultStr = derivTerms.join(" ");
        return {
          titleHint: "Symbolic Derivative",
          reasoningNote: `Applied power rule d/dx(a·xⁿ) = n·a·xⁿ⁻¹ term-by-term to "${derivMatch[1].trim()}".`,
          markdown: [
            "### Symbolic Differentiation",
            "",
            `- **Input:** \`f(x) = ${derivMatch[1].trim()}\``,
            `- **Derivative:** **\`f'(x) = ${resultStr}\`**`,
          ].join("\n"),
        };
      }
    }
  }

  // C. Prime Factorization & Divisors
  const factorMatch =
    /(?:prime\s+factors?(?:\s+of)?|factor(?:ize)?)\s+(\d{2,10})\b/i.exec(clean);
  if (factorMatch) {
    const n = Number(factorMatch[1]);
    if (n >= 2 && n <= 1_000_000_000) {
      const factors = primeFactors(n);
      const counts = new Map<number, number>();
      for (const f of factors) counts.set(f, (counts.get(f) ?? 0) + 1);
      const canonical = Array.from(counts.entries())
        .map(([p, e]) => (e > 1 ? `${p}^${e}` : String(p)))
        .join(" × ");
      return {
        titleHint: `Prime Factors of ${n}`,
        reasoningNote: `Executed trial division up to √${n} in pure TypeScript.`,
        markdown: [
          `### Prime Factorization of **${n.toLocaleString("en-US")}**`,
          "",
          `- **Factors:** \`${factors.join(" × ")}\``,
          `- **Exponential form:** \`${canonical}\``,
          `- **Is prime:** ${factors.length === 1 ? "Yes" : "No"}`,
        ].join("\n"),
      };
    }
  }

  // D. Base Conversions (dec -> hex / binary / octal)
  const baseMatch =
    /(?:convert\s+)?(0x[0-9a-f]+|0b[01]+|\d+)\s+(?:to|in)\s+(hex|hexadecimal|bin|binary|oct|octal|dec|decimal)\b/i.exec(
      clean
    );
  if (baseMatch) {
    const rawNum = baseMatch[1].toLowerCase();
    const target = baseMatch[2].toLowerCase();
    const val = rawNum.startsWith("0x")
      ? Number.parseInt(rawNum.slice(2), 16)
      : rawNum.startsWith("0b")
        ? Number.parseInt(rawNum.slice(2), 2)
        : Number.parseInt(rawNum, 10);
    if (Number.isSafeInteger(val) && val >= 0) {
      const out =
        target.startsWith("hex")
          ? `0x${val.toString(16).toUpperCase()}`
          : target.startsWith("bin")
            ? `0b${val.toString(2)}`
            : target.startsWith("oct")
              ? `0o${val.toString(8)}`
              : String(val);
      return {
        titleHint: `Base Conversion: ${baseMatch[1]}`,
        reasoningNote: `Parsed ${baseMatch[1]} cleanly and converted radix to ${target}.`,
        markdown: [
          `### Base Conversion`,
          "",
          `- **Result:** **\`${out}\`**`,
          `- **Decimal:** \`${val}\` | **Hex:** \`0x${val.toString(16).toUpperCase()}\` | **Binary:** \`0b${val.toString(2)}\` | **Octal:** \`0o${val.toString(8)}\``,
        ].join("\n"),
      };
    }
  }

  // E. GCD / LCM
  const gcdMatch = /\b(gcd|lcm)\s*\(?\s*(\d+)\s*[,\s]+\s*(\d+)\s*\)?/i.exec(clean);
  if (gcdMatch) {
    const fn = gcdMatch[1].toUpperCase();
    const a = Number(gcdMatch[2]);
    const b = Number(gcdMatch[3]);
    const g = gcdNum(a, b);
    const l = Math.round((a * b) / g);
    return {
      titleHint: `${fn}(${a}, ${b})`,
      reasoningNote: `Computed Euclidean GCD(${a}, ${b}) = ${g} and LCM = ${l}.`,
      markdown: `**${fn}(${a}, ${b})** = **\`${fn === "GCD" ? g : l}\`** (GCD = \`${g}\`, LCM = \`${l}\`)`,
    };
  }

  return null;
}

// ---------------------------------------------------------------------------
// 4. Pure-TS Dataset Statistical Analyzer + Interactive SVG Chart Builder
// ---------------------------------------------------------------------------
export function tryStatisticalAnalysisAndChart(prompt: string): {
  markdown: string;
  reasoningNote: string;
  titleHint: string;
} | null {
  const lower = prompt.toLowerCase();
  const wantsStatsOrChart =
    /\b(stats|statistics|analyze|mean|median|std\s*dev|variance|chart|plot|graph|visualize|trend)\b/i.test(
      lower
    );
  if (!wantsStatsOrChart) return null;

  // Support labeled pairs like "Q1=120 Q2=150 Q3=190" or "Jan: 40, Feb: 55"
  const labeledMatches = Array.from(
    prompt.matchAll(/([A-Za-z][A-Za-z0-9_-]{0,14})\s*[:=]\s*(-?\d+(?:\.\d+)?)/g)
  );
  let labels: string[] = [];
  let nums: number[] = [];

  if (labeledMatches.length >= 3) {
    labels = labeledMatches.map((m) => m[1]);
    nums = labeledMatches.map((m) => Number(m[2]));
  } else {
    const rawNums = prompt.match(/-?\d+(?:\.\d+)?/g) ?? [];
    if (rawNums.length < 3 || rawNums.length > 40) return null;
    nums = rawNums.map(Number).filter(Number.isFinite);
    labels = nums.map((_, i) => `#${i + 1}`);
  }

  if (nums.length < 3) return null;

  const n = nums.length;
  const sum = nums.reduce((a, b) => a + b, 0);
  const mean = sum / n;
  const sorted = [...nums].sort((a, b) => a - b);
  const median =
    n % 2 === 1 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  const min = sorted[0];
  const max = sorted[n - 1];
  const variance = nums.reduce((acc, v) => acc + (v - mean) ** 2, 0) / n;
  const stdDev = Math.sqrt(variance);

  // Least-squares linear regression trend
  const meanX = (n - 1) / 2;
  let numCov = 0;
  let denVarX = 0;
  for (let i = 0; i < n; i++) {
    numCov += (i - meanX) * (nums[i] - mean);
    denVarX += (i - meanX) ** 2;
  }
  const slope = denVarX > 0 ? numCov / denVarX : 0;
  const intercept = mean - slope * meanX;
  const ssTot = nums.reduce((acc, v) => acc + (v - mean) ** 2, 0);
  const ssRes = nums.reduce(
    (acc, v, i) => acc + (v - (slope * i + intercept)) ** 2,
    0
  );
  const rSquared = ssTot > 0 ? 1 - ssRes / ssTot : 1;

  // Build responsive SVG inside nexora-preview using workspace tokens
  const width = 560;
  const height = 210;
  const padX = 36;
  const padTop = 20;
  const padBottom = 34;
  const plotW = width - padX * 2;
  const plotH = height - padTop - padBottom;
  const minBar = Math.min(0, min);
  const span = Math.max(1e-6, max - minBar);
  const slotW = plotW / n;
  const barW = Math.max(12, Math.min(44, slotW * 0.62));

  const yFor = (val: number) => padTop + plotH - ((val - minBar) / span) * plotH;

  const barsSvg = nums
    .map((val, i) => {
      const x = padX + i * slotW + (slotW - barW) / 2;
      const yTop = yFor(Math.max(val, 0));
      const yZero = yFor(0);
      const rectY = Math.min(yTop, yZero);
      const rectH = Math.max(2, Math.abs(yZero - yFor(val)));
      return `
        <g>
          <rect x="${x.toFixed(1)}" y="${rectY.toFixed(1)}" width="${barW.toFixed(1)}" height="${rectH.toFixed(1)}" rx="5" fill="var(--accent, #2563eb)" opacity="0.88" />
          <text x="${(x + barW / 2).toFixed(1)}" y="${Math.max(12, rectY - 5).toFixed(1)}" text-anchor="middle" font-size="10" fill="var(--ink, #111827)">${formatNumClean(val)}</text>
          <text x="${(x + barW / 2).toFixed(1)}" y="${height - 10}" text-anchor="middle" font-size="10" fill="var(--ink-2, #6b7280)">${labels[i].replace(/[<>&]/g, "")}</text>
        </g>`;
    })
    .join("");

  const yStart = yFor(intercept);
  const yEnd = yFor(slope * (n - 1) + intercept);
  const xStart = padX + slotW / 2;
  const xEnd = padX + (n - 0.5) * slotW;

  const previewHtml = `<div style="padding:16px 18px;background:var(--surface,#ffffff);color:var(--ink,#1e2126);font-family:'DM Sans',system-ui,-apple-system,sans-serif;border-radius:10px;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:10px;">
    <strong style="font-family:'Plus Jakarta Sans',system-ui,sans-serif;font-size:13.5px;font-weight:600;letter-spacing:-0.015em;">Distribution &amp; Linear Trend</strong>
    <span style="font-family:'JetBrains Mono',ui-monospace,monospace;font-size:11px;font-variant-numeric:tabular-nums;color:var(--ink-2,#626773);">n=${n} · slope ${slope >= 0 ? "+" : ""}${formatNumClean(slope)} · R²=${rSquared.toFixed(3)}</span>
  </div>
  <svg viewBox="0 0 ${width} ${height}" width="100%" style="display:block;overflow:visible;">
    <line x1="${padX}" y1="${yFor(0).toFixed(1)}" x2="${width - padX}" y2="${yFor(0).toFixed(1)}" stroke="var(--line,#eceef1)" stroke-width="1" />
    ${barsSvg}
    <line x1="${xStart.toFixed(1)}" y1="${yStart.toFixed(1)}" x2="${xEnd.toFixed(1)}" y2="${yEnd.toFixed(1)}" stroke="var(--orange,#d97706)" stroke-width="2" stroke-dasharray="4 3" />
  </svg>
</div>`;

  return {
    titleHint: "Dataset Stats & Chart",
    reasoningNote: `Computed sample statistics and least-squares regression (slope=${formatNumClean(slope)}, R²=${rSquared.toFixed(3)}) across ${n} numeric observations in pure TypeScript.`,
    markdown: [
      `### Pure-TS Statistical Summary (${n} points)`,
      "",
      `| Metric | Value | Metric | Value |`,
      `| --- | ---: | --- | ---: |`,
      `| **Sum** | \`${formatNumClean(sum)}\` | **Mean** | \`${formatNumClean(mean)}\` |`,
      `| **Median** | \`${formatNumClean(median)}\` | **Std Dev** | \`${formatNumClean(stdDev)}\` |`,
      `| **Min / Max** | \`${formatNumClean(min)} .. ${formatNumClean(max)}\` | **Trend Slope / R²** | \`${formatNumClean(slope)} (${rSquared.toFixed(3)})\` |`,
      "",
      "```nexora-preview",
      previewHtml,
      "```",
    ].join("\n"),
  };
}

// ---------------------------------------------------------------------------
// 5. Pure-TS Code Assistant, Interactive Web Page Builder & Algorithm Synthesizer
// ---------------------------------------------------------------------------
function normalizeTypoTerms(text: string): string {
  return text
    .replace(/\brestourant\b/gi, "restaurant")
    .replace(/\bresturant\b/gi, "restaurant")
    .replace(/\brestaraunt\b/gi, "restaurant")
    .replace(/\bwebsit\b/gi, "website")
    .replace(/\bchiken\b/gi, "chicken");
}

function extractTopicTitle(prompt: string, fallback: string): string {
  const normalized = normalizeTypoTerms(prompt);

  const cleanExtracted = (raw: string): string => {
    const stripped = raw
      .replace(/^(?:a|an|the|my|our|new|simple|modern|responsive|clean|nice|cool|good|basic|complete|full|interactive)\s+/gi, "")
      .replace(/\b(?:using|with|from|in|html|css|js|javascript|website|web\s*site|webpage|web\s*page|landing\s*page|site|page|app|please)\b.*$/i, "")
      .trim();
    if (stripped.length < 2) return "";
    return stripped
      .split(/\s+/)
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
      .join(" ");
  };

  // 1. Explicit "for / about / called / named / titled <Topic>"
  const aboutMatches = Array.from(
    normalized.matchAll(
      /\b(?:for|about|called|named|titled)\s+(?:a\s+|an\s+|the\s+)?([a-zA-Z0-9][a-zA-Z0-9\s'&-]{1,42})/gi
    )
  );
  for (let i = aboutMatches.length - 1; i >= 0; i--) {
    const candidate = cleanExtracted(aboutMatches[i][1]);
    if (candidate && !/^(html|css|js|website|webpage|page|me)$/i.test(candidate)) {
      return candidate;
    }
  }

  // 2. Action + topic + website/page ("make me a chicken restaurant website")
  const buildMatches = Array.from(
    normalized.matchAll(
      /\b(?:make|create|build|design|write|generate|want)\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?([a-zA-Z0-9][a-zA-Z0-9\s'&-]{1,42}?)\s+(?:website|web\s*site|webpage|web\s*page|landing\s*page|site|page)\b/gi
    )
  );
  for (let i = buildMatches.length - 1; i >= 0; i--) {
    const candidate = cleanExtracted(buildMatches[i][1]);
    if (candidate) return candidate;
  }

  // 3. Topic directly before website/page ("chicken restaurant website")
  const preSiteMatches = Array.from(
    normalized.matchAll(
      /\b([a-zA-Z0-9][a-zA-Z0-9\s'&-]{1,36}?)\s+(?:website|web\s*site|webpage|web\s*page|landing\s*page)\b/gi
    )
  );
  for (let i = preSiteMatches.length - 1; i >= 0; i--) {
    const candidate = cleanExtracted(preSiteMatches[i][1]);
    if (candidate) return candidate;
  }

  return fallback;
}

function escapeHtmlText(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function buildNexoraHeadAssets(
  title: string,
  accentScheme: "cobalt" | "ember" | "espresso" | "emerald" | "violet" = "cobalt"
): string {
  const accentLight =
    accentScheme === "ember"
      ? "#c2410c"
      : accentScheme === "espresso"
        ? "#8b4513"
        : accentScheme === "emerald"
          ? "#059669"
          : accentScheme === "violet"
            ? "#7c3aed"
            : "#2f76ff";
  const accentInkLight =
    accentScheme === "ember"
      ? "#9a3412"
      : accentScheme === "espresso"
        ? "#6f3711"
        : accentScheme === "emerald"
          ? "#047857"
          : accentScheme === "violet"
            ? "#6d28d9"
            : "#2463d8";
  const accentTintLight =
    accentScheme === "ember"
      ? "#fff3ec"
      : accentScheme === "espresso"
        ? "#fdf4ee"
        : accentScheme === "emerald"
          ? "#ecfdf5"
          : accentScheme === "violet"
            ? "#f5f3ff"
            : "#eef4ff";
  const accentDark =
    accentScheme === "ember"
      ? "#f97316"
      : accentScheme === "espresso"
        ? "#d99b66"
        : accentScheme === "emerald"
          ? "#34d399"
          : accentScheme === "violet"
            ? "#a78bfa"
            : "#4c8dff";
  const accentInkDark =
    accentScheme === "ember"
      ? "#fdba74"
      : accentScheme === "espresso"
        ? "#f0c8a4"
        : accentScheme === "emerald"
          ? "#6ee7b7"
          : accentScheme === "violet"
            ? "#c4b5fd"
            : "#8cb6ff";
  const accentTintDark =
    accentScheme === "ember"
      ? "rgba(249, 115, 22, 0.16)"
      : accentScheme === "espresso"
        ? "rgba(217, 155, 102, 0.16)"
        : accentScheme === "emerald"
          ? "rgba(52, 211, 153, 0.16)"
          : accentScheme === "violet"
            ? "rgba(167, 139, 250, 0.16)"
            : "rgba(76, 141, 255, 0.16)";

  return `  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${escapeHtmlText(title)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@500;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --page: #fbfbfc;
      --canvas: #f3f5f8;
      --surface: #ffffff;
      --inset: #f6f7f9;
      --hover: #f1f3f6;
      --hover-2: #e6e9ef;
      --ink: #1e2126;
      --ink-2: #626773;
      --ink-3: #8e94a0;
      --line: #eceef1;
      --line-strong: #dfe2e7;
      --accent: ${accentLight};
      --accent-ink: ${accentInkLight};
      --accent-tint: ${accentTintLight};
      --accent-on: #ffffff;
      --green: #16a34a;
      --green-tint: #ecfdf3;
      --red: #dc2626;
      --red-tint: #fef2f2;
      --font-display: "Plus Jakarta Sans", system-ui, -apple-system, sans-serif;
      --font-sans: "DM Sans", system-ui, -apple-system, sans-serif;
      --font-mono: "JetBrains Mono", ui-monospace, monospace;
      --ease: cubic-bezier(0.23, 1, 0.32, 1);
      --shadow-btn: 0 0 0 1px var(--line-strong), 0 1px 2px rgba(15, 23, 42, 0.04);
      --shadow-card: 0 0 0 1px var(--line), 0 1px 2px rgba(15, 23, 42, 0.04), 0 2px 6px rgba(15, 23, 42, 0.03);
      --shadow-raised: 0 0 0 1px var(--line), 0 4px 12px rgba(15, 23, 42, 0.06), 0 1px 3px rgba(15, 23, 42, 0.04);
      color-scheme: light;
    }
    [data-theme="dark"] {
      --page: #18191d;
      --canvas: #1d1f24;
      --surface: #23252c;
      --inset: #1f2127;
      --hover: #2a2d35;
      --hover-2: #323540;
      --ink: #f3f5f8;
      --ink-2: #a3a8b4;
      --ink-3: #6c7280;
      --line: rgba(255, 255, 255, 0.09);
      --line-strong: rgba(255, 255, 255, 0.14);
      --accent: ${accentDark};
      --accent-ink: ${accentInkDark};
      --accent-tint: ${accentTintDark};
      --accent-on: #ffffff;
      --green: #4ade80;
      --green-tint: rgba(74, 222, 128, 0.14);
      --red: #f87171;
      --red-tint: rgba(248, 113, 113, 0.14);
      --shadow-btn: 0 0 0 1px rgba(255, 255, 255, 0.1), 0 1px 2px rgba(0, 0, 0, 0.32);
      --shadow-card: 0 0 0 1px rgba(255, 255, 255, 0.09), 0 1px 2px rgba(0, 0, 0, 0.24), 0 3px 10px rgba(0, 0, 0, 0.2);
      --shadow-raised: 0 0 0 1px rgba(255, 255, 255, 0.12), 0 8px 24px rgba(0, 0, 0, 0.34);
      color-scheme: dark;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html {
      scroll-behavior: smooth;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }
    body {
      background: var(--page);
      color: var(--ink);
      font-family: var(--font-sans);
      font-size: 14px;
      line-height: 1.55;
      letter-spacing: -0.01em;
      transition: background 180ms var(--ease), color 180ms var(--ease);
    }
    h1, h2, h3, h4 {
      font-family: var(--font-display);
      letter-spacing: -0.025em;
      text-wrap: balance;
    }
    p { text-wrap: pretty; }
    .num, .mono { font-variant-numeric: tabular-nums; }
    .mono { font-family: var(--font-mono); }
    button, input, select { font: inherit; color: inherit; }
    button:not(:disabled) { cursor: pointer; }
    :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
    .container { max-width: 1020px; margin: 0 auto; padding: 0 22px; }
    .topbar {
      position: sticky; top: 0; z-index: 30;
      height: 56px;
      background: color-mix(in srgb, var(--page) 86%, transparent);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--line);
    }
    .topbar-inner {
      height: 100%;
      display: flex; align-items: center; justify-content: space-between; gap: 16px;
    }
    .brand {
      display: inline-flex; align-items: center; gap: 8px;
      font-family: var(--font-display); font-weight: 600; font-size: 14.5px;
      letter-spacing: -0.02em; color: var(--ink); text-decoration: none; white-space: nowrap;
    }
    .brand-mark {
      width: 24px; height: 24px; border-radius: 7px;
      background: var(--surface); box-shadow: var(--shadow-btn);
      display: inline-flex; align-items: center; justify-content: center; color: var(--accent); flex-shrink: 0;
    }
    .nav-center {
      display: flex; align-items: center; gap: 4px;
    }
    .nav-link {
      padding: 6px 10px; border-radius: 7px; font-size: 13.5px; font-weight: 500;
      color: var(--ink-2); text-decoration: none; white-space: nowrap;
      transition: background 140ms var(--ease), color 140ms var(--ease);
    }
    .nav-link:hover { background: var(--hover); color: var(--ink); }
    @media (max-width: 640px) { .nav-center { display: none; } }
    .nav-actions { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
    .icon-btn {
      width: 32px; height: 32px; border-radius: 8px; border: none;
      background: var(--surface); box-shadow: var(--shadow-btn); color: var(--ink-2);
      display: inline-flex; align-items: center; justify-content: center;
      transition: background 140ms var(--ease), color 140ms var(--ease), transform 100ms var(--ease);
    }
    .icon-btn:hover { background: var(--hover); color: var(--ink); }
    .icon-btn:active { transform: scale(0.96); }
    .btn {
      display: inline-flex; align-items: center; justify-content: center; gap: 7px;
      height: 36px; padding: 0 15px; border-radius: 8px; border: none;
      font-size: 13.5px; font-weight: 500; text-decoration: none; white-space: nowrap;
      transition: background 140ms var(--ease), opacity 140ms var(--ease), transform 100ms var(--ease);
    }
    .btn:active { transform: scale(0.985); }
    .btn-sm { height: 32px; padding: 0 12px; font-size: 13px; }
    .btn-primary { background: var(--accent); color: var(--accent-on); }
    .btn-primary:hover { filter: brightness(1.06); }
    .btn-secondary { background: var(--surface); color: var(--ink); box-shadow: var(--shadow-btn); }
    .btn-secondary:hover { background: var(--hover); }
    .surface-card {
      background: var(--surface); border-radius: 12px; box-shadow: var(--shadow-card);
    }
    .seg-tabs {
      display: inline-flex; align-items: center; gap: 2px;
      background: var(--inset); padding: 3px; border-radius: 9px;
      border: 1px solid var(--line);
    }
    .seg-btn {
      border: none; background: transparent; color: var(--ink-2);
      padding: 5px 11px; border-radius: 6px; font-size: 12.5px; font-weight: 500;
      white-space: nowrap; transition: background 140ms var(--ease), color 140ms var(--ease), box-shadow 140ms var(--ease);
    }
    .seg-btn:hover { color: var(--ink); }
    .seg-btn.active { background: var(--surface); color: var(--ink); box-shadow: var(--shadow-btn); }
    .field-input {
      width: 100%; height: 38px; padding: 0 12px; border-radius: 8px;
      background: var(--inset); border: 1px solid var(--line-strong); color: var(--ink);
      font-size: 13.5px; outline: none;
      transition: border-color 140ms var(--ease), background 140ms var(--ease), box-shadow 140ms var(--ease);
    }
    .field-input::placeholder { color: var(--ink-3); }
    .field-input:focus {
      background: var(--surface); border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-tint);
    }
    footer.site-footer {
      border-top: 1px solid var(--line);
      padding: 20px 0; margin-top: 40px;
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px;
      font-size: 12.5px; color: var(--ink-3);
    }
    * { scrollbar-width: thin; scrollbar-color: rgba(140, 148, 165, 0.35) transparent; }
    *::-webkit-scrollbar { width: 6px; height: 6px; }
    *::-webkit-scrollbar-thumb { background: rgba(140, 148, 165, 0.35); border-radius: 999px; }`;
}

const SUN_MOON_ICON_SVG = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.93 4.93l1.41 1.41m11.32 11.32l1.41 1.41M2 12h2m16 0h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg>`;

function buildNexoraThemeScript(defaultDark: boolean): string {
  return `<script>
    (function () {
      const root = document.documentElement;
      const btn = document.getElementById("themeBtn");
      ${defaultDark ? 'root.setAttribute("data-theme", "dark");' : ""}
      if (!btn) return;
      const sync = () => {
        const isDark = root.getAttribute("data-theme") === "dark";
        btn.setAttribute("aria-pressed", String(isDark));
        btn.title = isDark ? "Switch to light surface" : "Switch to dark surface";
      };
      sync();
      btn.addEventListener("click", () => {
        if (root.getAttribute("data-theme") === "dark") root.removeAttribute("data-theme");
        else root.setAttribute("data-theme", "dark");
        sync();
      });
    })();
  </script>`;
}

function resolveRequestedPalette(
  lower: string,
  fallback: "cobalt" | "ember" | "espresso" = "cobalt"
): "cobalt" | "ember" | "espresso" | "emerald" | "violet" {
  if (/\b(coffee|espresso|warm\s+palette|mocha|caramel|brown)\b/i.test(lower)) {
    return "espresso";
  }
  if (/\b(emerald|mint|forest|green\s+palette)\b/i.test(lower)) {
    return "emerald";
  }
  if (/\b(violet|purple|lavender|plum)\b/i.test(lower)) {
    return "violet";
  }
  if (/\b(ember|orange|flame|coral)\b/i.test(lower)) {
    return "ember";
  }
  if (/\b(blue|cobalt)\b/i.test(lower)) {
    return "cobalt";
  }
  return fallback;
}

const PRICING_ADDON_HTML = `    <section id="pricing" style="margin:28px 0 8px;padding-top:22px;border-top:1px solid var(--line);">
      <div style="margin-bottom:14px;">
        <h2 style="font-size:19px;font-weight:600;margin-bottom:4px;">Transparent Plans &amp; Bundles</h2>
        <p style="font-size:13.5px;color:var(--ink-2);">Clear, predictable options with zero hidden fees.</p>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;">
        <article class="surface-card" style="padding:18px;display:flex;flex-direction:column;justify-content:space-between;gap:12px;">
          <div>
            <div style="font-size:12px;color:var(--ink-3);font-weight:500;">Starter</div>
            <div class="mono" style="font-size:24px;font-weight:600;margin:4px 0;">$14<span style="font-size:12.5px;color:var(--ink-2);font-weight:400;"> / order</span></div>
            <p style="font-size:13px;color:var(--ink-2);">Core essentials, express pickup access, and standard customization.</p>
          </div>
          <a class="btn btn-secondary btn-sm" href="#reserve">Choose Starter</a>
        </article>
        <article class="surface-card" style="padding:18px;display:flex;flex-direction:column;justify-content:space-between;gap:12px;box-shadow:0 0 0 1px color-mix(in srgb, var(--accent) 45%, transparent), var(--shadow-card);">
          <div>
            <div style="font-size:12px;color:var(--accent-ink);font-weight:600;">Most Popular</div>
            <div class="mono" style="font-size:24px;font-weight:600;margin:4px 0;">$29<span style="font-size:12.5px;color:var(--ink-2);font-weight:400;"> / bundle</span></div>
            <p style="font-size:13px;color:var(--ink-2);">Full signature selection, priority prep queue, and bonus sides.</p>
          </div>
          <a class="btn btn-primary btn-sm" href="#reserve">Select Popular</a>
        </article>
        <article class="surface-card" style="padding:18px;display:flex;flex-direction:column;justify-content:space-between;gap:12px;">
          <div>
            <div style="font-size:12px;color:var(--ink-3);font-weight:500;">Group / Event</div>
            <div class="mono" style="font-size:24px;font-weight:600;margin:4px 0;">$59<span style="font-size:12.5px;color:var(--ink-2);font-weight:400;"> / pack</span></div>
            <p style="font-size:13px;color:var(--ink-2);">Multi-person platters, scheduled table holding, and dedicated support.</p>
          </div>
          <a class="btn btn-secondary btn-sm" href="#reserve">Book Group Pack</a>
        </article>
      </div>
    </section>`;

function buildModernHtmlPage(prompt: string): {
  title: string;
  html: string;
  highlights: string[];
  summary: string;
} {
  const normalizedPrompt = normalizeTypoTerms(prompt);
  const lower = normalizedPrompt.toLowerCase();
  const darkMatches = Array.from(lower.matchAll(/\b(dark(?:\s+mode)?|light(?:\s+mode)?|obsidian|midnight|daylight)\b/gi));
  const lastThemeKeyword = darkMatches.length > 0 ? darkMatches[darkMatches.length - 1][1].toLowerCase() : "";
  const preferDark = lastThemeKeyword
    ? /\b(dark|obsidian|midnight)\b/.test(lastThemeKeyword)
    : false;
  const htmlThemeAttr = preferDark ? ' data-theme="dark"' : "";
  const wantsPricingSection = /\b(pricing|plans|tiers)\b/i.test(lower);

  const isPortfolio = /\b(portfolio|resume|cv|personal\s+site|bio)\b/i.test(lower);
  const isLogin = /\b(login|sign\s*in|auth|register|signup)\b/i.test(lower);
  const isDashboard = /\b(dashboard|analytics|admin|metrics|kpi)\b/i.test(lower);
  const isRestaurant =
    /\b(chicken|restaurant|food|kitchen|grill|roast|wings|burger|pizza|cafe|coffee|bakery|bistro|diner|bbq|taco|sushi|noodle|steak|menu|dining)\b/i.test(
      lower
    );
  const isPricing = !isRestaurant && !isDashboard && /\b(pricing|plans|subscription|saas)\b/i.test(lower);
  const isCalculator = /\b(calculator|calc)\b/i.test(lower);
  const isTodo = /\b(todo|to-do|task\s*list|tasks\s*app|checklist)\b/i.test(lower);

  if (isCalculator) {
    return {
      title: "Interactive Calculator",
      summary:
        "Built a sleek **Calculator** modeled on Nexora's layered hairline surfaces, monospace tabular readout, keyboard controls, and zero-`eval()` expression parser.",
      highlights: [
        "**Sleek Surface & Token Architecture:** Uses cool near-white / matte obsidian surfaces, `1px` hairline borders, `Plus Jakarta Sans` header, and `JetBrains Mono` `tabular-nums` display.",
        "**Safe Expression Engine & Keyboard Input:** Full mouse + keyboard (`0-9`, `+ - * / %`, `Enter`, `Backspace`, `Escape`) without `eval()`.",
        "**Theme Switcher:** Instant light and obsidian dark toggle built directly into the card header.",
      ],
      html: `<!DOCTYPE html>
<html lang="en"${htmlThemeAttr}>
<head>
${buildNexoraHeadAssets("Calculator", "cobalt")}
    body { min-height: 100vh; display: grid; place-items: center; padding: 20px; }
    .calc { width: 100%; max-width: 328px; padding: 16px; border-radius: 14px; }
    .calc-head {
      display: flex; align-items: center; justify-content: space-between;
      padding-bottom: 10px; border-bottom: 1px solid var(--line);
    }
    .calc-title { font-family: var(--font-display); font-size: 13px; font-weight: 600; color: var(--ink-2); }
    .screen {
      margin: 12px 0 14px; padding: 14px 12px;
      background: var(--inset); border: 1px solid var(--line); border-radius: 10px;
      text-align: right;
    }
    .prev { min-height: 18px; font-family: var(--font-mono); font-size: 12px; color: var(--ink-3); font-variant-numeric: tabular-nums; }
    .curr { font-family: var(--font-mono); font-size: 30px; font-weight: 500; color: var(--ink); font-variant-numeric: tabular-nums; word-break: break-all; margin-top: 2px; line-height: 1.15; }
    .keys { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
    .key {
      height: 46px; border: none; border-radius: 9px;
      background: var(--surface); box-shadow: var(--shadow-btn); color: var(--ink);
      font-family: var(--font-mono); font-size: 15px; font-weight: 500;
      transition: background 120ms var(--ease), transform 90ms var(--ease);
    }
    .key:hover { background: var(--hover); }
    .key:active { transform: scale(0.96); }
    .key.fn { background: var(--inset); color: var(--ink-2); font-family: var(--font-sans); font-size: 13px; font-weight: 600; }
    .key.op { background: var(--accent-tint); color: var(--accent-ink); box-shadow: 0 0 0 1px color-mix(in srgb, var(--accent) 22%, transparent); }
    .key.eq { background: var(--accent); color: var(--accent-on); box-shadow: none; }
    .key.eq:hover { filter: brightness(1.06); }
    .key.span2 { grid-column: span 2; }
  </style>
</head>
<body>
  <main class="surface-card calc" aria-label="Calculator">
    <div class="calc-head">
      <span class="calc-title">Calculator</span>
      <button type="button" class="icon-btn" id="themeBtn" aria-label="Toggle theme">${SUN_MOON_ICON_SVG}</button>
    </div>
    <div class="screen">
      <div class="prev" id="prev"></div>
      <div class="curr" id="curr">0</div>
    </div>
    <div class="keys" id="keys">
      <button data-act="clear" class="key fn">AC</button>
      <button data-act="del" class="key fn">DEL</button>
      <button data-op="%" class="key op">%</button>
      <button data-op="/" class="key op">&divide;</button>
      <button data-num="7" class="key">7</button><button data-num="8" class="key">8</button><button data-num="9" class="key">9</button>
      <button data-op="*" class="key op">&times;</button>
      <button data-num="4" class="key">4</button><button data-num="5" class="key">5</button><button data-num="6" class="key">6</button>
      <button data-op="-" class="key op">&minus;</button>
      <button data-num="1" class="key">1</button><button data-num="2" class="key">2</button><button data-num="3" class="key">3</button>
      <button data-op="+" class="key op">+</button>
      <button data-num="0" class="key span2">0</button>
      <button data-num="." class="key">.</button>
      <button data-act="eq" class="key eq">=</button>
    </div>
  </main>
  ${buildNexoraThemeScript(preferDark)}
  <script>
    let curr = "0", prev = "", op = null, resetNext = false;
    const currEl = document.getElementById("curr"), prevEl = document.getElementById("prev");
    const render = () => { currEl.textContent = curr; prevEl.textContent = op ? prev + " " + op : ""; };
    const compute = () => {
      const a = parseFloat(prev), b = parseFloat(curr);
      if (isNaN(a) || isNaN(b)) return curr;
      if (op === "/" && b === 0) return "Error";
      const res = op === "+" ? a + b : op === "-" ? a - b : op === "*" ? a * b : op === "/" ? a / b : op === "%" ? a % b : b;
      return String(Number(res.toPrecision(12)));
    };
    const pushNum = (n) => {
      if (curr === "Error" || resetNext) { curr = n === "." ? "0." : n; resetNext = false; }
      else if (n === "." && curr.includes(".")) return;
      else curr = curr === "0" && n !== "." ? n : curr + n;
      render();
    };
    const pushOp = (nextOp) => {
      if (curr === "Error") return;
      if (op && !resetNext) { curr = compute(); }
      prev = curr; op = nextOp; resetNext = true; render();
    };
    document.getElementById("keys").addEventListener("click", (e) => {
      const btn = e.target.closest("button"); if (!btn) return;
      if (btn.dataset.num) pushNum(btn.dataset.num);
      else if (btn.dataset.op) pushOp(btn.dataset.op);
      else if (btn.dataset.act === "clear") { curr = "0"; prev = ""; op = null; resetNext = false; render(); }
      else if (btn.dataset.act === "del") { curr = curr.length > 1 ? curr.slice(0, -1) : "0"; render(); }
      else if (btn.dataset.act === "eq" && op) { curr = compute(); prev = ""; op = null; resetNext = true; render(); }
    });
    window.addEventListener("keydown", (e) => {
      if (/^[0-9.]$/.test(e.key)) pushNum(e.key);
      else if (["+", "-", "*", "/", "%"].includes(e.key)) pushOp(e.key);
      else if (e.key === "Enter" || e.key === "=") { if (op) { curr = compute(); prev = ""; op = null; resetNext = true; render(); } }
      else if (e.key === "Backspace") { curr = curr.length > 1 ? curr.slice(0, -1) : "0"; render(); }
      else if (e.key === "Escape") { curr = "0"; prev = ""; op = null; resetNext = false; render(); }
    });
  </script>
</body>
</html>`,
    };
  }

  if (isTodo) {
    return {
      title: "Interactive Tasks Workspace",
      summary:
        "Built a sleek **Tasks Workspace** in clean Nexora surfaces featuring segmented `All / Active / Done` filters, live remaining counters, and sandbox-safe persistence.",
      highlights: [
        "**Segmented Filter Track:** Signature inset segmented switcher (`All / Active / Done`) with crisp active elevation.",
        "**Recessed Composer & Hairline Rows:** Accessible focus rings, custom toggle states, and `tabular-nums` item counts.",
        "**Sandbox-Safe Persistence:** Safely persists via `localStorage` with an automatic in-memory fallback inside restricted preview frames.",
      ],
      html: `<!DOCTYPE html>
<html lang="en"${htmlThemeAttr}>
<head>
${buildNexoraHeadAssets("Tasks", "cobalt")}
    body { min-height: 100vh; display: flex; justify-content: center; padding: 36px 16px; }
    .app { width: 100%; max-width: 500px; padding: 20px; border-radius: 14px; height: fit-content; }
    .head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 16px; }
    .head h1 { font-size: 18px; font-weight: 600; }
    form { display: flex; gap: 8px; margin-bottom: 14px; }
    .toolbar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 12px; }
    .count { font-size: 12.5px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
    ul { list-style: none; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; background: var(--surface); }
    li {
      display: flex; align-items: center; justify-content: space-between; gap: 10px;
      padding: 10px 12px; font-size: 13.5px; border-top: 1px solid var(--line);
      transition: background 120ms var(--ease);
    }
    li:first-child { border-top: none; }
    li:hover { background: var(--hover); }
    li label { display: flex; align-items: center; gap: 10px; flex: 1; cursor: pointer; min-width: 0; }
    li input[type="checkbox"] { width: 15px; height: 15px; accent-color: var(--accent); cursor: pointer; flex-shrink: 0; }
    li span { overflow: hidden; text-overflow: ellipsis; }
    li.done span { text-decoration: line-through; color: var(--ink-3); }
    .del {
      width: 24px; height: 24px; border: none; border-radius: 6px;
      background: transparent; color: var(--ink-3); display: inline-flex; align-items: center; justify-content: center;
      transition: background 120ms var(--ease), color 120ms var(--ease);
    }
    .del:hover { background: var(--hover-2); color: var(--ink); }
    .empty { padding: 22px 14px; text-align: center; font-size: 13px; color: var(--ink-3); border: 1px dashed var(--line-strong); border-radius: 10px; }
    .foot { margin-top: 12px; display: flex; justify-content: flex-end; }
    .clear-btn {
      border: none; background: transparent; color: var(--ink-2); font-size: 12.5px; font-weight: 500;
      padding: 4px 8px; border-radius: 6px; transition: background 120ms var(--ease), color 120ms var(--ease);
    }
    .clear-btn:hover { background: var(--hover); color: var(--ink); }
  </style>
</head>
<body>
  <main class="surface-card app">
    <div class="head">
      <h1>Tasks</h1>
      <button type="button" class="icon-btn" id="themeBtn" aria-label="Toggle theme">${SUN_MOON_ICON_SVG}</button>
    </div>
    <form id="form">
      <input id="input" class="field-input" type="text" placeholder="Add a new task..." required />
      <button class="btn btn-primary" type="submit">Add task</button>
    </form>
    <div class="toolbar">
      <div class="seg-tabs" role="tablist" aria-label="Task filter">
        <button type="button" class="seg-btn active" data-filter="all">All</button>
        <button type="button" class="seg-btn" data-filter="active">Active</button>
        <button type="button" class="seg-btn" data-filter="done">Done</button>
      </div>
      <span class="count" id="count">0 items left</span>
    </div>
    <ul id="list"></ul>
    <div class="empty" id="empty" hidden>No tasks in this view yet. Add one above to get started.</div>
    <div class="foot">
      <button id="clearDone" type="button" class="clear-btn">Clear completed</button>
    </div>
  </main>
  ${buildNexoraThemeScript(preferDark)}
  <script>
    const SEED = [
      { id: 1, text: "Review responsive hairline layout", done: false },
      { id: 2, text: "Test light and obsidian dark surface tokens", done: true }
    ];
    function loadItems() {
      try {
        const raw = window.localStorage.getItem("nexora_demo_todos");
        return raw ? JSON.parse(raw) : SEED;
      } catch (_) {
        return SEED;
      }
    }
    function saveItems(data) {
      try { window.localStorage.setItem("nexora_demo_todos", JSON.stringify(data)); } catch (_) {}
    }
    let items = loadItems();
    let filter = "all";
    const render = () => {
      const list = document.getElementById("list");
      const empty = document.getElementById("empty");
      const visible = items.filter((t) => filter === "all" ? true : filter === "done" ? t.done : !t.done);
      list.hidden = visible.length === 0;
      empty.hidden = visible.length > 0;
      list.innerHTML = visible.map((t) =>
        '<li class="' + (t.done ? "done" : "") + '">' +
          '<label><input type="checkbox" data-id="' + t.id + '"' + (t.done ? " checked" : "") + ' /><span>' + t.text.replace(/</g, "&lt;") + '</span></label>' +
          '<button type="button" class="del" aria-label="Remove task" data-del="' + t.id + '"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg></button>' +
        '</li>'
      ).join("");
      const left = items.filter((t) => !t.done).length;
      document.getElementById("count").textContent = left + " item" + (left === 1 ? "" : "s") + " remaining";
    };
    document.getElementById("form").addEventListener("submit", (e) => {
      e.preventDefault();
      const inp = document.getElementById("input");
      if (!inp.value.trim()) return;
      items.unshift({ id: Date.now(), text: inp.value.trim(), done: false });
      inp.value = ""; saveItems(items); render();
    });
    document.getElementById("list").addEventListener("click", (e) => {
      const cb = e.target.closest("input[type=checkbox]");
      if (cb) { items = items.map((t) => t.id == cb.dataset.id ? { ...t, done: cb.checked } : t); saveItems(items); render(); }
      const del = e.target.closest("[data-del]");
      if (del) { items = items.filter((t) => t.id != del.dataset.del); saveItems(items); render(); }
    });
    document.querySelectorAll(".seg-btn").forEach((b) => b.addEventListener("click", () => {
      document.querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
      b.classList.add("active"); filter = b.dataset.filter; render();
    }));
    document.getElementById("clearDone").addEventListener("click", () => { items = items.filter((t) => !t.done); saveItems(items); render(); });
    render();
  </script>
</body>
</html>`,
    };
  }

  if (isLogin) {
    const brand = extractTopicTitle(prompt, "Nexora Workspace");
    const safeBrand = escapeHtmlText(brand);
    return {
      title: `${brand} — Sign In`,
      summary: `Built a clean, accessible sign-in experience for **${brand}** styled with Nexora's layered card elevation, recessed inputs, password reveal control, and light/obsidian surfaces.`,
      highlights: [
        "**Sleek Auth Surface:** Centered `14px` card with hairline border, geometric brand mark, and subtle recessed form fields.",
        "**Interactive Controls:** Working show/hide password button, email + password validation, and instant confirmation feedback.",
        "**Light & Obsidian Themes:** Toggle seamlessly between crisp near-white and deep matte obsidian.",
      ],
      html: `<!DOCTYPE html>
<html lang="en"${htmlThemeAttr}>
<head>
${buildNexoraHeadAssets(`${brand} — Sign In`, "cobalt")}
    body { min-height: 100vh; display: grid; place-items: center; padding: 20px; }
    .auth-card { width: 100%; max-width: 396px; padding: 24px; border-radius: 14px; box-shadow: var(--shadow-raised); }
    .auth-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
    h1 { font-size: 20px; font-weight: 600; margin-bottom: 4px; }
    .sub { color: var(--ink-2); font-size: 13.5px; margin-bottom: 20px; }
    .field { margin-bottom: 14px; }
    label { display: block; font-size: 12.5px; font-weight: 500; color: var(--ink-2); margin-bottom: 6px; }
    .input-wrap { position: relative; }
    .pw-btn {
      position: absolute; right: 8px; top: 50%; transform: translateY(-50%);
      border: none; background: transparent; color: var(--ink-2);
      font-size: 12px; font-weight: 500; padding: 4px 6px; border-radius: 6px;
    }
    .pw-btn:hover { background: var(--hover); color: var(--ink); }
    .submit-btn { width: 100%; height: 40px; margin-top: 6px; }
    .status {
      margin-top: 14px; padding: 9px 12px; border-radius: 8px;
      font-size: 12.5px; font-weight: 500; text-align: center;
      display: none;
    }
    .status.ok { display: block; background: var(--green-tint); color: var(--green); }
    .status.err { display: block; background: var(--red-tint); color: var(--red); }
  </style>
</head>
<body>
  <main class="surface-card auth-card">
    <div class="auth-top">
      <a class="brand" href="#">
        <span class="brand-mark">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 19V5l14 14V5"/></svg>
        </span>
        <span>${safeBrand}</span>
      </a>
      <button type="button" class="icon-btn" id="themeBtn" aria-label="Toggle theme">${SUN_MOON_ICON_SVG}</button>
    </div>
    <h1>Sign in to workspace</h1>
    <p class="sub">Enter your credentials to continue.</p>
    <form id="loginForm" novalidate>
      <div class="field">
        <label for="email">Work email</label>
        <input id="email" class="field-input" type="email" autocomplete="email" placeholder="you@company.com" required />
      </div>
      <div class="field">
        <label for="password">Password</label>
        <div class="input-wrap">
          <input id="password" class="field-input" style="padding-right:56px;" type="password" autocomplete="current-password" placeholder="At least 8 characters" required />
          <button type="button" class="pw-btn" id="togglePw">Show</button>
        </div>
      </div>
      <button class="btn btn-primary submit-btn" type="submit">Continue</button>
      <div class="status" id="status" role="status"></div>
    </form>
  </main>
  ${buildNexoraThemeScript(preferDark)}
  <script>
    const pw = document.getElementById("password");
    document.getElementById("togglePw").addEventListener("click", (e) => {
      const show = pw.type === "password";
      pw.type = show ? "text" : "password";
      e.target.textContent = show ? "Hide" : "Show";
    });
    document.getElementById("loginForm").addEventListener("submit", (e) => {
      e.preventDefault();
      const email = document.getElementById("email").value.trim();
      const status = document.getElementById("status");
      if (!email.includes("@") || !email.includes(".") || pw.value.length < 8) {
        status.className = "status err";
        status.textContent = "Please enter a valid email and an 8+ character password.";
        return;
      }
      status.className = "status ok";
      status.textContent = "Authenticated cleanly as " + email;
    });
  </script>
</body>
</html>`,
    };
  }

  if (isDashboard) {
    const title = extractTopicTitle(prompt, "Operations Overview");
    const safeTitle = escapeHtmlText(title);
    return {
      title,
      summary: `Built a sleek **${title}** dashboard styled with Nexora's 3-zone top bar, segmented timeframe selector, ` + "`tabular-nums`" + ` KPI grid, interactive SVG trend chart, and searchable queue.`,
      highlights: [
        "**Nexora Workspace Chrome:** Single-row `56px` sticky header with clean breadcrumb navigation and segmented `7D / 30D / 90D` selector.",
        "**Single-Elevation Metric Grid & SVG Chart:** Tabular figures paired with an adaptive inline trend curve that updates per timeframe.",
        "**Filterable Operations Queue:** Searchable table rows (`38px` compact density) with interactive task completion checkboxes.",
      ],
      html: `<!DOCTYPE html>
<html lang="en"${htmlThemeAttr}>
<head>
${buildNexoraHeadAssets(title, "cobalt")}
    .crumb { display: flex; align-items: center; gap: 8px; font-size: 13.5px; color: var(--ink-2); white-space: nowrap; }
    .crumb strong { font-family: var(--font-display); font-weight: 600; color: var(--ink); }
    .dash-main { padding: 22px 0 36px; display: flex; flex-direction: column; gap: 16px; }
    .kpi-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    @media (max-width: 700px) { .kpi-grid { grid-template-columns: 1fr; } }
    .kpi { padding: 16px 18px; }
    .kpi-label { font-size: 12.5px; color: var(--ink-2); font-weight: 500; }
    .kpi-row { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; margin-top: 8px; }
    .kpi-val { font-family: var(--font-display); font-size: 24px; font-weight: 600; font-variant-numeric: tabular-nums; letter-spacing: -0.025em; }
    .kpi-delta { font-family: var(--font-mono); font-size: 12px; font-weight: 500; color: var(--green); font-variant-numeric: tabular-nums; }
    .panel { padding: 16px 18px; }
    .panel-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 12px; }
    .panel-head h2 { font-size: 14.5px; font-weight: 600; }
    .search-box { max-width: 220px; height: 32px; font-size: 12.5px; }
    .queue-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .queue-table th {
      text-align: left; font-size: 11.5px; font-weight: 500; color: var(--ink-3);
      padding: 6px 10px 8px; border-bottom: 1px solid var(--line);
    }
    .queue-table td { padding: 9px 10px; border-bottom: 1px solid var(--line); vertical-align: middle; }
    .queue-table tr:last-child td { border-bottom: none; }
    .queue-table tr:hover td { background: var(--hover); }
    .queue-table tr.done td.title-cell { text-decoration: line-through; color: var(--ink-3); }
    .state-txt { font-size: 12px; font-weight: 500; color: var(--green); }
    .state-txt.pending { color: var(--ink-2); }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="container topbar-inner">
      <div class="crumb">
        <span class="brand-mark">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 19V5l14 14V5"/></svg>
        </span>
        <span>Workspace</span>
        <span style="color:var(--ink-3)">/</span>
        <strong>${safeTitle}</strong>
      </div>
      <div class="nav-actions">
        <div class="seg-tabs" role="tablist" aria-label="Timeframe">
          <button type="button" class="seg-btn active" data-range="7d">7D</button>
          <button type="button" class="seg-btn" data-range="30d">30D</button>
          <button type="button" class="seg-btn" data-range="90d">90D</button>
        </div>
        <button type="button" class="icon-btn" id="themeBtn" aria-label="Toggle theme">${SUN_MOON_ICON_SVG}</button>
      </div>
    </div>
  </header>

  <main class="container dash-main">
    <section class="kpi-grid" aria-label="Key metrics">
      <article class="surface-card kpi">
        <div class="kpi-label">Active Sessions</div>
        <div class="kpi-row"><span class="kpi-val" id="mUsers">12,480</span><span class="kpi-delta" id="dUsers">+14.2%</span></div>
      </article>
      <article class="surface-card kpi">
        <div class="kpi-label">Conversion Rate</div>
        <div class="kpi-row"><span class="kpi-val" id="mConv">4.82%</span><span class="kpi-delta" id="dConv">+0.64%</span></div>
      </article>
      <article class="surface-card kpi">
        <div class="kpi-label">Median Latency</div>
        <div class="kpi-row"><span class="kpi-val" id="mLat">118 ms</span><span class="kpi-delta" id="dLat">-19 ms</span></div>
      </article>
    </section>

    <section class="surface-card panel">
      <div class="panel-head">
        <h2>Throughput Trend</h2>
        <span class="mono" style="font-size:12px;color:var(--ink-3);" id="chartCaption">7-day request index</span>
      </div>
      <svg id="trendSvg" viewBox="0 0 640 120" width="100%" height="120" style="display:block;overflow:visible;" aria-label="Throughput trend chart"></svg>
    </section>

    <section class="surface-card panel">
      <div class="panel-head">
        <h2>Action Queue</h2>
        <input id="queueFilter" class="field-input search-box" type="search" placeholder="Filter tasks..." />
      </div>
      <table class="queue-table">
        <thead>
          <tr><th style="width:32px;"></th><th>Task</th><th>ID</th><th style="text-align:right;">Status</th></tr>
        </thead>
        <tbody id="queueBody">
          <tr><td><input type="checkbox" /></td><td class="title-cell">Review deployment checklist before release</td><td class="mono" style="color:var(--ink-3);font-size:12px;">OPS-104</td><td style="text-align:right;"><span class="state-txt pending">Queued</span></td></tr>
          <tr class="done"><td><input type="checkbox" checked /></td><td class="title-cell">Verify responsive layout tokens across breakpoints</td><td class="mono" style="color:var(--ink-3);font-size:12px;">OPS-103</td><td style="text-align:right;"><span class="state-txt">Completed</span></td></tr>
          <tr><td><input type="checkbox" /></td><td class="title-cell">Export weekly summary telemetry CSV</td><td class="mono" style="color:var(--ink-3);font-size:12px;">OPS-102</td><td style="text-align:right;"><span class="state-txt pending">Queued</span></td></tr>
        </tbody>
      </table>
    </section>
  </main>
  ${buildNexoraThemeScript(preferDark)}
  <script>
    const STATS = {
      "7d":  { users: "12,480", dU: "+14.2%", conv: "4.82%", dC: "+0.64%", lat: "118 ms", dL: "-19 ms", cap: "7-day request index",  pts: [42, 48, 45, 58, 63, 74, 86] },
      "30d": { users: "48,910", dU: "+21.8%", conv: "5.14%", dC: "+0.91%", lat: "112 ms", dL: "-24 ms", cap: "30-day request index", pts: [30, 36, 44, 49, 61, 70, 91] },
      "90d": { users: "142,300", dU: "+38.4%", conv: "5.40%", dC: "+1.18%", lat: "105 ms", dL: "-31 ms", cap: "90-day request index", pts: [22, 34, 41, 55, 68, 80, 96] }
    };
    function drawChart(pts) {
      const svg = document.getElementById("trendSvg");
      const W = 640, H = 120, px = 12, py = 12;
      const max = Math.max(...pts, 100);
      const step = (W - px * 2) / (pts.length - 1);
      const coords = pts.map((v, i) => [px + i * step, H - py - (v / max) * (H - py * 2)]);
      const linePath = coords.map((p, i) => (i === 0 ? "M" : "L") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ");
      const areaPath = linePath + " L" + coords[coords.length - 1][0].toFixed(1) + "," + (H - py) + " L" + coords[0][0].toFixed(1) + "," + (H - py) + " Z";
      svg.innerHTML =
        '<line x1="0" y1="' + (H - py) + '" x2="' + W + '" y2="' + (H - py) + '" stroke="var(--line)" stroke-width="1"/>' +
        '<path d="' + areaPath + '" fill="var(--accent-tint)"/>' +
        '<path d="' + linePath + '" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>' +
        coords.map((p) => '<circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="3.5" fill="var(--surface)" stroke="var(--accent)" stroke-width="2"/>').join("");
    }
    document.querySelectorAll(".seg-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const s = STATS[btn.dataset.range];
        document.getElementById("mUsers").textContent = s.users;
        document.getElementById("dUsers").textContent = s.dU;
        document.getElementById("mConv").textContent = s.conv;
        document.getElementById("dConv").textContent = s.dC;
        document.getElementById("mLat").textContent = s.lat;
        document.getElementById("dLat").textContent = s.dL;
        document.getElementById("chartCaption").textContent = s.cap;
        drawChart(s.pts);
      });
    });
    document.querySelectorAll("#queueBody tr").forEach((tr) => {
      const cb = tr.querySelector("input");
      const badge = tr.querySelector(".state-txt");
      cb.addEventListener("change", () => {
        tr.classList.toggle("done", cb.checked);
        badge.textContent = cb.checked ? "Completed" : "Queued";
        badge.classList.toggle("pending", !cb.checked);
      });
    });
    document.getElementById("queueFilter").addEventListener("input", (e) => {
      const q = e.target.value.trim().toLowerCase();
      document.querySelectorAll("#queueBody tr").forEach((tr) => {
        tr.style.display = tr.textContent.toLowerCase().includes(q) ? "" : "none";
      });
    });
    drawChart(STATS["7d"].pts);
  </script>
</body>
</html>`,
    };
  }

  if (isRestaurant) {
    const isChicken = /\b(chicken|wings|roast|fried|tenders|coop|rooster)\b/i.test(lower);
    const isCoffeeOrBakery = /\b(coffee|cafe|bakery|espresso|pastry)\b/i.test(lower);
    const brandName = extractTopicTitle(
      normalizedPrompt,
      isChicken ? "Golden Coop Chicken House" : isCoffeeOrBakery ? "Roast & Crumb Cafe" : "Hearth & Table Kitchen"
    );
    const safeBrand = escapeHtmlText(brandName);
    const scheme = resolveRequestedPalette(lower, isCoffeeOrBakery ? "espresso" : "ember");
    const tagline = isChicken
      ? "Crispy buttermilk tenders, oak-fired half chicken, and scratch sauces prepared fresh to order."
      : isCoffeeOrBakery
        ? "Small-batch espresso, warm laminated pastries, and seasonal breakfast plates served daily."
        : "Wood-fired signatures, crisp seasonal sides, and attentive hospitality on every plate.";

    const menuItems = isChicken
      ? [
          { id: 1, cat: "Crispy", name: "Buttermilk Crispy Tenders", price: 13.5, desc: "24-hour brined tenders, golden crunch, house comeback glaze." },
          { id: 2, cat: "Crispy", name: "Nashville Hot Brioche", price: 14.5, desc: "Spiced crispy thigh, dill pickles, herb slaw, toasted bun." },
          { id: 3, cat: "Grilled", name: "Oak-Fired Half Chicken", price: 17.0, desc: "Citrus-garlic marinade, blistered skin, lemon herb jus." },
          { id: 4, cat: "Grilled", name: "Charcoal Wings (8pc)", price: 12.5, desc: "Hickory dry-rub wings tossed in warm garlic butter." },
          { id: 5, cat: "Sides", name: "Parmesan Waffle Fries", price: 6.5, desc: "Crisp lattice cuts dusted with aged parmesan and chives." },
          { id: 6, cat: "Sides", name: "Honey-Butter Biscuits (4pc)", price: 5.5, desc: "Flaky warm biscuits finished with wildflower honey." },
        ]
      : [
          { id: 1, cat: "Signatures", name: "House Roasted Entree", price: 16.5, desc: "Chef's daily preparation with roasted herbs and reduction." },
          { id: 2, cat: "Signatures", name: "Crispy Artisan Sandwich", price: 14.0, desc: "Toasted brioche, house slaw, pickles, signature sauce." },
          { id: 3, cat: "Grills", name: "Wood-Fired Platter", price: 19.0, desc: "Char-grilled seasonal cuts served with warm flatbread." },
          { id: 4, cat: "Sides", name: "Herb Roasted Potatoes", price: 6.5, desc: "Crisp golden edges with rosemary and flaky sea salt." },
        ];
    const categories = ["All", ...Array.from(new Set(menuItems.map((m) => m.cat)))];

    return {
      title: brandName,
      summary: `Built a sleek, interactive **${brandName}** storefront modeled on Nexora's clean hairline cards, segmented category switcher, live pickup order tray, and table booking.`,
      highlights: [
        `**Sleek Editorial Layout:** 3-zone sticky top bar, balanced hero + service hours panel, and refined typography (\`Plus Jakarta Sans\`, \`DM Sans\`, \`JetBrains Mono\`).`,
        "**Interactive Menu & Pickup Tray:** Filter by `" + categories.join(" / ") + "`, add items with live quantity steppers, and submit pickup orders with confirmation numbers.",
        "**Reservations & Dual Theme:** Validated table reservation form plus light / dark obsidian toggle.",
      ],
      html: `<!DOCTYPE html>
<html lang="en"${htmlThemeAttr}>
<head>
${buildNexoraHeadAssets(brandName, scheme)}
    .hero {
      padding: clamp(28px, 4.5vw, 52px) 0 32px;
      display: grid; grid-template-columns: 1.25fr 0.75fr; gap: 24px; align-items: stretch;
    }
    @media (max-width: 740px) { .hero { grid-template-columns: 1fr; } }
    .hero-copy { display: flex; flex-direction: column; justify-content: center; }
    .hero h1 { font-size: clamp(1.85rem, 3.6vw, 2.65rem); line-height: 1.1; margin-bottom: 12px; }
    .hero p { font-size: 15px; color: var(--ink-2); max-width: 520px; margin-bottom: 20px; }
    .actions { display: flex; flex-wrap: wrap; gap: 10px; }
    .hours-card { padding: 18px 20px; display: flex; flex-direction: column; justify-content: space-between; }
    .hours-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
    .hours-head h2 { font-size: 14px; font-weight: 600; }
    .hours-row {
      display: flex; justify-content: space-between; font-size: 13px;
      padding: 8px 0; border-top: 1px solid var(--line); color: var(--ink-2);
    }
    .hours-row strong { color: var(--ink); font-family: var(--font-mono); font-size: 12.5px; font-variant-numeric: tabular-nums; font-weight: 500; }
    .section-head {
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
      gap: 12px; margin: 12px 0 16px; padding-top: 20px; border-top: 1px solid var(--line);
    }
    .section-head h2 { font-size: 19px; font-weight: 600; }
    .menu-layout { display: grid; grid-template-columns: 1.55fr 1fr; gap: 18px; align-items: start; margin-bottom: 32px; }
    @media (max-width: 820px) { .menu-layout { grid-template-columns: 1fr; } }
    .menu-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(225px, 1fr)); gap: 12px; }
    .dish {
      padding: 16px; display: flex; flex-direction: column; justify-content: space-between; gap: 14px;
      transition: transform 160ms var(--ease), box-shadow 160ms var(--ease);
    }
    .dish:hover { transform: translateY(-2px); box-shadow: var(--shadow-raised); }
    .dish-meta { display: flex; justify-content: space-between; align-items: center; font-size: 11.5px; color: var(--ink-3); margin-bottom: 6px; }
    .dish-top { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; }
    .dish h3 { font-size: 15px; font-weight: 600; }
    .price { font-family: var(--font-mono); font-size: 13.5px; font-weight: 500; color: var(--accent-ink); font-variant-numeric: tabular-nums; }
    .dish p { font-size: 13px; color: var(--ink-2); margin-top: 5px; }
    .add-btn { width: 100%; height: 34px; font-size: 13px; }
    .cart { padding: 18px; position: sticky; top: 72px; }
    .cart-head { display: flex; align-items: center; justify-content: space-between; padding-bottom: 10px; border-bottom: 1px solid var(--line); margin-bottom: 12px; }
    .cart h3 { font-size: 15px; font-weight: 600; }
    .cart-badge { font-family: var(--font-mono); font-size: 12px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
    .cart-list { display: flex; flex-direction: column; gap: 8px; margin-bottom: 14px; }
    .cart-item {
      display: flex; justify-content: space-between; align-items: center; gap: 8px;
      font-size: 13px; padding-bottom: 8px; border-bottom: 1px solid var(--line);
    }
    .qty-ctrl { display: inline-flex; align-items: center; gap: 6px; }
    .qty-btn {
      width: 24px; height: 24px; border-radius: 6px; border: none;
      background: var(--inset); box-shadow: var(--shadow-btn); color: var(--ink);
      font-family: var(--font-mono); font-weight: 500;
    }
    .qty-btn:hover { background: var(--hover); }
    .total-row { display: flex; justify-content: space-between; font-weight: 600; font-size: 14px; margin-bottom: 12px; }
    .reserve { padding: 20px; }
    .reserve h2 { font-size: 17px; font-weight: 600; margin-bottom: 4px; }
    .reserve p { font-size: 13.5px; color: var(--ink-2); margin-bottom: 14px; }
    .form-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; align-items: center; }
    .status { margin-top: 10px; font-size: 13px; color: var(--accent-ink); font-weight: 500; min-height: 18px; }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="container topbar-inner">
      <a class="brand" href="#">
        <span class="brand-mark">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3c2.5 3.2 4 5.8 4 8.5a4 4 0 1 1-8 0c0-1.9.9-3.9 2.3-5.8.3 1.4 1.1 2.2 1.7 2.8z"/></svg>
        </span>
        <span>${safeBrand}</span>
      </a>
      <nav class="nav-center" aria-label="Main navigation">
        <a class="nav-link" href="#menu">Menu</a>
        <a class="nav-link" href="#reserve">Reservations</a>
      </nav>
      <div class="nav-actions">
        <button type="button" class="icon-btn" id="themeBtn" aria-label="Toggle theme">${SUN_MOON_ICON_SVG}</button>
        <a class="btn btn-sm btn-primary" href="#menu">Order (<span id="navCount" class="num">0</span>)</a>
      </div>
    </div>
  </header>

  <main class="container">
    <section class="hero">
      <div class="hero-copy">
        <h1>${safeBrand}</h1>
        <p>${escapeHtmlText(tagline)}</p>
        <div class="actions">
          <a class="btn btn-primary" href="#menu">Browse Menu &amp; Order</a>
          <a class="btn btn-secondary" href="#reserve">Reserve a Table</a>
        </div>
      </div>
      <aside class="surface-card hours-card">
        <div class="hours-head">
          <h2>Kitchen Hours &amp; Pickup</h2>
          <span class="mono" style="font-size:11.5px;color:var(--green);">Open daily</span>
        </div>
        <div>
          <div class="hours-row"><span>Mon – Thu</span><strong>11:00 – 22:00</strong></div>
          <div class="hours-row"><span>Fri – Sat</span><strong>11:00 – 23:30</strong></div>
          <div class="hours-row"><span>Sunday</span><strong>12:00 – 21:30</strong></div>
          <div class="hours-row"><span>Express Pickup</span><strong>~15 min prep</strong></div>
        </div>
      </aside>
    </section>

    <section id="menu">
      <div class="section-head">
        <h2>Signature Menu</h2>
        <div class="seg-tabs" id="filters" role="tablist" aria-label="Menu categories"></div>
      </div>
      <div class="menu-layout">
        <div class="menu-grid" id="menuGrid"></div>
        <aside class="surface-card cart">
          <div class="cart-head">
            <h3>Pickup Order</h3>
            <span class="cart-badge" id="itemBadge">0 items</span>
          </div>
          <div class="cart-list" id="cartList">
            <p style="font-size:13px;color:var(--ink-2);">Select dishes from the menu to start a pickup tray.</p>
          </div>
          <div class="total-row">
            <span>Subtotal</span>
            <span class="mono" id="cartTotal">$0.00</span>
          </div>
          <button type="button" class="btn btn-primary" style="width:100%;" id="checkoutBtn">Place Pickup Order</button>
          <div class="status" id="orderStatus" role="status"></div>
        </aside>
      </div>
    </section>

    <section id="reserve" class="surface-card reserve">
      <h2>Reserve a Table</h2>
      <p>Hold your table in seconds—instant confirmation with zero booking fees.</p>
      <form id="reserveForm" class="form-row">
        <input id="guestName" class="field-input" type="text" placeholder="Your name" required />
        <select id="partySize" class="field-input">
          <option value="2">2 Guests</option>
          <option value="4" selected>4 Guests</option>
          <option value="6">6 Guests</option>
          <option value="8">8+ Guests</option>
        </select>
        <input id="resTime" class="field-input" type="time" value="19:00" required />
        <button class="btn btn-primary" type="submit">Confirm Table</button>
      </form>
      <div class="status" id="reserveNote" role="status"></div>
    </section>

${wantsPricingSection ? `${PRICING_ADDON_HTML}\n\n` : ""}    <footer class="site-footer">
      <span>&copy; ${new Date().getFullYear()} ${safeBrand}</span>
      <span>Scratch kitchen · Prepared fresh to order</span>
    </footer>
  </main>

  ${buildNexoraThemeScript(preferDark)}
  <script>
    const MENU = ${JSON.stringify(menuItems)};
    const CATEGORIES = ${JSON.stringify(categories)};
    let activeCat = "All";
    const cart = new Map();

    const filtersEl = document.getElementById("filters");
    const gridEl = document.getElementById("menuGrid");
    const cartListEl = document.getElementById("cartList");
    const cartTotalEl = document.getElementById("cartTotal");
    const navCountEl = document.getElementById("navCount");
    const itemBadgeEl = document.getElementById("itemBadge");
    const orderStatusEl = document.getElementById("orderStatus");

    function renderFilters() {
      filtersEl.innerHTML = CATEGORIES.map((c) =>
        '<button type="button" class="seg-btn ' + (c === activeCat ? "active" : "") + '" data-cat="' + c + '">' + c + '</button>'
      ).join("");
    }

    function renderMenu() {
      const visible = activeCat === "All" ? MENU : MENU.filter((m) => m.cat === activeCat);
      gridEl.innerHTML = visible.map((item, idx) => {
        const qty = cart.get(item.id) || 0;
        return '<article class="surface-card dish">' +
          '<div>' +
            '<div class="dish-meta"><span>' + String(idx + 1).padStart(2, "0") + ' · ' + item.cat + '</span></div>' +
            '<div class="dish-top"><h3>' + item.name + '</h3><span class="price">$' + item.price.toFixed(2) + '</span></div>' +
            '<p>' + item.desc + '</p>' +
          '</div>' +
          '<button type="button" class="btn btn-secondary add-btn" data-add="' + item.id + '">' +
            (qty > 0 ? 'Add another (' + qty + ' in tray)' : '+ Add to order') +
          '</button>' +
        '</article>';
      }).join("");
    }

    function renderCart() {
      let total = 0;
      let count = 0;
      const rows = [];
      cart.forEach((qty, id) => {
        const item = MENU.find((m) => m.id === id);
        if (!item) return;
        total += item.price * qty;
        count += qty;
        rows.push(
          '<div class="cart-item">' +
            '<div><strong>' + item.name + '</strong><div class="mono" style="font-size:12px;color:var(--ink-2);">$' + (item.price * qty).toFixed(2) + '</div></div>' +
            '<div class="qty-ctrl">' +
              '<button type="button" class="qty-btn" data-dec="' + id + '">-</button>' +
              '<span class="mono" style="min-width:16px;text-align:center;">' + qty + '</span>' +
              '<button type="button" class="qty-btn" data-inc="' + id + '">+</button>' +
            '</div>' +
          '</div>'
        );
      });
      navCountEl.textContent = String(count);
      itemBadgeEl.textContent = count + " item" + (count === 1 ? "" : "s");
      if (count === 0) {
        cartListEl.innerHTML = '<p style="font-size:13px;color:var(--ink-2);">Select dishes from the menu to start a pickup tray.</p>';
        cartTotalEl.textContent = "$0.00";
        return;
      }
      cartListEl.innerHTML = rows.join("");
      cartTotalEl.textContent = "$" + total.toFixed(2);
    }

    filtersEl.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-cat]");
      if (!btn) return;
      activeCat = btn.dataset.cat;
      renderFilters();
      renderMenu();
    });

    gridEl.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-add]");
      if (!btn) return;
      const id = Number(btn.dataset.add);
      cart.set(id, (cart.get(id) || 0) + 1);
      orderStatusEl.textContent = "";
      renderCart();
      renderMenu();
    });

    cartListEl.addEventListener("click", (e) => {
      const inc = e.target.closest("[data-inc]");
      const dec = e.target.closest("[data-dec]");
      if (inc) {
        const id = Number(inc.dataset.inc);
        cart.set(id, (cart.get(id) || 0) + 1);
      } else if (dec) {
        const id = Number(dec.dataset.dec);
        const next = (cart.get(id) || 0) - 1;
        if (next <= 0) cart.delete(id);
        else cart.set(id, next);
      } else return;
      renderCart();
      renderMenu();
    });

    document.getElementById("checkoutBtn").addEventListener("click", () => {
      if (cart.size === 0) {
        orderStatusEl.textContent = "Select at least one dish first.";
        return;
      }
      const total = cartTotalEl.textContent;
      const orderNum = Math.floor(1000 + Math.random() * 9000);
      cart.clear();
      renderCart();
      renderMenu();
      orderStatusEl.textContent = "Order #" + orderNum + " confirmed (" + total + ") — preparing now.";
    });

    document.getElementById("reserveForm").addEventListener("submit", (e) => {
      e.preventDefault();
      const name = document.getElementById("guestName").value.trim();
      const size = document.getElementById("partySize").value;
      const time = document.getElementById("resTime").value;
      document.getElementById("reserveNote").textContent =
        "Reservation confirmed for " + name + " (" + size + " guests at " + time + ").";
      e.target.reset();
    });

    renderFilters();
    renderMenu();
    renderCart();
  </script>
</body>
</html>`,
    };
  }

  // Sleek Nexora-Styled Landing / Starter / Portfolio / Pricing Page
  const pageTitle = extractTopicTitle(
    normalizedPrompt,
    isPortfolio ? "Alex Morgan" : isPricing ? "Nexora Plans" : "Modern Starter Page"
  );
  const safeTitle = escapeHtmlText(pageTitle);
  const heroHeadline = isPortfolio
    ? "Designing fast, thoughtful software for the web."
    : isPricing
      ? "Clear plans built to grow with your team."
      : pageTitle !== "Modern Starter Page"
        ? `Clean, high-craft interface for ${pageTitle}.`
        : "Clean surfaces, hairline precision, zero visual noise.";
  const heroSubtitle = isPortfolio
    ? "Full-stack engineer and interface designer focused on accessible products, clean architecture, and crisp typography."
    : isPricing
      ? "Start with full core capabilities on a transparent plan, and scale capacity only when your workload grows."
      : "Built with Nexora's cool neutral palette, layered hairline elevation, segmented controls, and responsive CSS architecture.";

  const panelTabs = isPricing
    ? [
        {
          id: "starter",
          label: "For individuals",
          heading: "Starter Plan — $0 / month",
          desc: "Ideal for personal prototypes, interactive HTML previews, and local development.",
          metrics: [
            { k: "Workspaces", v: "3 active" },
            { k: "Build checks", v: "Unlimited" },
            { k: "Theme tokens", v: "Light + Obsidian" },
          ],
        },
        {
          id: "team",
          label: "For scaling teams",
          heading: "Team Plan — $24 / seat",
          desc: "Built for production web apps needing shared component tokens and team review workflows.",
          metrics: [
            { k: "Workspaces", v: "Unlimited" },
            { k: "Deploy targets", v: "Global edge" },
            { k: "Priority SLA", v: "99.95%" },
          ],
        },
      ]
    : [
        {
          id: "design",
          label: "Design System",
          heading: "Layered hairline depth without visual clutter",
          desc: "Cool near-white and matte obsidian surfaces paired with subtle 1px borders and Plus Jakarta Sans display tracking.",
          metrics: [
            { k: "Canvas", v: "#fbfbfc / #18191d" },
            { k: "Radii", v: "8px – 14px" },
            { k: "Motion", v: "200ms ease-out" },
          ],
        },
        {
          id: "code",
          label: "Architecture",
          heading: "Zero-dependency semantic HTML5 & CSS Grid",
          desc: "Clean custom properties decouple theme colors from layout structure so customizing light or dark palettes stays trivial.",
          metrics: [
            { k: "Bundle", v: "Single file" },
            { k: "Layout", v: "12-col bento" },
            { k: "Numbers", v: "tabular-nums" },
          ],
        },
      ];

  const resolvedPalette = resolveRequestedPalette(lower, "cobalt");
  return {
    title: pageTitle,
    summary: `Built a sleek **\`index.html\`** for **${pageTitle}** matching Nexora's design system—layered hairline surfaces, \`Plus Jakarta Sans\` + \`DM Sans\` typography, asymmetric bento cards, segmented view switcher, and light/obsidian theme toggle.`,
    highlights: [
      "**Nexora Aesthetic & Token System:** Cool near-white canvas (`#fbfbfc`), crisp white cards (`#ffffff`), subtle `1px` hairlines (`#eceef1`), layered shadows, and matte obsidian dark mode (`#18191d`).",
      "**Asymmetric Bento & Segmented Switcher:** Interactive `Design System / Architecture` tabs update live specifications alongside numbered `01 / 02` capability cards.",
      "**Single-Row 3-Zone Header & Interactive Form:** Sticky blurred top bar (`56px`), accessible focus rings, and instant contact form feedback.",
    ],
    html: `<!DOCTYPE html>
<html lang="en"${htmlThemeAttr}>
<head>
${buildNexoraHeadAssets(pageTitle, resolvedPalette)}
    .hero { padding: clamp(32px, 5vw, 58px) 0 32px; max-width: 680px; }
    .hero h1 { font-size: clamp(1.9rem, 3.8vw, 2.85rem); line-height: 1.1; margin-bottom: 14px; }
    .hero p { font-size: 15.5px; color: var(--ink-2); margin-bottom: 22px; max-width: 560px; }
    .actions { display: flex; flex-wrap: wrap; gap: 10px; }
    .bento {
      display: grid; grid-template-columns: repeat(12, 1fr); gap: 12px;
      margin-bottom: 28px;
    }
    .span-7 { grid-column: span 7; }
    .span-5 { grid-column: span 5; }
    .span-6 { grid-column: span 6; }
    @media (max-width: 760px) { .span-7, .span-5, .span-6 { grid-column: span 12; } }
    .bento-card { padding: 20px; display: flex; flex-direction: column; justify-content: space-between; gap: 14px; }
    .card-top { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .num-idx { font-family: var(--font-mono); font-size: 12px; color: var(--ink-3); font-variant-numeric: tabular-nums; }
    .bento-card h2 { font-size: 16px; font-weight: 600; margin-bottom: 6px; }
    .bento-card p { font-size: 13.5px; color: var(--ink-2); }
    .spec-strip {
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
      padding-top: 14px; border-top: 1px solid var(--line);
    }
    @media (max-width: 520px) { .spec-strip { grid-template-columns: 1fr; } }
    .spec-item span { display: block; font-size: 11.5px; color: var(--ink-3); }
    .spec-item strong { font-family: var(--font-mono); font-size: 12.5px; font-weight: 500; color: var(--ink); }
    .contact { padding: 20px; margin-bottom: 12px; }
    .contact h2 { font-size: 16.5px; font-weight: 600; margin-bottom: 4px; }
    .contact p { font-size: 13.5px; color: var(--ink-2); margin-bottom: 14px; }
    .form-row { display: flex; flex-wrap: wrap; gap: 10px; }
    .form-row .field-input { flex: 1 1 200px; }
    .note { margin-top: 10px; font-size: 13px; color: var(--green); font-weight: 500; min-height: 18px; }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="container topbar-inner">
      <a class="brand" href="#">
        <span class="brand-mark">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 19V5l14 14V5"/></svg>
        </span>
        <span>${safeTitle}</span>
      </a>
      <nav class="nav-center" aria-label="Main navigation">
        <a class="nav-link" href="#overview">Overview</a>
        <a class="nav-link" href="#contact">Contact</a>
      </nav>
      <div class="nav-actions">
        <button type="button" class="icon-btn" id="themeBtn" aria-label="Toggle theme">${SUN_MOON_ICON_SVG}</button>
        <a class="btn btn-sm btn-primary" href="#contact">Get in touch</a>
      </div>
    </div>
  </header>

  <main class="container">
    <section class="hero">
      <h1>${escapeHtmlText(heroHeadline)}</h1>
      <p>${escapeHtmlText(heroSubtitle)}</p>
      <div class="actions">
        <a class="btn btn-primary" href="#overview">Explore architecture</a>
        <a class="btn btn-secondary" href="#contact">Send a note</a>
      </div>
    </section>

    <section id="overview" class="bento">
      <article class="surface-card bento-card span-7">
        <div class="card-top">
          <div class="seg-tabs" id="specTabs" role="tablist" aria-label="Architecture view">
            <button type="button" class="seg-btn active" data-tab="0">${escapeHtmlText(panelTabs[0].label)}</button>
            <button type="button" class="seg-btn" data-tab="1">${escapeHtmlText(panelTabs[1].label)}</button>
          </div>
          <span class="num-idx">Interactive</span>
        </div>
        <div>
          <h2 id="panelHeading">${escapeHtmlText(panelTabs[0].heading)}</h2>
          <p id="panelDesc">${escapeHtmlText(panelTabs[0].desc)}</p>
        </div>
        <div class="spec-strip" id="panelMetrics"></div>
      </article>

      <article class="surface-card bento-card span-5">
        <div class="card-top">
          <span class="num-idx">01</span>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" style="color:var(--accent-ink)"><rect x="3" y="3" width="18" height="18" rx="3"/><path d="M3 9h18M9 21V9"/></svg>
        </div>
        <div>
          <h2>Precision Surface Hierarchy</h2>
          <p>Crisp white cards sit cleanly over a cool neutral page with hairline borders replacing heavy drop shadows.</p>
        </div>
      </article>

      <article class="surface-card bento-card span-6">
        <div class="card-top">
          <span class="num-idx">02</span>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" style="color:var(--ink-2)"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>
        </div>
        <div>
          <h2>Dual Light &amp; Obsidian Modes</h2>
          <p>Every token switches instantaneously between cool near-white daylight and deep obsidian without harsh pure black.</p>
        </div>
      </article>

      <article class="surface-card bento-card span-6">
        <div class="card-top">
          <span class="num-idx">03</span>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" style="color:var(--ink-2)"><path d="m16 18 6-6-6-6M8 6l-6 6 6 6"/></svg>
        </div>
        <div>
          <h2>Accessible Vanilla Interactions</h2>
          <p>Keyboard-focusable controls, balanced display headings, and tabular numeric alignment built with zero dependencies.</p>
        </div>
      </article>
    </section>

    <section id="contact" class="surface-card contact">
      <h2>Quick Message</h2>
      <p>Test the validated message composer right inside the preview.</p>
      <form id="contactForm" class="form-row">
        <input id="userEmail" class="field-input" type="email" placeholder="you@company.com" required />
        <input id="userMsg" class="field-input" type="text" placeholder="Tell us what you are building..." required />
        <button class="btn btn-primary" type="submit">Send message</button>
      </form>
      <div class="note" id="formNote" role="status"></div>
    </section>

${wantsPricingSection && !isPricing ? `${PRICING_ADDON_HTML.replace(/#reserve/g, "#contact")}\n\n` : ""}    <footer class="site-footer">
      <span>&copy; ${new Date().getFullYear()} ${safeTitle}</span>
      <span>Built with semantic HTML5, CSS tokens &amp; vanilla JavaScript</span>
    </footer>
  </main>

  ${buildNexoraThemeScript(preferDark)}
  <script>
    const PANELS = ${JSON.stringify(panelTabs)};
    function renderPanel(idx) {
      const p = PANELS[idx] || PANELS[0];
      document.getElementById("panelHeading").textContent = p.heading;
      document.getElementById("panelDesc").textContent = p.desc;
      document.getElementById("panelMetrics").innerHTML = p.metrics
        .map((m) => '<div class="spec-item"><span>' + m.k + '</span><strong>' + m.v + '</strong></div>')
        .join("");
    }
    document.getElementById("specTabs").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-tab]");
      if (!btn) return;
      document.querySelectorAll("#specTabs .seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderPanel(Number(btn.dataset.tab));
    });
    renderPanel(0);

    document.getElementById("contactForm").addEventListener("submit", (event) => {
      event.preventDefault();
      const email = document.getElementById("userEmail").value.trim();
      const note = document.getElementById("formNote");
      note.textContent = "Received message from " + email + " — thanks for reaching out.";
      event.target.reset();
    });
  </script>
</body>
</html>`,
  };
}

function detectTargetCodeLang(lower: string): "python" | "typescript" | "javascript" {
  if (/\b(python|py)\b/.test(lower)) return "python";
  if (/\b(javascript|js|node)\b/.test(lower) && !/\btypescript\b/.test(lower)) {
    return "javascript";
  }
  return "typescript";
}

function synthesizeDynamicProgram(
  prompt: string,
  history: AssistantHistoryTurn[] = []
): {
  markdown: string;
  reasoningNote: string;
  titleHint: string;
} | null {
  const clean = prompt.trim();
  const lower = clean.toLowerCase();

  // Check if this is a multi-turn code refinement on a previous coding request
  const isCodeFollowUp =
    /\b(instead|only\s+(?:even|odd)|now\s+(?:from|to|make|in|count)|change\s+(?:it\s+)?to|convert\s+(?:that|it)\s+to|make\s+it\s+(?:python|typescript|javascript|even|odd|1\s+to\s+\d+)|add\s+(?:reverse|backwards)|count\s+backwards)\b/i.test(
      lower
    ) && history.length > 0;

  let effectivePrompt = clean;
  if (isCodeFollowUp) {
    const priorCodeUserTurn = [...history]
      .reverse()
      .find(
        (t) =>
          t.role === "user" &&
          /\b(count|sum|fibonacci|factorial|prime|fizzbuzz|multiplication\s+table|python|typescript|javascript)\b/i.test(
            t.content
          )
      );
    const hasLangInCurr = /\b(python|py|typescript|ts|javascript|js|node)\b/i.test(lower);
    if (priorCodeUserTurn) {
      // Carry forward the original algorithmic intent unless overridden by current prompt
      const priorLangMatch = /\b(python|typescript|javascript)\b/i.exec(priorCodeUserTurn.content);
      const baseContext = priorCodeUserTurn.content.replace(/\b\d+\s*(?:to|-)\s*\d+\b/g, (m) =>
        /\d+\s*(?:to|through|until|-|\.\.)\s*\d+/i.test(lower) ? "" : m
      );
      effectivePrompt = `${baseContext} ${!hasLangInCurr && priorLangMatch ? priorLangMatch[1] : ""} ${clean}`.trim();
    }
  }

  const effLower = effectivePrompt.toLowerCase();
  const isBuildOrCode =
    /\b(make|write|create|build|generate|code|program|script|function|python|typescript|javascript|loop|count)\b/i.test(
      effLower
    );
  if (!isBuildOrCode) return null;

  const lang = detectTargetCodeLang(effLower);
  const langLabel =
    lang === "python" ? "Python" : lang === "javascript" ? "JavaScript" : "TypeScript";

  // 1. Counting / Number Range Loop (e.g. "count 1 to 10", "count down 10 to 1", "even numbers 1 to 20")
  const rangeMatches = Array.from(
    effLower.matchAll(
      /\b(?:count(?:ing)?(?:\s+down)?|print|list|display|numbers?)\b[\s\S]{0,24}?(?:from\s+)?(-?\d+)\s*(?:to|through|until|-|\.\.)\s*(-?\d+)/gi
    )
  );
  const currOnlyRange = /(-?\d+)\s*(?:to|through|until|-|\.\.)\s*(-?\d+)/i.exec(lower);
  const upToMatch = /\b(?:count(?:ing)?(?:\s+down)?|up)\s+(?:to|until)\s+(-?\d+)/i.exec(effLower);
  const rangeMatch =
    currOnlyRange ??
    (rangeMatches.length > 0 ? rangeMatches[rangeMatches.length - 1] : null) ??
    upToMatch;

  if (rangeMatch) {
    const start = rangeMatch[2] !== undefined ? Number(rangeMatch[1]) : 1;
    const end = rangeMatch[2] !== undefined ? Number(rangeMatch[2]) : Number(rangeMatch[1]);
    const onlyEven = /\beven\b/.test(effLower);
    const onlyOdd = /\bodd\b/.test(effLower);
    const reverse = start > end || /\b(down|backwards|reverse)\b/.test(effLower);
    const lo = reverse && start < end ? end : start;
    const hi = reverse && start < end ? start : end;
    const step = lo > hi ? -1 : 1;
    const filterNote = onlyEven ? " (even only)" : onlyOdd ? " (odd only)" : "";

    let code = "";
    if (lang === "python") {
      const stop = hi + step;
      if (onlyEven || onlyOdd) {
        const rem = onlyEven ? 0 : 1;
        code = `for i in range(${lo}, ${stop}, ${step}):\n    if i % 2 == ${rem}:\n        print(i)`;
      } else {
        code = `for i in range(${lo}, ${stop}${step === -1 ? ", -1" : ""}):\n    print(i)`;
      }
    } else if (lang === "typescript") {
      const cmp = step > 0 ? "i <= " + hi : "i >= " + hi;
      const upd = step > 0 ? "i++" : "i--";
      const cond = onlyEven
        ? `\n    if (i % 2 === 0) console.log(i);`
        : onlyOdd
          ? `\n    if (Math.abs(i % 2) === 1) console.log(i);`
          : `\n    console.log(i);`;
      code = `function countRange(start = ${lo}, end = ${hi}): void {\n  for (let i = start; ${cmp}; ${upd}) {${cond}\n  }\n}\n\ncountRange();`;
    } else {
      const cmp = step > 0 ? "i <= " + hi : "i >= " + hi;
      const upd = step > 0 ? "i++" : "i--";
      const cond = onlyEven
        ? `\n  if (i % 2 === 0) console.log(i);`
        : onlyOdd
          ? `\n  if (Math.abs(i % 2) === 1) console.log(i);`
          : `\n  console.log(i);`;
      code = `for (let i = ${lo}; ${cmp}; ${upd}) {${cond}\n}`;
    }

    return {
      titleHint: `${langLabel}: Count ${lo} to ${hi}`,
      reasoningNote: `Extracted numeric range (${lo} -> ${hi}, step=${step}${filterNote}) and synthesized clean ${langLabel} loop code.`,
      markdown: [
        `Here is a clean **${langLabel}** script to count from **${lo}** to **${hi}**${filterNote}:`,
        "",
        `\`\`\`${lang}`,
        code,
        "```",
      ].join("\n"),
    };
  }

  // 2. Sum of a range ("sum 1 to 100")
  const sumMatch =
    /\bsum\b[\s\S]{0,20}?(?:from\s+)?(\d+)\s*(?:to|through|-|\.\.)\s*(\d+)/i.exec(lower);
  if (sumMatch) {
    const a = Number(sumMatch[1]);
    const b = Number(sumMatch[2]);
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    const total = ((hi - lo + 1) * (lo + hi)) / 2;
    const code =
      lang === "python"
        ? `total = sum(range(${lo}, ${hi + 1}))\nprint(f"Sum from ${lo} to ${hi} = {total}")`
        : `let total = 0;\nfor (let i = ${lo}; i <= ${hi}; i++) total += i;\nconsole.log(\`Sum from ${lo} to ${hi} = \${total}\`);`;
    return {
      titleHint: `${langLabel}: Sum ${lo} to ${hi}`,
      reasoningNote: `Synthesized ${langLabel} range summation (${lo}..${hi} = ${total}).`,
      markdown: [
        `Here is **${langLabel}** code to sum from **${lo}** to **${hi}** (result: \`${total}\`):`,
        "",
        `\`\`\`${lang}`,
        code,
        "```",
      ].join("\n"),
    };
  }

  // 3. Multiplication table
  const tableMatch =
    /\b(?:multiplication|times)\s+table\b[\s\S]{0,16}?(\d+)/i.exec(lower);
  if (tableMatch) {
    const n = Number(tableMatch[1]);
    const code =
      lang === "python"
        ? `n = ${n}\nfor i in range(1, 13):\n    print(f"{n} x {i} = {n * i}")`
        : `const n = ${n};\nfor (let i = 1; i <= 12; i++) {\n  console.log(\`\${n} x \${i} = \${n * i}\`);\n}`;
    return {
      titleHint: `${langLabel}: ${n} Times Table`,
      reasoningNote: `Synthesized multiplication table loop for n=${n} in ${langLabel}.`,
      markdown: [
        `Here is the **${n} times table** in **${langLabel}**:`,
        "",
        `\`\`\`${lang}`,
        code,
        "```",
      ].join("\n"),
    };
  }

  // 4. Fibonacci
  if (/\bfibonacci\b/i.test(lower)) {
    const nMatch = /(\d+)/.exec(lower);
    const n = Math.min(50, Math.max(2, nMatch ? Number(nMatch[1]) : 10));
    const code =
      lang === "python"
        ? `def fibonacci(count: int = ${n}) -> list[int]:\n    seq = []\n    a, b = 0, 1\n    for _ in range(count):\n        seq.append(a)\n        a, b = b, a + b\n    return seq\n\n\nprint(fibonacci(${n}))`
        : `function fibonacci(count = ${n}): number[] {\n  const seq: number[] = [];\n  let a = 0, b = 1;\n  for (let i = 0; i < count; i++) {\n    seq.push(a);\n    [a, b] = [b, a + b];\n  }\n  return seq;\n}\n\nconsole.log(fibonacci(${n}));`;
    return {
      titleHint: `${langLabel}: Fibonacci (${n})`,
      reasoningNote: `Generated O(n) iterative Fibonacci generator for ${n} terms in ${langLabel}.`,
      markdown: [
        `Here is an iterative **Fibonacci** sequence generator for **${n}** terms in **${langLabel}**:`,
        "",
        `\`\`\`${lang}`,
        code,
        "```",
      ].join("\n"),
    };
  }

  // 5. Factorial
  if (/\bfactorial\b/i.test(lower)) {
    const nMatch = /(\d+)/.exec(lower);
    const n = Math.min(20, Math.max(1, nMatch ? Number(nMatch[1]) : 5));
    const code =
      lang === "python"
        ? `def factorial(n: int) -> int:\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n\n\nprint(f"${n}! = {factorial(${n})}")`
        : `function factorial(n: number): number {\n  let result = 1;\n  for (let i = 2; i <= n; i++) result *= i;\n  return result;\n}\n\nconsole.log(\`${n}! = \${factorial(${n})}\`);`;
    return {
      titleHint: `${langLabel}: Factorial`,
      reasoningNote: `Synthesized iterative factorial function in ${langLabel}.`,
      markdown: [
        `Here is an iterative **factorial** implementation in **${langLabel}**:`,
        "",
        `\`\`\`${lang}`,
        code,
        "```",
      ].join("\n"),
    };
  }

  // 6. Prime numbers / check prime
  if (/\bprimes?\b/i.test(lower)) {
    const nMatch = /(\d+)/.exec(lower);
    const limit = Math.min(500, Math.max(10, nMatch ? Number(nMatch[1]) : 50));
    const code =
      lang === "python"
        ? `def is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    for d in range(2, int(n ** 0.5) + 1):\n        if n % d == 0:\n            return False\n    return True\n\n\nprimes = [n for n in range(2, ${limit + 1}) if is_prime(n)]\nprint(primes)`
        : `function isPrime(n: number): boolean {\n  if (n < 2) return false;\n  for (let d = 2; d * d <= n; d++) {\n    if (n % d === 0) return false;\n  }\n  return true;\n}\n\nconst primes = Array.from({ length: ${limit - 1} }, (_, i) => i + 2).filter(isPrime);\nconsole.log(primes);`;
    return {
      titleHint: `${langLabel}: Prime Numbers`,
      reasoningNote: `Synthesized O(n√n) prime finder up to ${limit} in ${langLabel}.`,
      markdown: [
        `Here is a **prime number** finder up to **${limit}** in **${langLabel}**:`,
        "",
        `\`\`\`${lang}`,
        code,
        "```",
      ].join("\n"),
    };
  }

  // 7. FizzBuzz
  if (/\bfizz\s*buzz\b/i.test(lower)) {
    const nMatch = /(\d+)/.exec(lower);
    const limit = Math.min(200, Math.max(5, nMatch ? Number(nMatch[1]) : 30));
    const code =
      lang === "python"
        ? `for i in range(1, ${limit + 1}):\n    out = ("Fizz" if i % 3 == 0 else "") + ("Buzz" if i % 5 == 0 else "")\n    print(out or i)`
        : `for (let i = 1; i <= ${limit}; i++) {\n  const out = (i % 3 === 0 ? "Fizz" : "") + (i % 5 === 0 ? "Buzz" : "");\n  console.log(out || i);\n}`;
    return {
      titleHint: `${langLabel}: FizzBuzz`,
      reasoningNote: `Synthesized FizzBuzz 1..${limit} in ${langLabel}.`,
      markdown: [
        `Here is **FizzBuzz** from **1** to **${limit}** in **${langLabel}**:`,
        "",
        `\`\`\`${lang}`,
        code,
        "```",
      ].join("\n"),
    };
  }

  // 8. Number guessing game
  if (/\bguess(?:ing)?\s+(?:the\s+)?number\b|\bnumber\s+guess/i.test(lower) && lang === "python") {
    return {
      titleHint: "Python: Number Guessing Game",
      reasoningNote: "Synthesized interactive Python CLI number guessing game with random target and attempt counter.",
      markdown: [
        "Here is an interactive **Number Guessing Game** in **Python**:",
        "",
        "```python",
        `import random

secret = random.randint(1, 100)
attempts = 0

print("Guess a number between 1 and 100!")
while True:
    raw = input("Your guess: ").strip()
    if not raw.isdigit():
        print("Please enter a valid whole number.")
        continue
    guess = int(raw)
    attempts += 1
    if guess < secret:
        print("Too low!")
    elif guess > secret:
        print("Too high!")
    else:
        print(f"Correct! You guessed {secret} in {attempts} attempts.")
        break`,
        "```",
      ].join("\n"),
    };
  }

  // 9. Hello World (only when explicitly requested)
  if (/\bhello\s+world\b/i.test(lower)) {
    const code =
      lang === "python"
        ? `print("Hello, World!")`
        : `console.log("Hello, World!");`;
    return {
      titleHint: `${langLabel}: Hello World`,
      reasoningNote: `Generated minimal ${langLabel} Hello World snippet.`,
      markdown: [
        `\`\`\`${lang}`,
        code,
        "```",
      ].join("\n"),
    };
  }

  return null;
}

function splitHtmlIntoAssets(html: string): {
  indexHtml: string;
  css: string;
  js: string;
} | null {
  const styleMatch = /<style>([\s\S]*?)<\/style>/i.exec(html);
  const scriptMatches = Array.from(html.matchAll(/<script>([\s\S]*?)<\/script>/gi));
  if (!styleMatch || scriptMatches.length === 0) return null;
  const css = styleMatch[1].trim();
  const js = scriptMatches.map((m) => m[1].trim()).join("\n\n");
  const indexHtml = html
    .replace(/<style>[\s\S]*?<\/style>/i, '  <link rel="stylesheet" href="./styles.css" />')
    .replace(/<script>[\s\S]*?<\/script>/gi, "")
    .replace(/<\/body>/i, '  <script src="./app.js" defer></script>\n</body>');
  return { indexHtml, css, js };
}

export function tryCodeAssistant(
  prompt: string,
  history: AssistantHistoryTurn[] = []
): {
  markdown: string;
  reasoningNote: string;
  titleHint: string;
} | null {
  const clean = prompt.trim();
  const lower = clean.toLowerCase();

  // A. Analyze user-pasted code block
  const fencedMatch = /```([a-zA-Z0-9_-]*)\s*\n([\s\S]*?)```/.exec(clean);
  if (fencedMatch) {
    const lang = fencedMatch[1] || "code";
    const code = fencedMatch[2];
    const lines = code.split("\n");
    const fnNames = Array.from(
      code.matchAll(/(?:function\s+([a-zA-Z0-9_$]+)|(?:const|let)\s+([a-zA-Z0-9_$]+)\s*=\s*(?:async\s*)?\()/g)
    )
      .map((m) => m[1] || m[2])
      .filter(Boolean);
    const loopCount = (code.match(/\b(for|while|do)\b/g) ?? []).length;
    const branchCount = (code.match(/\b(if|else\s+if|switch|case|\?\s*:)\b/g) ?? []).length;
    const hints: string[] = [];
    if (/[^=!]==[^=]/.test(code)) hints.push("Uses loose equality `==` — prefer strict `===` to avoid type coercion bugs.");
    if (/\bvar\s+/.test(code)) hints.push("Uses function-scoped `var` — prefer `const` or `let`.");
    if (/JSON\.parse\s*\(/.test(code) && !/try\s*\{/.test(code)) hints.push("Calls `JSON\\.parse()` without a surrounding `try/catch` block.");
    if (loopCount >= 2 && /\bfor\b[\s\S]{0,140}\bfor\b/.test(code)) hints.push("Detected nested loops — verify whether $O(n^2)$ can be reduced to $O(n)$ with a `Map` or `Set`.");
    if (/\bfetch\s*\(/.test(code) && !/res\.ok|response\.ok/.test(code)) hints.push("Uses `fetch()` — ensure `if (!res.ok)` is checked before parsing the response.");
    if (hints.length === 0) hints.push("No obvious static anti-patterns found; consider adding explicit return types and edge-case unit tests.");

    return {
      titleHint: `Code Review (${lang})`,
      reasoningNote: `Performed static AST-pattern scan across ${lines.length} lines (${fnNames.length} functions, ${loopCount} loops, ${branchCount} branches).`,
      markdown: [
        `### Static Code Inspection (${lang})`,
        "",
        `- **Size & Complexity:** ${lines.length} lines · ${fnNames.length} function(s) · ${loopCount} loop(s) · ${branchCount} conditional branch(es)`,
        fnNames.length > 0 ? `- **Detected symbols:** \`${fnNames.slice(0, 8).join("`, `")}\`` : "",
        "",
        "**Review Notes:**",
        ...hints.map((h) => `- ${h}`),
      ]
        .filter(Boolean)
        .join("\n"),
    };
  }

  // B. Interactive HTML Preview Widgets (zero external CDNs)
  if (
    /\b(pomodoro|countdown|stopwatch|timer)\b/i.test(lower) &&
    /\b(build|create|make|show|widget|html|preview|app)\b/i.test(lower)
  ) {
    return {
      titleHint: "Interactive Focus Timer",
      reasoningNote: "Designed state machine (idle/running/paused) and synthesized self-contained interactive timer widget inside nexora-preview.",
      markdown: [
        "Here is an interactive **Focus Timer** widget ready to use inline:",
        "",
        "```nexora-preview",
        `<div style="padding:18px;background:var(--surface,#fff);color:var(--ink,#1e2126);font-family:'DM Sans',system-ui,-apple-system,sans-serif;text-align:center;border-radius:10px;">
  <div style="font-size:12px;font-weight:500;color:var(--ink-2,#626773);">Focus Timer</div>
  <div id="t" style="font-family:'JetBrains Mono',ui-monospace,monospace;font-size:38px;font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-0.03em;margin:8px 0;">25:00</div>
  <div style="display:flex;justify-content:center;gap:8px;">
    <button id="go" style="height:34px;padding:0 16px;border-radius:8px;border:none;background:var(--accent,#2f76ff);color:#fff;cursor:pointer;font-size:12.5px;font-weight:500;">Start</button>
    <button id="rst" style="height:34px;padding:0 14px;border-radius:8px;border:1px solid var(--line-strong,#dfe2e7);background:var(--surface,#fff);color:var(--ink,#1e2126);cursor:pointer;font-size:12.5px;font-weight:500;">Reset</button>
  </div>
  <script>
    let sec = 1500, id = null;
    const t = document.getElementById('t'), go = document.getElementById('go');
    const draw = () => { const m = String(Math.floor(sec/60)).padStart(2,'0'), s = String(sec%60).padStart(2,'0'); t.textContent = m + ':' + s; };
    go.onclick = () => { if (id) { clearInterval(id); id = null; go.textContent = 'Start'; } else { go.textContent = 'Pause'; id = setInterval(() => { if (sec > 0) { sec--; draw(); } else { clearInterval(id); id = null; go.textContent = 'Start'; } }, 1000); } };
    document.getElementById('rst').onclick = () => { clearInterval(id); id = null; sec = 1500; go.textContent = 'Start'; draw(); };
  </script>
</div>`,
        "```",
      ].join("\n"),
    };
  }

  // C. Generative HTML5 / CSS / JS Page & UI Builder + Multi-turn HTML Refinements
  const hasPriorHtml = history.some(
    (h) => h.role === "assistant" && /```html\s*\n<!DOCTYPE html>/i.test(h.content)
  );
  const isHtmlFollowUpEdit =
    hasPriorHtml &&
    /\b(dark\s*mode|light\s*mode|obsidian|palette|color\s*scheme|coffee\s+palette|emerald\s+palette|violet\s+palette|pricing\s+section|split\s+into\s+separate|separate\s+css|rename\s+(?:it|the\s+brand|title)|change\s+(?:the\s+)?(?:name|title|brand)|make\s+it\s+(?:dark|light|warm|emerald|purple)|add\s+(?:a\s+)?pricing|switch\s+to\s+(?:dark|light))\b/i.test(
      lower
    );

  const wantsHtmlOrWebpage =
    isHtmlFollowUpEdit ||
    (/\b(html|webpage|web\s*page|website|web\s*site|landing\s*page|site|portfolio|login\s*page|dashboard|pricing\s*page|calculator|todo|to-do|task\s*app)\b/i.test(
      lower
    ) &&
      /\b(make|create|build|generate|write|design|give|show|want|need|simple|modern|responsive|interactive)\b/i.test(
        lower
      )) ||
    /^(?:a\s+)?simple\s+html\b/i.test(lower);

  if (wantsHtmlOrWebpage && !/^(what\s+is|define)\b/i.test(lower)) {
    let combinedPrompt = clean;
    if (isHtmlFollowUpEdit) {
      const priorHtmlUsers = history
        .filter((h) => h.role === "user")
        .map((h) => h.content.trim())
        .filter(
          (txt) =>
            /\b(html|webpage|web\s*page|website|site|portfolio|login|dashboard|pricing|calculator|todo|task|dark|light|palette|coffee|emerald|violet|rename|brand|title)\b/i.test(
              txt
            )
        );
      if (priorHtmlUsers.length > 0) {
        combinedPrompt = [...priorHtmlUsers, clean].join(" . ");
      }
    }

    const built = buildModernHtmlPage(combinedPrompt);
    if (/\bsplit\s+into\s+separate|\bseparate\s+css\b/i.test(lower)) {
      const split = splitHtmlIntoAssets(built.html);
      if (split) {
        return {
          titleHint: `${built.title} (Split)`.slice(0, 38),
          reasoningNote: `Refined "${built.title}" from conversation context and separated HTML markup, Nexora CSS tokens, and vanilla JS cleanly.`,
          markdown: [
            `Split **${built.title}** into modular \`index.html\`, \`styles.css\`, and \`app.js\` files:`,
            "",
            "**`index.html`**",
            "```html",
            split.indexHtml,
            "```",
            "",
            "**`styles.css`**",
            "```css",
            split.css,
            "```",
            "",
            "**`app.js`**",
            "```javascript",
            split.js,
            "```",
          ].join("\n"),
        };
      }
    }

    return {
      titleHint: built.title.slice(0, 38),
      reasoningNote: isHtmlFollowUpEdit
        ? `Updated the previous "${built.title}" build from conversation history with your requested refinements ("${clean}").`
        : `Planned semantic HTML5 landmarks, responsive CSS Grid & light/dark tokens, and interactive vanilla JS handlers for "${built.title}".`,
      markdown: [
        isHtmlFollowUpEdit
          ? `Updated **${built.title}** based on your earlier build (${clean}):`
          : built.summary,
        "",
        "```html",
        built.html,
        "```",
        "",
        "### What's included",
        ...built.highlights.map((h) => `- ${h}`),
        "",
        "Click **Preview** at the top of the code block to interact with it right here, or tell me what to tailor next (for example: *\"use a warm coffee palette\"*, *\"add a pricing section\"*, or *\"split into separate CSS & JS files\"*).",
      ].join("\n"),
    };
  }

  // D. Intent & Parameter-Aware Code Synthesizer (Python / TypeScript / JavaScript)
  const synthesized = synthesizeDynamicProgram(clean, history);
  if (synthesized) {
    return synthesized;
  }

  // E. Verified CS & Systems Templates (only when specifically requested)
  const algoTemplates: Array<{
    test: RegExp;
    title: string;
    lang: string;
    summary: string;
    complexity: string;
    code: string;
    usage?: string;
  }> = [
    {
      test: /\b(usefetch|react\s+hook|react\s+component|custom\s+hook)\b/i,
      title: "Typed React Data Hook",
      lang: "tsx",
      summary: "A reusable React + TypeScript pattern with abort-safe `fetch` and loading/error states.",
      complexity: "Abort-safe `AbortController` cleanup · Zero external libraries",
      code: `import { useEffect, useState } from "react";

export function useFetchJson<T>(url: string) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    setLoading(true);
    setError(null);

    fetch(url, { signal: ctrl.signal })
      .then((res) => {
        if (!res.ok) throw new Error(\`HTTP \${res.status}\`);
        return res.json() as Promise<T>;
      })
      .then((json) => setData(json))
      .catch((err) => {
        if (err?.name !== "AbortError") {
          setError(err instanceof Error ? err.message : "Request failed");
        }
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });

    return () => ctrl.abort();
  }, [url]);

  return { data, loading, error };
}`,
    },
    {
      test: /\b(express|rest\s*api|crud\s*api|backend\s*server|node\s*server)\b/i,
      title: "Express TypeScript REST API Starter",
      lang: "typescript",
      summary: "Production-ready Express CRUD router with validation, typed payloads, and proper HTTP status codes.",
      complexity: "Routes: `GET /api/items`, `POST /api/items`, `DELETE /api/items/:id`",
      code: `import express, { type Request, type Response } from "express";

interface Item {
  id: string;
  title: string;
  createdAt: number;
}

const app = express();
app.use(express.json());
const items = new Map<string, Item>();

app.get("/api/items", (_req: Request, res: Response) => {
  res.json({ items: Array.from(items.values()) });
});

app.post("/api/items", (req: Request, res: Response) => {
  const title = String(req.body?.title ?? "").trim();
  if (!title) return res.status(400).json({ error: "title is required" });
  const item: Item = { id: crypto.randomUUID(), title, createdAt: Date.now() };
  items.set(item.id, item);
  return res.status(201).json({ item });
});

app.delete("/api/items/:id", (req: Request, res: Response) => {
  const deleted = items.delete(req.params.id);
  return deleted ? res.status(204).end() : res.status(404).json({ error: "Not found" });
});

app.listen(3000, () => console.log("Listening on http://localhost:3000"));`,
    },
    {
      test: /\b(save\s+tasks|task\s+json|dataclass\s+task)\b/i,
      title: "Python Dataclass & JSON Script",
      lang: "python",
      summary: "Modular Python 3 script using type hints, `dataclasses`, and JSON file persistence.",
      complexity: "Standard library only (`json`, `pathlib`, `dataclasses`)",
      code: `from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class Task:
    id: int
    title: str
    done: bool = False


def save_tasks(path: Path, tasks: list[Task]) -> None:
    path.write_text(json.dumps([asdict(t) for t in tasks], indent=2), encoding="utf-8")


if __name__ == "__main__":
    tasks = [Task(id=1, title="Set up project", done=True), Task(id=2, title="Verify tests")]
    save_tasks(Path("tasks.json"), tasks)
    print(f"Saved {len(tasks)} tasks")`,
    },
    {
      test: /\b(rate\s*limit|token\s*bucket)\b/i,
      title: "Token-Bucket Rate Limiter in Pure TypeScript",
      lang: "typescript",
      summary: "Smooth per-key rate limiter using continuous token refill rather than spiky fixed windows.",
      complexity: "Time: **O(1)** per check · Space: **O(active keys)**",
      code: `export class TokenBucketLimiter {
  private readonly buckets = new Map<string, { tokens: number; updatedAt: number }>();

  constructor(
    private readonly capacity: number,
    private readonly refillPerSec: number
  ) {}

  allow(key: string, cost = 1): boolean {
    const now = Date.now() / 1000;
    const prev = this.buckets.get(key) ?? { tokens: this.capacity, updatedAt: now };
    const elapsed = Math.max(0, now - prev.updatedAt);
    const refilled = Math.min(this.capacity, prev.tokens + elapsed * this.refillPerSec);

    if (refilled < cost) {
      this.buckets.set(key, { tokens: refilled, updatedAt: now });
      return false;
    }
    this.buckets.set(key, { tokens: refilled - cost, updatedAt: now });
    return true;
  }
}`,
    },
    {
      test: /\bbinary\s+search\b/i,
      title: "Binary Search in Pure TypeScript",
      lang: "typescript",
      summary: "Overflow-safe iterative binary search returning the index of `target` or `-1`.",
      complexity: "Time: **O(log n)** · Space: **O(1)**",
      code: `export function binarySearch<T>(sorted: readonly T[], target: T): number {
  let lo = 0;
  let hi = sorted.length - 1;
  while (lo <= hi) {
    const mid = lo + ((hi - lo) >> 1);
    if (sorted[mid] === target) return mid;
    if (sorted[mid] < target) lo = mid + 1;
    else hi = mid - 1;
  }
  return -1;
}`,
    },
    {
      test: /\blru\s+cache\b/i,
      title: "O(1) LRU Cache in Pure TypeScript",
      lang: "typescript",
      summary: "Uses `Map` insertion ordering to achieve true `O(1)` eviction of least-recently-used entries.",
      complexity: "Time: **O(1)** `get`/`put` · Space: **O(capacity)**",
      code: `export class LRUCache<K, V> {
  private readonly map = new Map<K, V>();
  constructor(private readonly capacity: number) {}

  get(key: K): V | undefined {
    if (!this.map.has(key)) return undefined;
    const val = this.map.get(key)!;
    this.map.delete(key);
    this.map.set(key, val);
    return val;
  }

  put(key: K, value: V): void {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, value);
    if (this.map.size > this.capacity) {
      const oldestKey = this.map.keys().next().value;
      if (oldestKey !== undefined) this.map.delete(oldestKey);
    }
  }
}`,
    },
    {
      test: /\bquick\s*sort\b/i,
      title: "Quicksort in Pure TypeScript",
      lang: "typescript",
      summary: "In-place Hoare-style partition quicksort using median/middle pivot selection.",
      complexity: "Time: **O(n log n)** avg, **O(n²)** worst · Space: **O(log n)** stack",
      code: `export function quickSort(arr: number[], lo = 0, hi = arr.length - 1): number[] {
  if (lo >= hi) return arr;
  const pivot = arr[lo + ((hi - lo) >> 1)];
  let i = lo, j = hi;
  while (i <= j) {
    while (arr[i] < pivot) i++;
    while (arr[j] > pivot) j--;
    if (i <= j) {
      [arr[i], arr[j]] = [arr[j], arr[i]];
      i++;
      j--;
    }
  }
  if (lo < j) quickSort(arr, lo, j);
  if (i < hi) quickSort(arr, i, hi);
  return arr;
}`,
    },
    {
      test: /\bmerge\s*sort\b/i,
      title: "Stable Merge Sort in Pure TypeScript",
      lang: "typescript",
      summary: "Deterministic `O(n log n)` divide-and-conquer sort that preserves relative order of equal keys.",
      complexity: "Time: **O(n log n)** worst-case · Space: **O(n)**",
      code: `export function mergeSort(nums: readonly number[]): number[] {
  if (nums.length <= 1) return [...nums];
  const mid = nums.length >> 1;
  const left = mergeSort(nums.slice(0, mid));
  const right = mergeSort(nums.slice(mid));
  const out: number[] = [];
  let i = 0, j = 0;
  while (i < left.length && j < right.length) {
    out.push(left[i] <= right[j] ? left[i++] : right[j++]);
  }
  return out.concat(left.slice(i), right.slice(j));
}`,
    },
    {
      test: /\b(debounce|throttle)\b/i,
      title: "Typed Debounce Utility",
      lang: "typescript",
      summary: "Zero-dependency generic debounce preserving full TypeScript argument inference.",
      complexity: "Zero dependencies · Strict TypeScript inference",
      code: `export function debounce<Args extends unknown[]>(
  fn: (...args: Args) => void,
  waitMs: number
): (...args: Args) => void {
  let timer: ReturnType<typeof setTimeout> | undefined;
  return (...args: Args) => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => fn(...args), waitMs);
  };
}`,
    },
    {
      test: /\b(dijkstra|shortest\s+path)\b/i,
      title: "Dijkstra's Shortest Path in Pure TypeScript",
      lang: "typescript",
      summary: "Computes single-source shortest path distances across weighted non-negative directed graphs.",
      complexity: "Time: **O((V + E) log V)** with min-heap / **O(V²)** dense array",
      code: `export function dijkstra(
  graph: Record<string, Array<{ to: string; weight: number }>>,
  start: string
): Record<string, number> {
  const dist: Record<string, number> = { [start]: 0 };
  const visited = new Set<string>();
  const queue: Array<[number, string]> = [[0, start]];
  while (queue.length > 0) {
    queue.sort((a, b) => a[0] - b[0]);
    const [d, u] = queue.shift()!;
    if (visited.has(u)) continue;
    visited.add(u);
    for (const edge of graph[u] ?? []) {
      const alt = d + edge.weight;
      if (alt < (dist[edge.to] ?? Infinity)) {
        dist[edge.to] = alt;
        queue.push([alt, edge.to]);
      }
    }
  }
  return dist;
}`,
    },
  ];

  for (const tpl of algoTemplates) {
    if (tpl.test.test(lower)) {
      return {
        titleHint: tpl.title,
        reasoningNote: `Selected verified implementation pattern (${tpl.title}) ensuring edge-case handling and clean typing.`,
        markdown: [
          `### ${tpl.title}`,
          "",
          tpl.summary,
          "",
          `**Complexity / Notes:** ${tpl.complexity}`,
          "",
          `\`\`\`${tpl.lang}`,
          tpl.code,
          "```",
        ].join("\n"),
      };
    }
  }

  return null;
}

// ---------------------------------------------------------------------------
// 6. Structured Text Transformations (JSON, Table, Word Count, Case)
// ---------------------------------------------------------------------------
export function tryStructuredUtility(prompt: string): {
  markdown: string;
  reasoningNote: string;
  titleHint: string;
} | null {
  const clean = prompt.trim();
  const tableMatch =
    /^(?:make|convert|turn|create)\s+(?:a\s+)?(?:markdown\s+)?table\s+(?:from|of)\s*[:\-\n]+\s*([\s\S]+)$/i.exec(
      clean
    );
  if (tableMatch) {
    const lines = tableMatch[1]
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    if (lines.length >= 2) {
      const rows = lines.map((l) =>
        l.split(/[,|\t;]+/).map((cell) => cell.trim())
      );
      const colCount = Math.max(...rows.map((r) => r.length));
      if (colCount >= 2) {
        const norm = rows.map((r) =>
          Array.from({ length: colCount }, (_, i) => r[i] ?? "")
        );
        const header = `| ${norm[0].join(" | ")} |`;
        const sep = `| ${norm[0].map(() => "---").join(" | ")} |`;
        const body = norm.slice(1).map((r) => `| ${r.join(" | ")} |`);
        return {
          titleHint: "Markdown Table Output",
          reasoningNote: `Parsed ${norm.length} delimited rows across ${colCount} columns into GitHub-Flavored Markdown table.`,
          markdown: [header, sep, ...body].join("\n"),
        };
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// 7. Frontier-Style Structured Reasoning, Plans, Comparisons & Writing
// ---------------------------------------------------------------------------
export function tryStructuredReasoningAndWriting(prompt: string): {
  markdown: string;
  reasoningNote: string;
  titleHint: string;
} | null {
  const clean = prompt.trim();
  if (clean.length < 4) return null;

  // A. "X vs Y" Comparison Framework
  const vsMatch =
    /^(?:compare\s+)?([a-zA-Z0-9 .+#_-]{2,32}?)\s+(?:vs\.?|versus|compared\s+to)\s+([a-zA-Z0-9 .+#_-]{2,32})[?.!]*$/i.exec(
      clean
    );
  if (vsMatch) {
    const a = vsMatch[1].trim();
    const b = vsMatch[2].trim();
    return {
      titleHint: `${a} vs ${b}`.slice(0, 36),
      reasoningNote: `Structured side-by-side comparison criteria and decision matrix for "${a}" vs "${b}".`,
      markdown: [
        `### ${a} vs ${b} — Structured Comparison`,
        "",
        `When choosing between **${a}** and **${b}**, evaluate them across four practical dimensions:`,
        "",
        `| Dimension | ${a} | ${b} |`,
        `| --- | --- | --- |`,
        `| **Primary Fit** | Best when aligning with core strengths of ${a} | Best when workflows favor ${b} |`,
        `| **Learning Curve** | Compare ecosystem conventions & setup overhead | Compare tooling maturity & team familiarity |`,
        `| **Performance / Scale** | Benchmark latency & throughput under realistic load | Verify memory footprint & scaling limits |`,
        `| **Maintenance** | Check community support, updates & long-term cost | Check integration surface & operational complexity |`,
        "",
        "**Decision rule:** Pick the option that minimizes friction for your main bottleneck today, rather than optimizing for hypothetical future scale.",
      ].join("\n"),
    };
  }

  // B. Writing / Email / Proposal / Blog / README Drafting
  const writeMatch =
    /^(?:write|draft|compose|create)\s+(?:a|an|me\s+a|me\s+an)?\s*(email|message|cover\s+letter|proposal|outline|blog\s+post|readme|announcement|summary)\s*(?:about|for|to|on)?\s*([\s\S]*)$/i.exec(
      clean
    );
  if (writeMatch) {
    const kind = writeMatch[1].toLowerCase();
    const subject = (writeMatch[2] || "your topic").replace(/[?.!]+$/, "").trim();
    if (kind === "email" || kind === "message") {
      return {
        titleHint: `Email: ${subject}`.slice(0, 36),
        reasoningNote: `Drafted clear, direct ${kind} structure with subject line, context, call to action, and sign-off for "${subject}".`,
        markdown: [
          `Here is a crisp, ready-to-send draft for **${subject}**:`,
          "",
          `**Subject:** Quick update on ${subject}`,
          "",
          "Hi [Recipient Name],",
          "",
          `I'm reaching out regarding **${subject}**. Here are the key details so we can keep things moving smoothly:`,
          "",
          "- **Current status:** [1–2 sentences summarizing where things stand]",
          "- **Next step / ask:** [Clear action item or decision needed]",
          "- **Timeline:** [Target date or preferred window]",
          "",
          "Let me know if you have any questions or would like to adjust anything.",
          "",
          "Best regards,  ",
          "[Your Name]",
        ].join("\n"),
      };
    }

    return {
      titleHint: `${kind.charAt(0).toUpperCase() + kind.slice(1)}: ${subject}`.slice(0, 36),
      reasoningNote: `Structured ${kind} template with hook, core sections, and clear takeaways for "${subject}".`,
      markdown: [
        `### Draft ${kind.charAt(0).toUpperCase() + kind.slice(1)}: ${subject}`,
        "",
        `#### 1. Overview`,
        `Introduce **${subject}**, why it matters right now, and the core outcome readers should expect.`,
        "",
        `#### 2. Key Points`,
        `- **Context & Problem:** Clarify what challenge **${subject}** addresses.`,
        `- **Approach / Solution:** Break down the main workflow, architecture, or insight into 2–3 concrete steps.`,
        `- **Impact & Results:** Highlight measurable benefits, examples, or trade-offs.`,
        "",
        `#### 3. Next Steps`,
        `Close with one clear recommendation or action item for the reader.`,
      ].join("\n"),
    };
  }

  // C. "How to / Plan / Roadmap / Learn" Guide
  const guideMatch =
    /^(?:how\s+(?:do\s+i|to|can\s+i)|steps\s+to|guide\s+to|plan\s+(?:to|for)|roadmap\s+(?:to|for))\s+([\s\S]+)$/i.exec(
      clean
    );
  if (guideMatch) {
    const goal = guideMatch[1].replace(/[?.!]+$/, "").trim();
    return {
      titleHint: `Guide: ${goal}`.slice(0, 36),
      reasoningNote: `Decomposed "${goal}" into structured phases: prerequisites, execution steps, and verification.`,
      markdown: [
        `### Step-by-Step Guide: ${goal}`,
        "",
        `1. **Define the target scope:** Narrow **${goal}** down to a concrete first milestone you can verify quickly.`,
        `2. **Prepare inputs & environment:** Gather the tools, references, or constraints needed before starting.`,
        `3. **Build in small iterations:** Complete the core path first, test early, and avoid premature complexity.`,
        `4. **Validate & refine:** Check edge cases, measure the result against your goal, and polish what remains.`,
        "",
        `If you share your current stack, constraints, or starting point for **${goal}**, I can tailor exact steps or code.`,
      ].join("\n"),
    };
  }

  return null;
}

export interface AssistantHistoryTurn {
  role: "user" | "assistant";
  content: string;
}

function safeEvalArithmetic(expr: string): number | null {
  const tokens = expr.replace(/\s+/g, "").match(/\d+(?:\.\d+)?|[+\-*/^()]/g);
  if (!tokens || tokens.join("") !== expr.replace(/\s+/g, "")) return null;
  let pos = 0;

  const parseExpr = (): number => {
    let val = parseTerm();
    while (tokens[pos] === "+" || tokens[pos] === "-") {
      const op = tokens[pos++];
      const rhs = parseTerm();
      val = op === "+" ? val + rhs : val - rhs;
    }
    return val;
  };
  const parseTerm = (): number => {
    let val = parsePower();
    while (tokens[pos] === "*" || tokens[pos] === "/") {
      const op = tokens[pos++];
      const rhs = parsePower();
      if (op === "/" && rhs === 0) throw new Error("div0");
      val = op === "*" ? val * rhs : val / rhs;
    }
    return val;
  };
  const parsePower = (): number => {
    let val = parseFactor();
    while (tokens[pos] === "^") {
      pos++;
      val = Math.pow(val, parsePower());
    }
    return val;
  };
  const parseFactor = (): number => {
    if (tokens[pos] === "-") {
      pos++;
      return -parseFactor();
    }
    if (tokens[pos] === "+") {
      pos++;
      return parseFactor();
    }
    if (tokens[pos] === "(") {
      pos++;
      const inner = parseExpr();
      if (tokens[pos] !== ")") throw new Error("paren");
      pos++;
      return inner;
    }
    const num = Number(tokens[pos++]);
    if (!Number.isFinite(num)) throw new Error("nan");
    return num;
  };

  try {
    const res = parseExpr();
    return pos === tokens.length && Number.isFinite(res) ? res : null;
  } catch {
    return null;
  }
}

function extractLastNumericAnswer(history: AssistantHistoryTurn[]): number | null {
  for (let i = history.length - 1; i >= 0; i--) {
    if (history[i].role !== "assistant") continue;
    const content = history[i].content;
    const eqMatch = /=\s*\*\*(-?\d+(?:\.\d+)?)(?:°[CFK])?\*\*/i.exec(content);
    if (eqMatch) {
      const n = Number(eqMatch[1]);
      if (Number.isFinite(n)) return n;
    }
    const bareMatch = /^\*\*(-?\d+(?:\.\d+)?)\*\*$/m.exec(content.trim());
    if (bareMatch) {
      const n = Number(bareMatch[1]);
      if (Number.isFinite(n)) return n;
    }
  }
  return null;
}

export function tryQuickMathOrUnits(
  prompt: string,
  history: AssistantHistoryTurn[] = []
): string | null {
  const clean = prompt.trim();
  const lower = clean.toLowerCase();

  // Follow-up arithmetic on previous numeric result ("double that", "half", "+ 15", "multiply by 3")
  if (history.length > 0) {
    const prevVal = extractLastNumericAnswer(history);
    if (prevVal !== null) {
      if (/^(?:double\s+(?:that|it)|times\s+2|multiply\s+(?:that|it)\s+by\s+2)[?.!]*$/i.test(lower)) {
        return `\`${prevVal} * 2\` = **${Number((prevVal * 2).toFixed(8))}**`;
      }
      if (/^(?:half\s+(?:of\s+)?(?:that|it)|halve\s+(?:that|it)|divide\s+(?:that|it)\s+by\s+2)[?.!]*$/i.test(lower)) {
        return `\`${prevVal} / 2\` = **${Number((prevVal / 2).toFixed(8))}**`;
      }
      if (/^(?:square\s+(?:that|it)|(?:that|it)\s+squared)[?.!]*$/i.test(lower)) {
        return `\`${prevVal}^2\` = **${Number((prevVal * prevVal).toFixed(8))}**`;
      }
      if (/^(?:square\s+root\s+of\s+(?:that|it)|sqrt\s+(?:that|it))[?.!]*$/i.test(lower) && prevVal >= 0) {
        return `\`sqrt(${prevVal})\` = **${Number(Math.sqrt(prevVal).toFixed(8))}**`;
      }
      const opThatMatch =
        /^(?:and\s+|now\s+)?(?:(add|plus|\+|subtract|minus|-|multiply\s+by|times|\*|divide\s+by|\/)\s+(?:that|it\s+by\s+)?)?([+\-*/])\s*(-?\d+(?:\.\d+)?)[?.!]*$/i.exec(
          lower
        ) ??
        /^(?:and\s+|now\s+)?(add|plus|subtract|minus|multiply\s+by|times|divide\s+by)\s+(-?\d+(?:\.\d+)?)[?.!]*$/i.exec(
          lower
        );
      if (opThatMatch) {
        const rawOp = (opThatMatch[1] || opThatMatch[2] || "").toLowerCase();
        const rhs = Number(opThatMatch[3] ?? opThatMatch[2]);
        if (Number.isFinite(rhs)) {
          const sym =
            rawOp === "+" || rawOp === "add" || rawOp === "plus"
              ? "+"
              : rawOp === "-" || rawOp === "subtract" || rawOp === "minus"
                ? "-"
                : rawOp === "*" || rawOp === "times" || rawOp.startsWith("multiply")
                  ? "*"
                  : "/";
          if (sym === "/" && rhs === 0) return "Cannot divide by zero.";
          const val =
            sym === "+" ? prevVal + rhs : sym === "-" ? prevVal - rhs : sym === "*" ? prevVal * rhs : prevVal / rhs;
          return `\`${prevVal} ${sym} ${rhs}\` = **${Number(val.toFixed(8))}**`;
        }
      }
    }
  }

  // Percentage: "15% of 240"
  const pctMatch =
    /^(?:what\s+is\s+)?(-?\d+(?:\.\d+)?)\s*%\s+of\s+(-?\d+(?:\.\d+)?)[?.!]*$/i.exec(clean);
  if (pctMatch) {
    const pct = Number(pctMatch[1]);
    const base = Number(pctMatch[2]);
    const val = (pct / 100) * base;
    return `**${pct}% of ${base}** = **${Number(val.toFixed(6))}**`;
  }

  // Temperature: "72 f to c" / "25 c to f"
  const tempMatch =
    /^(?:convert\s+)?(-?\d+(?:\.\d+)?)\s*°?\s*(c|celsius|f|fahrenheit|k|kelvin)\s+(?:to|in)\s+°?\s*(c|celsius|f|fahrenheit|k|kelvin)[?.!]*$/i.exec(
      clean
    );
  if (tempMatch) {
    const val = Number(tempMatch[1]);
    const from = tempMatch[2][0].toLowerCase();
    const to = tempMatch[3][0].toLowerCase();
    const celsius = from === "c" ? val : from === "f" ? ((val - 32) * 5) / 9 : val - 273.15;
    const out = to === "c" ? celsius : to === "f" ? (celsius * 9) / 5 + 32 : celsius + 273.15;
    return `**${val}°${from.toUpperCase()}** = **${Number(out.toFixed(2))}°${to.toUpperCase()}**`;
  }

  // Unit conversions (length & mass)
  const UNIT_TO_METER: Record<string, number> = {
    m: 1, meter: 1, meters: 1,
    km: 1000, kilometer: 1000, kilometers: 1000,
    cm: 0.01, mm: 0.001,
    mi: 1609.344, mile: 1609.344, miles: 1609.344,
    ft: 0.3048, foot: 0.3048, feet: 0.3048,
    in: 0.0254, inch: 0.0254, inches: 0.0254,
  };
  const UNIT_TO_KG: Record<string, number> = {
    kg: 1, kilogram: 1, kilograms: 1,
    g: 0.001, gram: 0.001, grams: 0.001,
    lb: 0.45359237, lbs: 0.45359237, pound: 0.45359237, pounds: 0.45359237,
    oz: 0.0283495231, ounce: 0.0283495231, ounces: 0.0283495231,
  };
  const unitMatch =
    /^(?:convert\s+)?(-?\d+(?:\.\d+)?)\s*([a-z]+)\s+(?:to|in)\s+([a-z]+)[?.!]*$/i.exec(clean);
  if (unitMatch) {
    const val = Number(unitMatch[1]);
    const u1 = unitMatch[2].toLowerCase();
    const u2 = unitMatch[3].toLowerCase();
    if (UNIT_TO_METER[u1] && UNIT_TO_METER[u2]) {
      const res = (val * UNIT_TO_METER[u1]) / UNIT_TO_METER[u2];
      return `**${val} ${u1}** = **${Number(res.toFixed(4))} ${u2}**`;
    }
    if (UNIT_TO_KG[u1] && UNIT_TO_KG[u2]) {
      const res = (val * UNIT_TO_KG[u1]) / UNIT_TO_KG[u2];
      return `**${val} ${u1}** = **${Number(res.toFixed(4))} ${u2}**`;
    }
  }

  // Arithmetic expression: "what is (12 + 8) * 3?"
  const stripped = clean
    .replace(/^(?:what\s+is|calculate|compute|eval)\s+/i, "")
    .replace(/[=?.\s]+$/, "")
    .trim();
  if (/^[\d+\-*/^().\s]+$/.test(stripped) && /[+\-*/^]/.test(stripped)) {
    const ans = safeEvalArithmetic(stripped);
    if (ans !== null) {
      return `\`${stripped}\` = **${Number(ans.toFixed(8))}**`;
    }
  }

  return null;
}

export function resolveConversationalFollowUpTopic(
  prompt: string,
  history: AssistantHistoryTurn[] = []
): string {
  const clean = prompt.trim();
  if (!clean || history.length === 0) return clean;
  const isBareFollowUp =
    /^(?:tell\s+me\s+more|more\s+details?|go\s+on|continue|expand(?:\s+on\s+that)?|explain(?:\s+(?:that|it|more|simpler|simply))?|why\??|how\s+does\s+(?:that|it)\s+work|give\s+(?:me\s+)?an?\s+example|what\s+about\s+that|elaborate)[\s!?.]*$/i.test(
      clean
    ) ||
    (clean.split(/\s+/).length <= 5 && /\b(it|that|this|them|those)\b/i.test(clean));
  if (!isBareFollowUp) return clean;

  const priorTopicTurn = [...history]
    .reverse()
    .find((h) => h.role === "user" && h.content.trim().length >= 3);
  if (!priorTopicTurn) return clean;
  const keywords = tokenizeKeywords(priorTopicTurn.content).slice(0, 6).join(" ");
  return keywords ? `${keywords} ${clean}`.trim() : `${priorTopicTurn.content} ${clean}`.trim();
}


