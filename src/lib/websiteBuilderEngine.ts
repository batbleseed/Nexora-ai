import type { ChatMemoryFact, ThreadMemoryDigest } from "./chatMemory";
import type { AssistantHistoryTurn, ResearchPassage } from "./pureTsIntelligence";

export type WebsiteProfileKind =
  | "saas"
  | "dashboard"
  | "ecommerce"
  | "portfolio"
  | "restaurant"
  | "calculator"
  | "todo"
  | "login";

export type AccentScheme = "cobalt" | "ember" | "espresso" | "emerald" | "violet";

export interface WebsiteBuilderContext {
  history?: AssistantHistoryTurn[];
  memoryFacts?: ChatMemoryFact[];
  threadDigests?: ThreadMemoryDigest[];
  researchPassages?: ResearchPassage[];
  corpusSentences?: string[];
}

export interface WebsiteFactBrief {
  profile: WebsiteProfileKind;
  title: string;
  ownerName: string;
  domain: string;
  tagline: string;
  accent: AccentScheme;
  preferDark: boolean;
  wantsPricingAddon: boolean;
  connectedFacts: string[];
  extractedItems: string[];
  metrics: Array<{ label: string; value: string; note: string }>;
}

const SUN_MOON_ICON_SVG = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.93 4.93l1.41 1.41m11.32 11.32l1.41 1.41M2 12h2m16 0h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg>`;

export function escapeHtml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function normalizeWebTypoTerms(text: string): string {
  return text
    .replace(/\brestourant\b/gi, "restaurant")
    .replace(/\bresturant\b/gi, "restaurant")
    .replace(/\brestaraunt\b/gi, "restaurant")
    .replace(/\bwebsit\b/gi, "website")
    .replace(/\bchiken\b/gi, "chicken")
    .replace(/\becommerece\b/gi, "ecommerce")
    .replace(/\bportofolio\b/gi, "portfolio")
    .replace(/\bdasboard\b/gi, "dashboard");
}

function resolvePalette(lower: string, fallback: AccentScheme = "cobalt"): AccentScheme {
  if (/\b(coffee|espresso|warm\s+palette|mocha|caramel|brown)\b/i.test(lower)) return "espresso";
  if (/\b(emerald|mint|forest|green\s+palette)\b/i.test(lower)) return "emerald";
  if (/\b(violet|purple|lavender|plum)\b/i.test(lower)) return "violet";
  if (/\b(ember|orange|flame|coral)\b/i.test(lower)) return "ember";
  if (/\b(blue|cobalt)\b/i.test(lower)) return "cobalt";
  return fallback;
}

export function detectWebsiteProfile(prompt: string): WebsiteProfileKind {
  const lower = normalizeWebTypoTerms(prompt).toLowerCase();

  if (/\b(calculator|calc)\b/i.test(lower)) return "calculator";
  if (/\b(todo|to-do|task\s*list|tasks\s*app|checklist)\b/i.test(lower)) return "todo";
  if (/\b(login|sign\s*in|auth\s+page|register\s+page|signup\s+page)\b/i.test(lower)) return "login";

  // Check explicit profile requests in clear priority order
  const wantsDashboard =
    /\b(dashboard|analytics|admin\s+panel|admin|metrics|kpi|control\s+panel|crm|monitoring|ops\s+console|sidebar.*chart)\b/i.test(
      lower
    );
  const wantsEcommerce =
    /\b(e-?commerce|online\s+store|storefront|shop|marketplace|merch|boutique|catalog|product\s+cards|shopping\s+cart)\b/i.test(
      lower
    );
  const wantsPortfolio =
    /\b(portfolio|resume|cv|personal\s+site|personal\s+website|developer\s+site|designer\s+site|bio\s+page|freelance\s+site|my\s+work)\b/i.test(
      lower
    );
  const wantsRestaurant =
    /\b(chicken|restaurant|food|kitchen|grill|roast|wings|burger|pizza|cafe|bakery|bistro|diner|bbq|taco|sushi|noodle|steak|menu|dining)\b/i.test(
      lower
    );
  const wantsSaas =
    /\b(saas|landing\s+page|software|platform|startup|waitlist|pricing\s+page|testimonials|subscription|cloud|ai\s+product|app\s+landing)\b/i.test(
      lower
    );

  if (wantsDashboard && !wantsEcommerce && !wantsPortfolio) return "dashboard";
  if (wantsEcommerce) return "ecommerce";
  if (wantsPortfolio) return "portfolio";
  if (wantsRestaurant) return "restaurant";
  if (wantsSaas) return "saas";
  // Default rich website experience -> SaaS / Editorial Product Showcase instead of generic stub
  return "saas";
}

function cleanTitleCandidate(raw: string): string {
  const stripped = raw
    .replace(
      /^(?:a|an|the|my|our|new|simple|modern|responsive|clean|nice|cool|good|basic|complete|full|interactive|sleek|saas|ecommerce|e-commerce|portfolio|dashboard)\s+/gi,
      ""
    )
    .replace(
      /\b(?:using|with|from|in|html|css|js|javascript|website|web\s*site|webpage|web\s*page|landing\s*page|dashboard|portfolio|storefront|store|shop|site|page|app|please|dark\s+mode|light\s+mode|palette)\b.*$/i,
      ""
    )
    .trim();
  if (stripped.length < 2) return "";
  return stripped
    .split(/\s+/)
    .slice(0, 5)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ");
}

function extractListedPhrases(text: string): string[] {
  const results: string[] = [];
  const clauseRegex =
    /\b(?:featuring|including|selling|with|skills?|projects?|features?|products?|items?|about|covering|topics?)\s*[:\-]?\s*([^.\n]{6,180})/gi;
  let m: RegExpExecArray | null;
  while ((m = clauseRegex.exec(text)) !== null) {
    const parts = m[1]
      .split(/,|;|\band\b|\|/i)
      .map((p) =>
        p
          .replace(/^(?:a|an|the|some)\s+/i, "")
          .replace(/\b(?:using|in)\s+(?:html|css|js|nexora).*$/i, "")
          .trim()
      )
      .filter((p) => p.length >= 3 && p.length <= 52);
    for (const part of parts) {
      if (!/^(dark mode|light mode|pricing section|warm coffee palette|html|css|js)$/i.test(part)) {
        results.push(part);
      }
    }
  }
  return Array.from(new Set(results)).slice(0, 8);
}

function extractSentencesFromText(text: string): string[] {
  return text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/[#*`>]+/g, " ")
    .split(/(?<=[.!?])\s+|\n+/)
    .map((s) => s.replace(/\s+/g, " ").trim())
    .filter(
      (s) =>
        s.length >= 28 &&
        s.length <= 240 &&
        /[a-zA-Z]/.test(s) &&
        !/^(click\s+preview|here\s+is|what's\s+included|updated\s+\*\*|built\s+a\s+sleek)/i.test(s)
    );
}

function extractTopicKeywordsSimple(text: string): string[] {
  const stop = new Set([
    "a", "an", "the", "make", "build", "create", "design", "write", "give", "me",
    "for", "about", "with", "using", "from", "and", "or", "website", "webpage",
    "page", "site", "html", "css", "js", "saas", "landing", "dashboard", "ecommerce",
    "store", "shop", "portfolio", "app", "modern", "clean", "sleek", "dark", "mode",
    "light", "palette", "pricing", "section", "please", "want", "need", "some",
  ]);
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length >= 3 && !stop.has(w));
}

export function extractWebsiteFactBrief(
  prompt: string,
  context?: WebsiteBuilderContext
): WebsiteFactBrief {
  const normalized = normalizeWebTypoTerms(prompt);
  const lower = normalized.toLowerCase();
  const profile = detectWebsiteProfile(normalized);

  const darkMatches = Array.from(
    lower.matchAll(/\b(dark(?:\s+mode)?|light(?:\s+mode)?|obsidian|midnight|daylight)\b/gi)
  );
  const lastThemeKeyword =
    darkMatches.length > 0 ? darkMatches[darkMatches.length - 1][1].toLowerCase() : "";
  const preferDark = lastThemeKeyword ? /\b(dark|obsidian|midnight)\b/.test(lastThemeKeyword) : false;
  const wantsPricingAddon = /\b(pricing|plans|tiers)\b/i.test(lower);

  const memories = context?.memoryFacts ?? [];
  const savedName = memories.find((m) => m.id === "identity_user_name")?.value;
  const savedRole = memories.find((m) => m.id === "identity_user_role")?.value;
  const savedProject = memories.find((m) => m.category === "project")?.value;

  // Determine title / brand connected to prompt + memory
  let title = "";
  const explicitBrandMatches = Array.from(
    normalized.matchAll(
      /\b(?:called|named|titled|brand(?:ed)?|for\s+(?:my|our)?)\s+["']?([a-zA-Z0-9][a-zA-Z0-9\s'&.-]{1,42})["']?/gi
    )
  );
  for (let i = explicitBrandMatches.length - 1; i >= 0; i--) {
    const cand = cleanTitleCandidate(explicitBrandMatches[i][1]);
    if (
      cand &&
      !/^(saas|landing|dashboard|ecommerce|store|shop|portfolio|html|css|js|website|page|me)$/i.test(
        cand
      )
    ) {
      title = cand;
      break;
    }
  }
  if (!title) {
    const buildMatches = Array.from(
      normalized.matchAll(
        /\b(?:make|create|build|design|write|generate|want)\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?([a-zA-Z0-9][a-zA-Z0-9\s'&-]{1,42}?)\s+(?:website|web\s*site|webpage|web\s*page|landing\s*page|dashboard|portfolio|storefront|store|shop|site|page)\b/gi
      )
    );
    for (let i = buildMatches.length - 1; i >= 0; i--) {
      const cand = cleanTitleCandidate(buildMatches[i][1]);
      if (cand) {
        title = cand;
        break;
      }
    }
  }
  if (!title && savedProject) {
    title = savedProject;
  }
  if (!title) {
    title =
      profile === "portfolio"
        ? savedName || "Alex Rivera"
        : profile === "dashboard"
          ? "Pulse Analytics Workspace"
          : profile === "ecommerce"
            ? "Atelier Supply Store"
            : profile === "restaurant"
              ? "Golden Coop Kitchen"
              : "Nexora Cloud Platform";
  }

  const ownerName = savedName || (profile === "portfolio" ? title : "Nexora Studio");

  const defaultAccent: AccentScheme =
    profile === "restaurant"
      ? /\b(coffee|cafe|bakery|espresso)\b/i.test(lower)
        ? "espresso"
        : "ember"
      : profile === "ecommerce"
        ? "emerald"
        : profile === "portfolio"
          ? "cobalt"
          : "cobalt";
  const accent = resolvePalette(lower, defaultAccent);

  // Gather connected facts from: (1) researchPassages, (2) history assistant/user context, (3) saved memoryFacts, (4) TFJS trained corpus
  const topicWords = extractTopicKeywordsSimple(normalized);
  const connectedFacts: string[] = [];

  if (context?.researchPassages && context.researchPassages.length > 0) {
    for (const p of context.researchPassages) {
      for (const s of extractSentencesFromText(p.text).slice(0, 4)) {
        connectedFacts.push(s);
      }
    }
  }

  // Pull real facts discussed earlier in the current thread or saved memories
  if (context?.history && context.history.length > 0) {
    for (const turn of [...context.history].reverse().slice(0, 8)) {
      const sents = extractSentencesFromText(turn.content);
      for (const s of sents) {
        const sl = s.toLowerCase();
        const matchesTopic =
          topicWords.length === 0 ||
          topicWords.some((w) => sl.includes(w)) ||
          /\b(facts?|what we discussed|from earlier|about this|based on)\b/i.test(lower);
        if (matchesTopic) {
          connectedFacts.push(s);
        }
      }
    }
  }

  for (const mem of memories) {
    connectedFacts.push(`${mem.key}: ${mem.value}`);
  }

  if (context?.threadDigests && context.threadDigests.length > 0 && topicWords.length > 0) {
    for (const d of context.threadDigests) {
      const hay = `${d.title} ${d.summary}`.toLowerCase();
      if (topicWords.some((w) => hay.includes(w)) && d.lastAssistantSnippet) {
        connectedFacts.push(`${d.title} — ${d.lastAssistantSnippet}`);
      }
    }
  }

  if (context?.corpusSentences && topicWords.length > 0 && connectedFacts.length < 4) {
    for (const cs of context.corpusSentences) {
      const csl = cs.toLowerCase();
      if (topicWords.some((w) => csl.includes(w))) {
        connectedFacts.push(cs);
      }
      if (connectedFacts.length >= 10) break;
    }
  }

  const dedupedFacts = Array.from(new Set(connectedFacts)).slice(0, 8);
  const extractedItems = extractListedPhrases(normalized);

  // Extract domain tag from prompt words
  const domain =
    extractedItems[0] ||
    (topicWords.slice(0, 3).join(" ") || title).replace(/\s+/g, " ").trim();

  const tagline =
    dedupedFacts.find((f) => f.length >= 36 && !f.includes(":")) ||
    (profile === "portfolio"
      ? `${savedRole || "Product Engineer & Interface Architect"} crafting resilient ${domain || "web"} software with measurable outcomes.`
      : profile === "dashboard"
        ? `Live operational telemetry, performance breakdowns, and searchable queue filtering for ${title}.`
        : profile === "ecommerce"
          ? `Curated ${domain || "signature essentials"} engineered with transparent specs, live search filtering, and instant cart calculation.`
          : `Purpose-built ${domain || "software"} workflow uniting fast execution, clear pricing, and verified real-world results for ${title}.`);

  // Parse any numbers mentioned in prompt/facts for metrics
  const numericMatches = Array.from(
    `${normalized} ${dedupedFacts.join(" ")}`.matchAll(
      /\b(\d+(?:\.\d+)?\s*(?:%|ms|s|x|k|M|\+|°[CF]|deg))\b/gi
    )
  ).map((m) => m[1]);

  const metrics = [
    {
      label: profile === "ecommerce" ? "Catalog Ready" : profile === "portfolio" ? "Shipped Work" : "Response Speed",
      value: numericMatches[0] || (profile === "ecommerce" ? "24h dispatch" : profile === "portfolio" ? "18+ launches" : "99.98%"),
      note: "Verified baseline",
    },
    {
      label: profile === "ecommerce" ? "Verified Rating" : profile === "portfolio" ? "Core Focus" : "Throughput",
      value: numericMatches[1] || (profile === "ecommerce" ? "4.9 / 5.0" : profile === "portfolio" ? "TypeScript · UI" : "42ms p95"),
      note: "Observed metric",
    },
    {
      label: profile === "ecommerce" ? "Guarantee" : profile === "portfolio" ? "Availability" : "Team Scale",
      value: numericMatches[2] || (profile === "ecommerce" ? "30-day return" : profile === "portfolio" ? "Open Q4" : "12.4k active"),
      note: "Live target",
    },
  ];

  return {
    profile,
    title,
    ownerName,
    domain,
    tagline,
    accent,
    preferDark,
    wantsPricingAddon,
    connectedFacts: dedupedFacts,
    extractedItems,
    metrics,
  };
}

function buildDesignFamilyHead(title: string, accentScheme: AccentScheme): string {
  const paletteMap: Record<
    AccentScheme,
    { l: string; li: string; lt: string; d: string; di: string; dt: string }
  > = {
    cobalt: {
      l: "#2f76ff",
      li: "#225ad1",
      lt: "#eef4ff",
      d: "#4c8dff",
      di: "#93bdff",
      dt: "rgba(76, 141, 255, 0.16)",
    },
    ember: {
      l: "#c2410c",
      li: "#9a3412",
      lt: "#fff3ec",
      d: "#f97316",
      di: "#fdba74",
      dt: "rgba(249, 115, 22, 0.16)",
    },
    espresso: {
      l: "#8b4513",
      li: "#6f3711",
      lt: "#fdf4ee",
      d: "#d99b66",
      di: "#f0c8a4",
      dt: "rgba(217, 155, 102, 0.16)",
    },
    emerald: {
      l: "#059669",
      li: "#047857",
      lt: "#ecfdf5",
      d: "#34d399",
      di: "#6ee7b7",
      dt: "rgba(52, 211, 153, 0.16)",
    },
    violet: {
      l: "#7c3aed",
      li: "#6d28d9",
      lt: "#f5f3ff",
      d: "#a78bfa",
      di: "#c4b5fd",
      dt: "rgba(167, 139, 250, 0.16)",
    },
  };
  const p = paletteMap[accentScheme];

  return `  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${escapeHtml(title)}</title>
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
      --ink: #1b1e24;
      --ink-2: #5b616e;
      --ink-3: #8a919e;
      --line: #eceef1;
      --line-strong: #dfe2e7;
      --accent: ${p.l};
      --accent-ink: ${p.li};
      --accent-tint: ${p.lt};
      --accent-on: #ffffff;
      --green: #16a34a;
      --green-tint: #ecfdf3;
      --red: #dc2626;
      --red-tint: #fef2f2;
      --font-display: "Plus Jakarta Sans", system-ui, -apple-system, sans-serif;
      --font-sans: "DM Sans", system-ui, -apple-system, sans-serif;
      --font-mono: "JetBrains Mono", ui-monospace, monospace;
      --radius-card: 14px;
      --radius-inner: 10px;
      --radius-ctrl: 8px;
      --space-section: clamp(32px, 5vw, 56px);
      --space-card: 22px;
      --ease: cubic-bezier(0.23, 1, 0.32, 1);
      --shadow-btn: 0 0 0 1px var(--line-strong), 0 1px 2px rgba(15, 23, 42, 0.04);
      --shadow-card: 0 0 0 1px var(--line), 0 1px 2px rgba(15, 23, 42, 0.04), 0 4px 12px rgba(15, 23, 42, 0.03);
      --shadow-raised: 0 0 0 1px var(--line-strong), 0 8px 24px rgba(15, 23, 42, 0.07), 0 2px 6px rgba(15, 23, 42, 0.04);
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
      --ink-2: #a5abb7;
      --ink-3: #6f7684;
      --line: rgba(255, 255, 255, 0.09);
      --line-strong: rgba(255, 255, 255, 0.15);
      --accent: ${p.d};
      --accent-ink: ${p.di};
      --accent-tint: ${p.dt};
      --accent-on: #ffffff;
      --green: #4ade80;
      --green-tint: rgba(74, 222, 128, 0.14);
      --red: #f87171;
      --red-tint: rgba(248, 113, 113, 0.14);
      --shadow-btn: 0 0 0 1px rgba(255, 255, 255, 0.11), 0 1px 2px rgba(0, 0, 0, 0.32);
      --shadow-card: 0 0 0 1px rgba(255, 255, 255, 0.09), 0 1px 3px rgba(0, 0, 0, 0.28), 0 6px 18px rgba(0, 0, 0, 0.2);
      --shadow-raised: 0 0 0 1px rgba(255, 255, 255, 0.13), 0 12px 30px rgba(0, 0, 0, 0.38);
      color-scheme: dark;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html {
      scroll-behavior: smooth;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }
    body {
      position: relative;
      min-height: 100vh;
      background-color: var(--page);
      background-image:
        radial-gradient(ellipse 85% 44% at 50% -6%, color-mix(in srgb, var(--accent) 10%, transparent), transparent 72%),
        radial-gradient(ellipse 55% 36% at 88% 12%, color-mix(in srgb, var(--accent) 5%, transparent), transparent 70%);
      color: var(--ink);
      font-family: var(--font-sans);
      font-size: 14.5px;
      line-height: 1.6;
      letter-spacing: -0.01em;
      transition: background-color 180ms var(--ease), color 180ms var(--ease);
    }
    /* Subtle architectural grid overlay that fades smoothly downward */
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: -1;
      background-image:
        linear-gradient(to right, var(--line) 1px, transparent 1px),
        linear-gradient(to bottom, var(--line) 1px, transparent 1px);
      background-size: 48px 48px;
      opacity: 0.45;
      -webkit-mask-image: radial-gradient(ellipse 90% 55% at 50% 0%, #000 10%, transparent 78%);
      mask-image: radial-gradient(ellipse 90% 55% at 50% 0%, #000 10%, transparent 78%);
    }
    h1, h2, h3, h4 {
      font-family: var(--font-display);
      letter-spacing: -0.028em;
      text-wrap: balance;
      color: var(--ink);
      overflow-wrap: anywhere;
    }
    p, li, td, th, span, strong {
      overflow-wrap: anywhere;
    }
    p { text-wrap: pretty; max-width: 65ch; }
    .num, .mono { font-variant-numeric: tabular-nums; }
    .mono { font-family: var(--font-mono); }
    button, input, select, textarea { font: inherit; color: inherit; }
    button:not(:disabled) { cursor: pointer; }
    :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
    .container { width: 100%; max-width: 1100px; margin: 0 auto; padding: 0 24px; }
    .topbar {
      position: sticky; top: 0; z-index: 40;
      height: 58px;
      background: color-mix(in srgb, var(--page) 84%, transparent);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--line);
    }
    .topbar-inner {
      height: 100%;
      display: flex; align-items: center; justify-content: space-between; gap: 16px;
    }
    .brand {
      display: inline-flex; align-items: center; gap: 9px;
      font-family: var(--font-display); font-weight: 600; font-size: 14.5px;
      letter-spacing: -0.02em; color: var(--ink); text-decoration: none; white-space: nowrap;
      min-width: 0;
    }
    .brand-mark {
      width: 26px; height: 26px; border-radius: 7px;
      background: linear-gradient(180deg, var(--surface), var(--inset));
      box-shadow: var(--shadow-btn);
      display: inline-flex; align-items: center; justify-content: center;
      color: var(--accent); flex-shrink: 0;
    }
    .nav-center { display: flex; align-items: center; gap: 6px; }
    .nav-link {
      padding: 6px 11px; border-radius: 7px; font-size: 13.5px; font-weight: 500;
      color: var(--ink-2); text-decoration: none; white-space: nowrap;
      transition: background 140ms var(--ease), color 140ms var(--ease);
    }
    .nav-link:hover { background: var(--hover); color: var(--ink); }
    @media (max-width: 700px) { .nav-center { display: none; } }
    .nav-actions { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
    .icon-btn {
      width: 36px; height: 36px; border-radius: var(--radius-ctrl); border: none;
      background: var(--surface); box-shadow: var(--shadow-btn); color: var(--ink-2);
      display: inline-flex; align-items: center; justify-content: center;
      transition: background 140ms var(--ease), color 140ms var(--ease), transform 100ms var(--ease);
    }
    .icon-btn:hover { background: var(--hover); color: var(--ink); }
    .icon-btn:active { transform: scale(0.96); }
    .btn {
      display: inline-flex; align-items: center; justify-content: center; gap: 8px;
      height: 40px; padding: 0 18px; border-radius: var(--radius-ctrl); border: none;
      font-size: 13.5px; font-weight: 500; text-decoration: none; white-space: nowrap;
      transition: background 140ms var(--ease), filter 140ms var(--ease), transform 100ms var(--ease);
    }
    .btn:active { transform: scale(0.985); }
    .btn-sm { height: 34px; padding: 0 13px; font-size: 13px; }
    .btn-primary {
      background: linear-gradient(180deg, var(--accent), color-mix(in srgb, var(--accent) 88%, #000));
      color: var(--accent-on);
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.14);
    }
    .btn-primary:hover { filter: brightness(1.07); }
    .btn-secondary { background: var(--surface); color: var(--ink); box-shadow: var(--shadow-btn); }
    .btn-secondary:hover { background: var(--hover); }
    .surface-card {
      background: var(--surface);
      border-radius: var(--radius-card);
      box-shadow: var(--shadow-card);
      min-width: 0;
    }
    .surface-inset {
      background: var(--inset);
      border: 1px solid var(--line);
      border-radius: var(--radius-inner);
      min-width: 0;
    }
    .meta-line {
      display: flex; flex-wrap: wrap; align-items: center; gap: 6px;
      font-size: 12.5px; color: var(--ink-3); font-weight: 500;
    }
    .section-block {
      padding: var(--space-section) 0 0;
    }
    .section-header {
      margin-bottom: 20px;
      display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 14px;
    }
    .section-header h2 { font-size: clamp(1.28rem, 2.3vw, 1.62rem); font-weight: 600; line-height: 1.22; }
    .section-header p { font-size: 14px; color: var(--ink-2); margin-top: 4px; max-width: 58ch; }
    .seg-tabs {
      display: inline-flex; flex-wrap: wrap; align-items: center; gap: 3px;
      background: var(--inset); padding: 3px; border-radius: 9px;
      border: 1px solid var(--line);
    }
    .seg-btn {
      border: none; background: transparent; color: var(--ink-2);
      padding: 6px 12px; border-radius: 6px; font-size: 12.5px; font-weight: 500;
      white-space: nowrap; transition: background 140ms var(--ease), color 140ms var(--ease), box-shadow 140ms var(--ease);
    }
    .seg-btn:hover { color: var(--ink); }
    .seg-btn.active { background: var(--surface); color: var(--ink); box-shadow: var(--shadow-btn); }
    .field-input {
      width: 100%; height: 40px; padding: 0 13px; border-radius: var(--radius-ctrl);
      background: var(--inset); border: 1px solid var(--line-strong); color: var(--ink);
      font-size: 13.5px; outline: none;
      transition: border-color 140ms var(--ease), background 140ms var(--ease), box-shadow 140ms var(--ease);
    }
    textarea.field-input { height: auto; min-height: 92px; padding: 10px 13px; resize: vertical; }
    .field-input::placeholder { color: var(--ink-3); }
    .field-input:focus {
      background: var(--surface); border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-tint);
    }
    footer.site-footer {
      border-top: 1px solid var(--line);
      padding: 24px 0 36px; margin-top: var(--space-section);
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px;
      font-size: 12.5px; color: var(--ink-3);
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { transition-duration: 0.01ms !important; animation-duration: 0.01ms !important; scroll-behavior: auto !important; }
    }`;
}

function buildThemeScript(defaultDark: boolean): string {
  return `<script>
    (function () {
      const root = document.documentElement;
      const btn = document.getElementById("themeBtn");
      ${defaultDark ? 'root.setAttribute("data-theme", "dark");' : ""}
      if (!btn) return;
      const sync = () => {
        const isDark = root.getAttribute("data-theme") === "dark";
        btn.setAttribute("aria-pressed", String(isDark));
        btn.title = isDark ? "Switch to light surface" : "Switch to obsidian dark surface";
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

// ---------------------------------------------------------------------------
// Profile 1: SaaS Landing Page (Hero + Features + Pricing + Testimonials + CTA)
// ---------------------------------------------------------------------------
function buildSaasLandingHtml(brief: WebsiteFactBrief): {
  title: string;
  summary: string;
  highlights: string[];
  html: string;
} {
  const safeBrand = escapeHtml(brief.title);
  const themeAttr = brief.preferDark ? ' data-theme="dark"' : "";

  const factPool = brief.connectedFacts;
  const items = brief.extractedItems;

  const features = [
    {
      idx: "01",
      title: items[0] ? items[0].replace(/^\w/, (c) => c.toUpperCase()) : `${brief.title} Core Engine`,
      desc:
        factPool[0] ||
        `Centralizes ${brief.domain || "team operations"} inside one structured workspace with deterministic checks and low-latency updates.`,
    },
    {
      idx: "02",
      title: items[1] ? items[1].replace(/^\w/, (c) => c.toUpperCase()) : "Connected Knowledge & Context",
      desc:
        factPool[1] ||
        `Carries relevant domain facts, configuration history, and operational metrics directly into every workflow without manual re-entry.`,
    },
    {
      idx: "03",
      title: items[2] ? items[2].replace(/^\w/, (c) => c.toUpperCase()) : "Predictable Governance & Speed",
      desc:
        factPool[2] ||
        `Built-in role controls, clean audit trails, and tabular cost visibility scale seamlessly from solo builders to multi-team deployments.`,
    },
  ];

  const testimonials = [
    {
      quote:
        factPool[3] ||
        `${brief.title} replaced three fragmented tools for our engineering group and cut weekly triage overhead noticeably.`,
      author: "Hannah Cho",
      role: `VP Engineering · Northstar`,
    },
    {
      quote:
        factPool[4] ||
        `Every detail stayed connected across sessions—our team moved from prototype validation to live production in days.`,
      author: "Marcus Bell",
      role: `Product Lead · Fieldwork`,
    },
    {
      quote:
        `Clear pricing, responsive performance across light and obsidian surfaces, and zero visual noise during daily work.`,
      author: brief.ownerName !== brief.title ? brief.ownerName : "Elena Ortiz",
      role: `Founder · ${brief.title} Partner`,
    },
  ];

  const html = `<!DOCTYPE html>
<html lang="en"${themeAttr}>
<head>
${buildDesignFamilyHead(brief.title, brief.accent)}
    .hero-grid {
      padding: clamp(36px, 6vw, 68px) 0 18px;
      display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 32px; align-items: center;
    }
    @media (max-width: 860px) { .hero-grid { grid-template-columns: 1fr; } }
    .hero-copy h1 {
      font-size: clamp(2.1rem, 4.3vw, 3.2rem); line-height: 1.07; margin: 10px 0 14px;
    }
    .hero-copy p { font-size: 16px; color: var(--ink-2); margin-bottom: 24px; max-width: 56ch; }
    .actions { display: flex; flex-wrap: wrap; gap: 10px; }
    .hero-panel {
      padding: 22px;
      background: linear-gradient(180deg, var(--surface), var(--inset));
      box-shadow: var(--shadow-raised);
    }
    .hero-panel-head {
      display: flex; justify-content: space-between; align-items: baseline;
      padding-bottom: 12px; border-bottom: 1px solid var(--line); margin-bottom: 14px;
    }
    .hero-panel-head h2 { font-size: 14px; font-weight: 600; }
    .metric-row {
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 14px;
    }
    @media (max-width: 500px) { .metric-row { grid-template-columns: 1fr; } }
    .metric-box { padding: 12px; }
    .metric-box span { display: block; font-size: 11.5px; color: var(--ink-3); }
    .metric-box strong { display: block; font-family: var(--font-mono); font-size: 16px; font-weight: 600; margin-top: 3px; color: var(--ink); }
    .fact-list { list-style: none; display: flex; flex-direction: column; gap: 9px; }
    .fact-list li {
      font-size: 13px; color: var(--ink-2); line-height: 1.5;
      padding-top: 9px; border-top: 1px solid var(--line);
    }
    .feat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
    .feat-card { padding: var(--space-card); display: flex; flex-direction: column; gap: 12px; }
    .feat-idx { font-family: var(--font-mono); font-size: 12px; color: var(--accent-ink); font-weight: 500; }
    .feat-card h3 { font-size: 16.5px; font-weight: 600; }
    .feat-card p { font-size: 13.5px; color: var(--ink-2); line-height: 1.62; }
    .pricing-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }
    .plan-card { padding: var(--space-card); display: flex; flex-direction: column; justify-content: space-between; gap: 18px; }
    .plan-card.featured {
      box-shadow: 0 0 0 1.5px var(--accent), var(--shadow-raised);
      background: linear-gradient(180deg, color-mix(in srgb, var(--accent-tint) 35%, var(--surface)), var(--surface));
    }
    .plan-price { font-family: var(--font-mono); font-size: 30px; font-weight: 600; color: var(--ink); margin: 8px 0 6px; }
    .plan-price span { font-family: var(--font-sans); font-size: 13px; font-weight: 400; color: var(--ink-2); }
    .plan-list { list-style: none; margin-top: 14px; border-top: 1px solid var(--line); }
    .plan-list li { padding: 8px 0; border-bottom: 1px solid var(--line); font-size: 13px; color: var(--ink-2); }
    .quote-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; }
    .quote-card { padding: var(--space-card); display: flex; flex-direction: column; justify-content: space-between; gap: 16px; }
    .quote-card blockquote { font-size: 14px; color: var(--ink); line-height: 1.65; }
    .cta-band {
      margin-top: var(--space-section);
      padding: clamp(24px, 4vw, 38px);
      display: grid; grid-template-columns: 1.2fr 1fr; gap: 24px; align-items: center;
      background: linear-gradient(135deg, color-mix(in srgb, var(--accent-tint) 55%, var(--surface)), var(--surface));
    }
    @media (max-width: 780px) { .cta-band { grid-template-columns: 1fr; } }
    .cta-band h2 { font-size: clamp(1.3rem, 2.5vw, 1.75rem); margin-bottom: 6px; }
    .cta-band p { font-size: 14px; color: var(--ink-2); }
    .cta-form { display: flex; flex-wrap: wrap; gap: 10px; }
    .cta-form .field-input { flex: 1 1 200px; }
    .cta-status { margin-top: 8px; font-size: 12.5px; color: var(--green); font-weight: 500; min-height: 18px; }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="container topbar-inner">
      <a class="brand" href="#">
        <span class="brand-mark">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 19V5l14 14V5"/></svg>
        </span>
        <span>${safeBrand}</span>
      </a>
      <nav class="nav-center" aria-label="Main navigation">
        <a class="nav-link" href="#features">Features</a>
        <a class="nav-link" href="#pricing">Pricing</a>
        <a class="nav-link" href="#testimonials">Testimonials</a>
      </nav>
      <div class="nav-actions">
        <button type="button" class="icon-btn" id="themeBtn" aria-label="Toggle theme">${SUN_MOON_ICON_SVG}</button>
        <a class="btn btn-sm btn-primary" href="#cta">Get Started</a>
      </div>
    </div>
  </header>

  <main class="container">
    <section class="hero-grid">
      <div class="hero-copy">
        <div class="meta-line">
          <span>${safeBrand}</span>
          <span aria-hidden="true">&middot;</span>
          <span>${escapeHtml(brief.domain || "Software Platform")}</span>
        </div>
        <h1>Build and ship ${escapeHtml(brief.domain || brief.title)} with connected clarity.</h1>
        <p>${escapeHtml(brief.tagline)}</p>
        <div class="actions">
          <a class="btn btn-primary" href="#pricing">Compare Plans</a>
          <a class="btn btn-secondary" href="#features">Explore Capabilities</a>
        </div>
      </div>

      <aside class="surface-card hero-panel" aria-label="Connected Overview">
        <div class="hero-panel-head">
          <h2>Connected Snapshot</h2>
          <span class="mono" style="font-size:11.5px;color:var(--ink-3);">${escapeHtml(brief.ownerName)}</span>
        </div>
        <div class="metric-row">
          ${brief.metrics
            .map(
              (m) =>
                `<div class="surface-inset metric-box"><span>${escapeHtml(m.label)}</span><strong>${escapeHtml(m.value)}</strong></div>`
            )
            .join("")}
        </div>
        <ul class="fact-list">
          ${(factPool.length > 0 ? factPool.slice(0, 2) : [
            `Unified architecture tailored around ${brief.title} goals.`,
            `Clean light and obsidian theme surfaces with accessible controls.`,
          ])
            .map((f) => `<li>${escapeHtml(f)}</li>`)
            .join("")}
        </ul>
      </aside>
    </section>

    <section id="features" class="section-block">
      <div class="section-header">
        <div>
          <h2>Core Capabilities</h2>
          <p>Structured workflows designed to keep every specification, metric, and user story aligned.</p>
        </div>
      </div>
      <div class="feat-grid">
        ${features
          .map(
            (f) => `<article class="surface-card feat-card">
          <span class="feat-idx">${f.idx}</span>
          <h3>${escapeHtml(f.title)}</h3>
          <p>${escapeHtml(f.desc)}</p>
        </article>`
          )
          .join("")}
      </div>
    </section>

    <section id="pricing" class="section-block">
      <div class="section-header">
        <div>
          <h2>Transparent Pricing</h2>
          <p>Switch billing cadence anytime without hidden platform surcharges.</p>
        </div>
        <div class="seg-tabs" role="tablist" aria-label="Billing interval">
          <button type="button" class="seg-btn active" data-billing="monthly">Monthly</button>
          <button type="button" class="seg-btn" data-billing="annual">Annual (Save 20%)</button>
        </div>
      </div>
      <div class="pricing-grid">
        <article class="surface-card plan-card">
          <div>
            <div class="meta-line"><span>Starter</span><span>&middot;</span><span>Solo Builders</span></div>
            <div class="plan-price"><span data-price="0">$0</span><span> / mo</span></div>
            <p style="font-size:13.5px;color:var(--ink-2);">Full core workspace to prototype and verify ${safeBrand} workflows.</p>
            <ul class="plan-list">
              <li>Up to 3 active projects</li>
              <li>Light &amp; Obsidian UI tokens</li>
              <li>Community export formats</li>
            </ul>
          </div>
          <a class="btn btn-secondary" href="#cta">Start Free</a>
        </article>
        <article class="surface-card plan-card featured">
          <div>
            <div class="meta-line" style="color:var(--accent-ink);"><span>Growth</span><span>&middot;</span><span>Recommended</span></div>
            <div class="plan-price"><span data-price="24">$24</span><span> / seat / mo</span></div>
            <p style="font-size:13.5px;color:var(--ink-2);">Connected team collaboration, custom branding, and priority throughput.</p>
            <ul class="plan-list">
              <li>Unlimited shared workspaces</li>
              <li>Custom domain &amp; palette tokens</li>
              <li>Priority SLA &amp; audit log history</li>
            </ul>
          </div>
          <a class="btn btn-primary" href="#cta">Choose Growth</a>
        </article>
        <article class="surface-card plan-card">
          <div>
            <div class="meta-line"><span>Scale</span><span>&middot;</span><span>Organizations</span></div>
            <div class="plan-price"><span data-price="79">$79</span><span> / team / mo</span></div>
            <p style="font-size:13.5px;color:var(--ink-2);">Dedicated capacity controls, SAML SSO, and tailored onboarding.</p>
            <ul class="plan-list">
              <li>SAML SSO &amp; fine-grained RBAC</li>
              <li>Dedicated security review</li>
              <li>99.98% uptime guarantee</li>
            </ul>
          </div>
          <a class="btn btn-secondary" href="#cta">Contact Sales</a>
        </article>
      </div>
    </section>

    <section id="testimonials" class="section-block">
      <div class="section-header">
        <div>
          <h2>Teams Shipping With ${safeBrand}</h2>
          <p>Grounded feedback from engineers and product leads using structured workflows daily.</p>
        </div>
      </div>
      <div class="quote-grid">
        ${testimonials
          .map(
            (t) => `<article class="surface-card quote-card">
          <blockquote>"${escapeHtml(t.quote)}"</blockquote>
          <div class="meta-line">
            <strong style="color:var(--ink);font-weight:600;">${escapeHtml(t.author)}</strong>
            <span>&middot;</span>
            <span>${escapeHtml(t.role)}</span>
          </div>
        </article>`
          )
          .join("")}
      </div>
    </section>

    <section id="cta" class="surface-card cta-band">
      <div>
        <h2>Ready to launch with ${safeBrand}?</h2>
        <p>Request workspace access and connect your first project in under two minutes.</p>
      </div>
      <div>
        <form id="ctaForm" class="cta-form">
          <input id="ctaEmail" class="field-input" type="email" placeholder="you@company.com" required />
          <button class="btn btn-primary" type="submit">Request Access</button>
        </form>
        <div id="ctaStatus" class="cta-status" role="status"></div>
      </div>
    </section>

    <footer class="site-footer">
      <span>&copy; ${new Date().getFullYear()} ${safeBrand}</span>
      <span>Built for clarity &middot; Light &amp; Obsidian Surfaces</span>
    </footer>
  </main>

  ${buildThemeScript(brief.preferDark)}
  <script>
    document.querySelectorAll("[data-billing]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-billing]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const annual = btn.dataset.billing === "annual";
        document.querySelectorAll("[data-price]").forEach((el) => {
          const base = Number(el.dataset.price || 0);
          const next = annual ? Math.round(base * 0.8) : base;
          el.textContent = "$" + next;
        });
      });
    });
    document.getElementById("ctaForm").addEventListener("submit", (e) => {
      e.preventDefault();
      const email = document.getElementById("ctaEmail").value.trim();
      document.getElementById("ctaStatus").textContent = "Workspace invite queued for " + email + ".";
      e.target.reset();
    });
  </script>
</body>
</html>`;

  return {
    title: brief.title,
    summary: `Built a **SaaS Landing Page** for **${brief.title}** following the **SaaS Profile** (Split Hero + Connected Snapshot, Features, Monthly/Annual Interactive Pricing, Testimonials, and CTA) in the Nexora design family.`,
    highlights: [
      "**SaaS Profile Architecture:** Asymmetric split hero paired with a live connected metrics panel, 3-column feature cards, interactive `Monthly / Annual` billing switcher, testimonials, and access CTA.",
      "**Fact-Connected Copy:** Woven your prompt topic, domain items, and remembered workspace facts directly into the hero snapshot, feature cards, and quotes.",
      "**Nexora Design Family:** Subtle ambient radial gradient + architectural hairline grid backdrop, crisp display-to-body scale, and long-text wrapping protection.",
    ],
    html,
  };
}

// ---------------------------------------------------------------------------
// Profile 2: Dashboard (Sidebar + Charts + Statistics + Tables + Filters)
// ---------------------------------------------------------------------------
function buildDashboardHtml(brief: WebsiteFactBrief): {
  title: string;
  summary: string;
  highlights: string[];
  html: string;
} {
  const safeTitle = escapeHtml(brief.title);
  const themeAttr = brief.preferDark ? ' data-theme="dark"' : "";
  const items = brief.extractedItems;
  const facts = brief.connectedFacts;

  const seedRows = [
    {
      id: "OP-401",
      name: items[0] ? items[0] : facts[0]?.slice(0, 68) || `${brief.title} Primary Pipeline`,
      cat: "Core",
      metric: brief.metrics[0]?.value || "99.9%",
      status: "Active",
    },
    {
      id: "OP-402",
      name: items[1] ? items[1] : facts[1]?.slice(0, 68) || `${brief.domain || "Workspace"} Verification Run`,
      cat: "Analytics",
      metric: brief.metrics[1]?.value || "42ms",
      status: "Completed",
    },
    {
      id: "OP-403",
      name: items[2] ? items[2] : facts[2]?.slice(0, 68) || "Customer Cohort Indexing",
      cat: "Sync",
      metric: "14.2k",
      status: "Active",
    },
    {
      id: "OP-404",
      name: items[3] ? items[3] : "Automated Weekly Snapshot Export",
      cat: "Reports",
      metric: "100%",
      status: "Queued",
    },
  ];

  const html = `<!DOCTYPE html>
<html lang="en"${themeAttr}>
<head>
${buildDesignFamilyHead(`${brief.title} — Dashboard`, brief.accent)}
    .shell {
      min-height: 100vh;
      display: grid; grid-template-columns: 236px 1fr;
    }
    @media (max-width: 860px) { .shell { grid-template-columns: 1fr; } }
    .sidebar {
      background: color-mix(in srgb, var(--surface) 92%, var(--canvas));
      border-right: 1px solid var(--line);
      padding: 18px 14px;
      display: flex; flex-direction: column; justify-content: space-between; gap: 20px;
      position: sticky; top: 0; height: 100vh;
    }
    @media (max-width: 860px) {
      .sidebar { position: static; height: auto; border-right: none; border-bottom: 1px solid var(--line); }
    }
    .side-nav { display: flex; flex-direction: column; gap: 4px; margin-top: 18px; }
    .side-link {
      display: flex; align-items: center; justify-content: space-between;
      padding: 8px 11px; border-radius: var(--radius-ctrl); border: none;
      background: transparent; color: var(--ink-2); font-size: 13.5px; font-weight: 500;
      text-decoration: none; text-align: left;
      transition: background 130ms var(--ease), color 130ms var(--ease);
    }
    .side-link:hover { background: var(--hover); color: var(--ink); }
    .side-link.active { background: var(--surface); color: var(--ink); box-shadow: var(--shadow-btn); }
    .side-note {
      padding: 12px; font-size: 12.5px; color: var(--ink-2); line-height: 1.5;
    }
    .workspace-col { min-width: 0; display: flex; flex-direction: column; }
    .dash-bar {
      position: sticky; top: 0; z-index: 30;
      height: 58px; padding: 0 24px;
      background: color-mix(in srgb, var(--page) 86%, transparent);
      backdrop-filter: blur(12px); border-bottom: 1px solid var(--line);
      display: flex; align-items: center; justify-content: space-between; gap: 12px;
    }
    .dash-content { padding: 24px; max-width: 1180px; width: 100%; display: flex; flex-direction: column; gap: 18px; }
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(195px, 1fr)); gap: 14px; }
    .kpi { padding: 18px; }
    .kpi-label { font-size: 12.5px; color: var(--ink-2); font-weight: 500; }
    .kpi-val { font-family: var(--font-display); font-size: 26px; font-weight: 700; margin: 6px 0 2px; font-variant-numeric: tabular-nums; }
    .kpi-delta { font-family: var(--font-mono); font-size: 12px; color: var(--green); }
    .charts-row { display: grid; grid-template-columns: 1.45fr 1fr; gap: 16px; }
    @media (max-width: 960px) { .charts-row { grid-template-columns: 1fr; } }
    .chart-card { padding: 20px; }
    .chart-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 14px; }
    .chart-head h2 { font-size: 15.5px; font-weight: 600; }
    .bar-row { margin-bottom: 12px; }
    .bar-row:last-child { margin-bottom: 0; }
    .bar-meta { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 5px; }
    .bar-track { height: 8px; background: var(--inset); border-radius: 999px; overflow: hidden; border: 1px solid var(--line); }
    .bar-fill { height: 100%; background: var(--accent); border-radius: 999px; transition: width 260ms var(--ease); }
    .table-card { padding: 18px 20px; }
    .table-toolbar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 14px; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
    th {
      text-align: left; font-size: 11.5px; font-weight: 600; color: var(--ink-3);
      padding: 10px 8px; border-bottom: 1px solid var(--line);
    }
    td { padding: 11px 8px; border-bottom: 1px solid var(--line); vertical-align: middle; }
    tbody tr:hover { background: var(--hover); }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar" aria-label="Workspace navigation">
      <div>
        <a class="brand" href="#">
          <span class="brand-mark">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/></svg>
          </span>
          <span>${safeTitle}</span>
        </a>
        <nav class="side-nav" id="sideNav">
          <a class="side-link active" href="#stats"><span>Overview</span><span class="mono" style="font-size:11px;">01</span></a>
          <a class="side-link" href="#analytics"><span>Trend Charts</span><span class="mono" style="font-size:11px;">02</span></a>
          <a class="side-link" href="#records"><span>Records Table</span><span class="mono" style="font-size:11px;">03</span></a>
        </nav>
      </div>
      <div class="surface-inset side-note">
        <div style="font-weight:600;color:var(--ink);margin-bottom:4px;">Workspace Context</div>
        <div>${escapeHtml(brief.tagline.slice(0, 120))}</div>
      </div>
    </aside>

    <div class="workspace-col">
      <header class="dash-bar">
        <div class="meta-line">
          <strong style="color:var(--ink);font-family:var(--font-display);font-size:15px;">${safeTitle}</strong>
          <span>&middot;</span>
          <span>${escapeHtml(brief.ownerName)}</span>
        </div>
        <div class="nav-actions">
          <div class="seg-tabs" role="tablist" aria-label="Timeframe">
            <button type="button" class="seg-btn active" data-range="7d">7D</button>
            <button type="button" class="seg-btn" data-range="30d">30D</button>
            <button type="button" class="seg-btn" data-range="90d">90D</button>
          </div>
          <button type="button" class="icon-btn" id="themeBtn" aria-label="Toggle theme">${SUN_MOON_ICON_SVG}</button>
        </div>
      </header>

      <main class="dash-content">
        <section id="stats" class="kpi-grid">
          <article class="surface-card kpi">
            <div class="kpi-label">Active Throughput</div>
            <div class="kpi-val" id="kThroughput">14,820</div>
            <div class="kpi-delta" id="dThroughput">+16.4% vs prior</div>
          </article>
          <article class="surface-card kpi">
            <div class="kpi-label">Completion Rate</div>
            <div class="kpi-val" id="kRate">96.4%</div>
            <div class="kpi-delta" id="dRate">+1.8% improvement</div>
          </article>
          <article class="surface-card kpi">
            <div class="kpi-label">Median Latency</div>
            <div class="kpi-val" id="kLatency">38 ms</div>
            <div class="kpi-delta" id="dLatency">-9 ms faster</div>
          </article>
          <article class="surface-card kpi">
            <div class="kpi-label">Tracked Domain</div>
            <div class="kpi-val" style="font-size:18px;padding-top:6px;">${escapeHtml(brief.domain || brief.title)}</div>
            <div class="kpi-delta" style="color:var(--ink-3);">Connected metric</div>
          </article>
        </section>

        <section id="analytics" class="charts-row">
          <article class="surface-card chart-card">
            <div class="chart-head">
              <h2>Performance Trajectory</h2>
              <span id="chartRangeLabel" class="mono" style="font-size:12px;color:var(--ink-3);">Last 7 days</span>
            </div>
            <svg id="trendSvg" viewBox="0 0 640 155" width="100%" height="155" role="img" aria-label="Performance trend line"></svg>
          </article>

          <article class="surface-card chart-card">
            <div class="chart-head">
              <h2>Channel Breakdown</h2>
              <span class="mono" style="font-size:12px;color:var(--ink-3);">Share</span>
            </div>
            <div id="channelBars"></div>
          </article>
        </section>

        <section id="records" class="surface-card table-card">
          <div class="table-toolbar">
            <div style="display:flex;flex-wrap:wrap;align-items:center;gap:12px;">
              <h2 style="font-size:15.5px;font-weight:600;">Operations Log</h2>
              <div class="seg-tabs" role="tablist" aria-label="Status filter">
                <button type="button" class="seg-btn active" data-status="all">All</button>
                <button type="button" class="seg-btn" data-status="Active">Active</button>
                <button type="button" class="seg-btn" data-status="Completed">Completed</button>
              </div>
            </div>
            <input id="rowSearch" class="field-input" style="max-width:240px;height:36px;" type="search" placeholder="Search records..." />
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>ID</th><th>Workstream / Fact</th><th>Category</th><th>Metric</th><th style="text-align:right;">State</th></tr>
              </thead>
              <tbody id="recordsBody"></tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  </div>

  ${buildThemeScript(brief.preferDark)}
  <script>
    const SERIES = {
      "7d": { tp: "14,820", dTp: "+16.4% vs prior", rate: "96.4%", dRate: "+1.8% improvement", lat: "38 ms", dLat: "-9 ms faster", cap: "Last 7 days", pts: [44, 52, 49, 63, 71, 78, 90], bars: [["Core Flow", 78], ["Sync Jobs", 56], ["Exports", 38]] },
      "30d": { tp: "58,410", dTp: "+24.1% vs prior", rate: "97.1%", dRate: "+2.4% improvement", lat: "34 ms", dLat: "-12 ms faster", cap: "Last 30 days", pts: [31, 40, 48, 55, 68, 82, 93], bars: [["Core Flow", 84], ["Sync Jobs", 64], ["Exports", 46]] },
      "90d": { tp: "169,200", dTp: "+39.0% vs prior", rate: "97.8%", dRate: "+3.1% improvement", lat: "31 ms", dLat: "-16 ms faster", cap: "Last 90 days", pts: [22, 33, 45, 58, 72, 85, 97], bars: [["Core Flow", 89], ["Sync Jobs", 71], ["Exports", 52]] }
    };
    const ROWS = ${JSON.stringify(seedRows)};
    let activeStatus = "all";
    let searchTerm = "";

    function drawChart(pts) {
      const svg = document.getElementById("trendSvg");
      const W = 640, H = 155, px = 14, py = 16;
      const max = Math.max(...pts, 100);
      const step = (W - px * 2) / (pts.length - 1);
      const coords = pts.map((v, i) => [px + i * step, H - py - (v / max) * (H - py * 2)]);
      const line = coords.map((p, i) => (i === 0 ? "M" : "L") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ");
      const area = line + " L" + coords[coords.length - 1][0].toFixed(1) + "," + (H - py) + " L" + coords[0][0].toFixed(1) + "," + (H - py) + " Z";
      svg.innerHTML =
        '<line x1="0" y1="' + (H - py) + '" x2="' + W + '" y2="' + (H - py) + '" stroke="var(--line)" stroke-width="1"/>' +
        '<path d="' + area + '" fill="var(--accent-tint)"/>' +
        '<path d="' + line + '" fill="none" stroke="var(--accent)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>' +
        coords.map((p) => '<circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="3.5" fill="var(--surface)" stroke="var(--accent)" stroke-width="2"/>').join("");
    }

    function renderBars(bars) {
      document.getElementById("channelBars").innerHTML = bars.map((b) =>
        '<div class="bar-row"><div class="bar-meta"><span>' + b[0] + '</span><span class="mono">' + b[1] + '%</span></div><div class="bar-track"><div class="bar-fill" style="width:' + b[1] + '%"></div></div></div>'
      ).join("");
    }

    function renderRows() {
      const body = document.getElementById("recordsBody");
      const filtered = ROWS.filter((r) => {
        const matchStatus = activeStatus === "all" || r.status === activeStatus;
        const matchText = !searchTerm || (r.id + " " + r.name + " " + r.cat).toLowerCase().includes(searchTerm);
        return matchStatus && matchText;
      });
      body.innerHTML = filtered.map((r) =>
        '<tr>' +
          '<td class="mono" style="color:var(--ink-3);font-size:12px;">' + r.id + '</td>' +
          '<td style="font-weight:500;">' + r.name + '</td>' +
          '<td style="color:var(--ink-2);">' + r.cat + '</td>' +
          '<td class="mono">' + r.metric + '</td>' +
          '<td style="text-align:right;font-size:12.5px;color:' + (r.status === "Completed" ? "var(--green)" : "var(--ink-2)") + ';">' + r.status + '</td>' +
        '</tr>'
      ).join("");
    }

    document.querySelectorAll("[data-range]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-range]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const s = SERIES[btn.dataset.range];
        document.getElementById("kThroughput").textContent = s.tp;
        document.getElementById("dThroughput").textContent = s.dTp;
        document.getElementById("kRate").textContent = s.rate;
        document.getElementById("dRate").textContent = s.dRate;
        document.getElementById("kLatency").textContent = s.lat;
        document.getElementById("dLatency").textContent = s.dLat;
        document.getElementById("chartRangeLabel").textContent = s.cap;
        drawChart(s.pts);
        renderBars(s.bars);
      });
    });

    document.querySelectorAll("[data-status]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-status]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        activeStatus = btn.dataset.status;
        renderRows();
      });
    });

    document.getElementById("rowSearch").addEventListener("input", (e) => {
      searchTerm = e.target.value.trim().toLowerCase();
      renderRows();
    });

    document.querySelectorAll(".side-link").forEach((lnk) => {
      lnk.addEventListener("click", () => {
        document.querySelectorAll(".side-link").forEach((x) => x.classList.remove("active"));
        lnk.classList.add("active");
      });
    });

    drawChart(SERIES["7d"].pts);
    renderBars(SERIES["7d"].bars);
    renderRows();
  </script>
</body>
</html>`;

  return {
    title: `${brief.title} Dashboard`,
    summary: `Built an interactive **Dashboard** for **${brief.title}** following the **Dashboard Profile** (Sidebar Navigation, KPI Statistics, Timeframe Charts, Searchable Table, and Status Filters) in the Nexora design family.`,
    highlights: [
      "**Dashboard Profile Architecture:** Persistent left navigation sidebar, 4-metric KPI grid (`tabular-nums`), interactive `7D / 30D / 90D` SVG curve + distribution bars, and searchable operations table.",
      "**Connected Fact Rows:** Seeded workstream entries and workspace context directly from your prompt and saved conversation memory.",
      "**Unified Nexora Surfaces:** Consistent hairline borders, subtle top radial light wash, and responsive table scrolling.",
    ],
    html,
  };
}

// ---------------------------------------------------------------------------
// Profile 3: E-commerce (Product Cards + Categories + Search + Cart)
// ---------------------------------------------------------------------------
function buildEcommerceHtml(brief: WebsiteFactBrief): {
  title: string;
  summary: string;
  highlights: string[];
  html: string;
} {
  const safeStore = escapeHtml(brief.title);
  const themeAttr = brief.preferDark ? ' data-theme="dark"' : "";
  const items = brief.extractedItems;
  const facts = brief.connectedFacts;

  const products =
    items.length >= 2
      ? items.slice(0, 6).map((rawName, idx) => {
          const cleanName = rawName.replace(/^\w/, (c) => c.toUpperCase());
          const cat = idx % 3 === 0 ? "Featured" : idx % 3 === 1 ? "Essentials" : "Bundles";
          const price = Number((24 + idx * 14.5).toFixed(2));
          return {
            id: idx + 1,
            name: cleanName,
            cat,
            price,
            spec: idx % 2 === 0 ? "In stock · Ships in 24h" : "Small-batch edition",
            desc:
              facts[idx] ||
              `Crafted for ${brief.domain || brief.title} enthusiasts with verified materials and clean everyday utility.`,
          };
        })
      : [
          {
            id: 1,
            name: `${brief.title} Flagship Edition`,
            cat: "Featured",
            price: 68,
            spec: "In stock · 2-year warranty",
            desc: facts[0] || `Our primary ${brief.domain || "signature"} release built with precision hardware and durable finishes.`,
          },
          {
            id: 2,
            name: "Studio Companion Kit",
            cat: "Bundles",
            price: 94,
            spec: "Best value · Ready to ship",
            desc: facts[1] || `Complete paired bundle tailored around daily ${brief.domain || "workspace"} workflows.`,
          },
          {
            id: 3,
            name: "Precision Core Module",
            cat: "Essentials",
            price: 39,
            spec: "Modular fit",
            desc: facts[2] || "Clean drop-in component engineered for low friction and dependable long-term wear.",
          },
          {
            id: 4,
            name: "Matte Obsidian Desk Edition",
            cat: "Featured",
            price: 54,
            spec: "Limited run",
            desc: "Anodized neutral finish matching modern studio setups with soft tactile dampening.",
          },
          {
            id: 5,
            name: "Starter Accessory Pack",
            cat: "Essentials",
            price: 22,
            spec: "Same-day dispatch",
            desc: "Everyday replacement parts, travel sleeve, and maintenance cloth in one tidy pouch.",
          },
          {
            id: 6,
            name: "Team Multi-Pack (3x)",
            cat: "Bundles",
            price: 149,
            spec: "Save 18% on bundle",
            desc: "Three matched units boxed cleanly for studios, gifts, or multi-desk setups.",
          },
        ];

  const categories = ["All", ...Array.from(new Set(products.map((p) => p.cat)))];

  const html = `<!DOCTYPE html>
<html lang="en"${themeAttr}>
<head>
${buildDesignFamilyHead(brief.title, brief.accent)}
    .hero-shop {
      padding: clamp(32px, 5vw, 56px) 0 24px;
      display: grid; grid-template-columns: 1.25fr 0.75fr; gap: 24px; align-items: stretch;
    }
    @media (max-width: 820px) { .hero-shop { grid-template-columns: 1fr; } }
    .hero-shop h1 { font-size: clamp(2rem, 4vw, 2.95rem); line-height: 1.08; margin: 10px 0 12px; }
    .hero-shop p { font-size: 15.5px; color: var(--ink-2); margin-bottom: 20px; }
    .spotlight { padding: 20px; display: flex; flex-direction: column; justify-content: space-between; gap: 14px; }
    .store-controls {
      display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px;
      padding: 18px 0 16px; border-top: 1px solid var(--line);
    }
    .store-layout {
      display: grid; grid-template-columns: 1.65fr 1fr; gap: 20px; align-items: start;
    }
    @media (max-width: 880px) { .store-layout { grid-template-columns: 1fr; } }
    .product-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; }
    .product-card {
      padding: 18px; display: flex; flex-direction: column; justify-content: space-between; gap: 14px;
      transition: transform 150ms var(--ease), box-shadow 150ms var(--ease);
    }
    .product-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-raised); }
    .p-head { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; margin-top: 6px; }
    .p-head h3 { font-size: 15.5px; font-weight: 600; }
    .price { font-family: var(--font-mono); font-size: 14px; font-weight: 600; color: var(--accent-ink); white-space: nowrap; }
    .p-desc { font-size: 13px; color: var(--ink-2); margin-top: 6px; line-height: 1.55; }
    .cart-card { padding: 20px; position: sticky; top: 76px; }
    .cart-head { display: flex; justify-content: space-between; align-items: center; padding-bottom: 12px; border-bottom: 1px solid var(--line); margin-bottom: 12px; }
    .cart-items { display: flex; flex-direction: column; gap: 10px; margin-bottom: 14px; }
    .cart-row { display: flex; justify-content: space-between; align-items: center; gap: 10px; padding-bottom: 10px; border-bottom: 1px solid var(--line); font-size: 13px; }
    .qty-ctrl { display: inline-flex; align-items: center; gap: 6px; }
    .qty-btn { width: 26px; height: 26px; border-radius: 6px; border: none; background: var(--inset); box-shadow: var(--shadow-btn); font-family: var(--font-mono); }
    .sum-line { display: flex; justify-content: space-between; font-size: 13.5px; color: var(--ink-2); margin-bottom: 6px; }
    .sum-total { display: flex; justify-content: space-between; font-size: 15px; font-weight: 600; padding-top: 8px; border-top: 1px solid var(--line); margin-bottom: 14px; }
    .status-note { margin-top: 10px; font-size: 12.5px; color: var(--green); font-weight: 500; min-height: 18px; }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="container topbar-inner">
      <a class="brand" href="#">
        <span class="brand-mark">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 2 3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4Z"/><path d="M3 6h18"/><path d="M16 10a4 4 0 0 1-8 0"/></svg>
        </span>
        <span>${safeStore}</span>
      </a>
      <nav class="nav-center" aria-label="Store navigation">
        <a class="nav-link" href="#catalog">Catalog</a>
        <a class="nav-link" href="#cartPanel">Cart</a>
      </nav>
      <div class="nav-actions">
        <button type="button" class="icon-btn" id="themeBtn" aria-label="Toggle theme">${SUN_MOON_ICON_SVG}</button>
        <a class="btn btn-sm btn-primary" href="#cartPanel">Bag (<span id="navCartCount" class="num">0</span>)</a>
      </div>
    </div>
  </header>

  <main class="container">
    <section class="hero-shop">
      <div>
        <div class="meta-line">
          <span>${safeStore}</span>
          <span>&middot;</span>
          <span>${escapeHtml(brief.domain || "Curated Goods")}</span>
        </div>
        <h1>Thoughtful ${escapeHtml(brief.domain || "goods")} built to last daily.</h1>
        <p>${escapeHtml(brief.tagline)}</p>
        <a class="btn btn-primary" href="#catalog">Browse Collection</a>
      </div>
      <aside class="surface-card spotlight">
        <div>
          <div class="meta-line"><span>Featured Release</span><span>&middot;</span><span>${escapeHtml(brief.ownerName)}</span></div>
          <h2 style="font-size:17px;margin:8px 0 6px;">${escapeHtml(products[0].name)}</h2>
          <p style="font-size:13.5px;color:var(--ink-2);">${escapeHtml(products[0].desc)}</p>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;padding-top:12px;border-top:1px solid var(--line);">
          <span class="mono" style="font-weight:600;color:var(--accent-ink);">$${products[0].price.toFixed(2)}</span>
          <button type="button" class="btn btn-sm btn-secondary" id="quickAddHero">Quick Add</button>
        </div>
      </aside>
    </section>

    <section id="catalog">
      <div class="store-controls">
        <div class="seg-tabs" id="catTabs" role="tablist" aria-label="Product categories"></div>
        <input id="productSearch" class="field-input" style="max-width:260px;height:36px;" type="search" placeholder="Search products..." />
      </div>

      <div class="store-layout">
        <div class="product-grid" id="productGrid"></div>

        <aside id="cartPanel" class="surface-card cart-card" aria-label="Shopping cart">
          <div class="cart-head">
            <h2 style="font-size:16px;font-weight:600;">Your Cart</h2>
            <span class="mono" id="cartItemLabel" style="font-size:12px;color:var(--ink-3);">0 items</span>
          </div>
          <div class="cart-items" id="cartItems">
            <p style="font-size:13px;color:var(--ink-2);">Add items from the catalog to calculate totals.</p>
          </div>
          <div class="sum-line"><span>Subtotal</span><span class="mono" id="subTotal">$0.00</span></div>
          <div class="sum-line"><span>Shipping</span><span class="mono" id="shippingFee">$0.00</span></div>
          <div class="sum-total"><span>Total</span><span class="mono" id="grandTotal">$0.00</span></div>
          <button type="button" class="btn btn-primary" style="width:100%;" id="checkoutBtn">Complete Order</button>
          <div id="checkoutNote" class="status-note" role="status"></div>
        </aside>
      </div>
    </section>

    <footer class="site-footer">
      <span>&copy; ${new Date().getFullYear()} ${safeStore}</span>
      <span>Transparent specs &middot; Express fulfillment</span>
    </footer>
  </main>

  ${buildThemeScript(brief.preferDark)}
  <script>
    const PRODUCTS = ${JSON.stringify(products)};
    const CATS = ${JSON.stringify(categories)};
    let activeCat = "All";
    let searchQ = "";
    const cart = new Map();

    function renderCats() {
      document.getElementById("catTabs").innerHTML = CATS.map((c) =>
        '<button type="button" class="seg-btn ' + (c === activeCat ? "active" : "") + '" data-cat="' + c + '">' + c + '</button>'
      ).join("");
    }

    function renderGrid() {
      const list = PRODUCTS.filter((p) => {
        const matchCat = activeCat === "All" || p.cat === activeCat;
        const matchQ = !searchQ || (p.name + " " + p.desc + " " + p.cat).toLowerCase().includes(searchQ);
        return matchCat && matchQ;
      });
      const grid = document.getElementById("productGrid");
      if (list.length === 0) {
        grid.innerHTML = '<div class="surface-card" style="padding:24px;color:var(--ink-2);">No products matched "' + searchQ.replace(/</g, "&lt;") + '".</div>';
        return;
      }
      grid.innerHTML = list.map((p) => {
        const qty = cart.get(p.id) || 0;
        return '<article class="surface-card product-card">' +
          '<div>' +
            '<div class="meta-line"><span>' + p.cat + '</span><span>&middot;</span><span>' + p.spec + '</span></div>' +
            '<div class="p-head"><h3>' + p.name + '</h3><span class="price">$' + p.price.toFixed(2) + '</span></div>' +
            '<p class="p-desc">' + p.desc + '</p>' +
          '</div>' +
          '<button type="button" class="btn btn-sm ' + (qty > 0 ? "btn-primary" : "btn-secondary") + '" style="width:100%;" data-add="' + p.id + '">' +
            (qty > 0 ? "Add another (" + qty + " in cart)" : "Add to Cart") +
          '</button>' +
        '</article>';
      }).join("");
    }

    function renderCart() {
      let count = 0, sub = 0;
      const rows = [];
      cart.forEach((qty, id) => {
        const p = PRODUCTS.find((x) => x.id === id);
        if (!p || qty <= 0) return;
        count += qty;
        sub += p.price * qty;
        rows.push(
          '<div class="cart-row">' +
            '<div style="min-width:0;"><div style="font-weight:500;">' + p.name + '</div><div class="mono" style="font-size:12px;color:var(--ink-2);">$' + (p.price * qty).toFixed(2) + '</div></div>' +
            '<div class="qty-ctrl">' +
              '<button type="button" class="qty-btn" data-dec="' + p.id + '">-</button>' +
              '<span class="mono">' + qty + '</span>' +
              '<button type="button" class="qty-btn" data-inc="' + p.id + '">+</button>' +
            '</div>' +
          '</div>'
        );
      });
      const ship = sub > 0 && sub < 100 ? 8 : 0;
      document.getElementById("navCartCount").textContent = String(count);
      document.getElementById("cartItemLabel").textContent = count + (count === 1 ? " item" : " items");
      document.getElementById("subTotal").textContent = "$" + sub.toFixed(2);
      document.getElementById("shippingFee").textContent = sub === 0 ? "$0.00" : (ship === 0 ? "Free" : "$" + ship.toFixed(2));
      document.getElementById("grandTotal").textContent = "$" + (sub + ship).toFixed(2);
      document.getElementById("cartItems").innerHTML = rows.length ? rows.join("") : '<p style="font-size:13px;color:var(--ink-2);">Add items from the catalog to calculate totals.</p>';
    }

    function addProduct(id) {
      cart.set(id, (cart.get(id) || 0) + 1);
      renderGrid();
      renderCart();
    }

    document.getElementById("catTabs").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-cat]");
      if (!btn) return;
      activeCat = btn.dataset.cat;
      renderCats();
      renderGrid();
    });

    document.getElementById("productSearch").addEventListener("input", (e) => {
      searchQ = e.target.value.trim().toLowerCase();
      renderGrid();
    });

    document.getElementById("productGrid").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-add]");
      if (btn) addProduct(Number(btn.dataset.add));
    });

    document.getElementById("quickAddHero").addEventListener("click", () => addProduct(PRODUCTS[0].id));

    document.getElementById("cartItems").addEventListener("click", (e) => {
      const inc = e.target.closest("[data-inc]");
      const dec = e.target.closest("[data-dec]");
      if (inc) addProduct(Number(inc.dataset.inc));
      if (dec) {
        const id = Number(dec.dataset.dec);
        const next = (cart.get(id) || 0) - 1;
        if (next <= 0) cart.delete(id); else cart.set(id, next);
        renderGrid();
        renderCart();
      }
    });

    document.getElementById("checkoutBtn").addEventListener("click", () => {
      const note = document.getElementById("checkoutNote");
      if (cart.size === 0) {
        note.textContent = "Select at least one product first.";
        return;
      }
      const total = document.getElementById("grandTotal").textContent;
      cart.clear();
      renderGrid();
      renderCart();
      note.textContent = "Order #" + Math.floor(10000 + Math.random() * 89999) + " confirmed (" + total + ").";
    });

    renderCats();
    renderGrid();
    renderCart();
  </script>
</body>
</html>`;

  return {
    title: brief.title,
    summary: `Built a complete **E-commerce Storefront** for **${brief.title}** following the **E-commerce Profile** (Featured Hero, Segmented Categories, Live Search, Product Cards, and Interactive Cart Calculator) in the Nexora design family.`,
    highlights: [
      "**E-commerce Profile Architecture:** Asymmetric spotlight hero, segmented category filter bar, real-time search input, responsive product grid, and interactive cart with live subtotal/shipping math.",
      "**Connected Catalog Facts:** Generated product cards and descriptions directly from the items, facts, and domain context in your conversation.",
      "**Unified Nexora Surfaces:** Consistent card elevation, crisp tabular pricing (`JetBrains Mono`), and long-text protection.",
    ],
    html,
  };
}

// ---------------------------------------------------------------------------
// Profile 4: Portfolio (Introduction + Projects + Skills + Contact)
// ---------------------------------------------------------------------------
function buildPortfolioHtml(brief: WebsiteFactBrief): {
  title: string;
  summary: string;
  highlights: string[];
  html: string;
} {
  const personName = brief.ownerName !== "Nexora Studio" ? brief.ownerName : brief.title;
  const safeName = escapeHtml(personName);
  const themeAttr = brief.preferDark ? ' data-theme="dark"' : "";
  const items = brief.extractedItems;
  const facts = brief.connectedFacts;

  const projects = [
    {
      id: 1,
      cat: "Systems",
      year: "2026",
      title: items[0] ? items[0].replace(/^\w/, (c) => c.toUpperCase()) : `${brief.title} Architecture`,
      summary:
        facts[0] ||
        `Designed and shipped end-to-end architecture with clear state boundaries, deterministic checks, and sub-50ms interactions.`,
      stack: brief.domain || "TypeScript · Systems",
    },
    {
      id: 2,
      cat: "Interface",
      year: "2025",
      title: items[1] ? items[1].replace(/^\w/, (c) => c.toUpperCase()) : "High-Craft Design System",
      summary:
        facts[1] ||
        `Built unified light/obsidian surface tokens, keyboard-first workflows, and tabular numeric reporting views.`,
      stack: "CSS Tokens · Accessibility",
    },
    {
      id: 3,
      cat: "Systems",
      year: "2025",
      title: items[2] ? items[2].replace(/^\w/, (c) => c.toUpperCase()) : "Connected Knowledge Engine",
      summary:
        facts[2] ||
        `Integrated local neural scoring with durable multi-session context so workflows preserve user intent accurately.`,
      stack: "Neural Engine · Persistence",
    },
  ];

  const skillColumns = [
    {
      group: "Core Engineering",
      skills: items.length >= 3 ? items.slice(0, 3) : ["TypeScript & Architecture", "API & State Modeling", "Automated Verification"],
    },
    {
      group: "Interface & Visual Craft",
      skills: ["Design Token Systems", "Responsive CSS Grid", "Accessible Interactions"],
    },
    {
      group: "Data & Tooling",
      skills: ["Performance Profiling", "Search & Ranking", "Cloud & Offline Sync"],
    },
  ];

  const html = `<!DOCTYPE html>
<html lang="en"${themeAttr}>
<head>
${buildDesignFamilyHead(`${personName} — Portfolio`, brief.accent)}
    .intro-hero {
      padding: clamp(36px, 6vw, 72px) 0 28px;
      display: grid; grid-template-columns: 1.35fr 0.65fr; gap: 28px; align-items: start;
    }
    @media (max-width: 820px) { .intro-hero { grid-template-columns: 1fr; } }
    .intro-hero h1 { font-size: clamp(2.1rem, 4.4vw, 3.25rem); line-height: 1.07; margin: 10px 0 14px; }
    .intro-hero p { font-size: 16px; color: var(--ink-2); margin-bottom: 22px; }
    .bio-card { padding: 20px; }
    .bio-row { padding: 10px 0; border-top: 1px solid var(--line); display: flex; justify-content: space-between; gap: 12px; font-size: 13px; }
    .bio-row:first-of-type { border-top: none; padding-top: 0; }
    .bio-row span { color: var(--ink-3); }
    .bio-row strong { color: var(--ink); font-weight: 500; text-align: right; }
    .proj-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
    .proj-card { padding: var(--space-card); display: flex; flex-direction: column; justify-content: space-between; gap: 14px; }
    .proj-card h3 { font-size: 17px; font-weight: 600; margin-top: 4px; }
    .proj-card p { font-size: 13.5px; color: var(--ink-2); line-height: 1.62; }
    .skills-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }
    .skill-col { padding: var(--space-card); }
    .skill-col h3 { font-size: 15px; font-weight: 600; margin-bottom: 12px; }
    .skill-list { list-style: none; border-top: 1px solid var(--line); }
    .skill-list li { padding: 9px 0; border-bottom: 1px solid var(--line); font-size: 13.5px; color: var(--ink-2); }
    .contact-card { padding: 24px; display: grid; grid-template-columns: 1fr 1.2fr; gap: 24px; align-items: start; }
    @media (max-width: 780px) { .contact-card { grid-template-columns: 1fr; } }
    .contact-form { display: flex; flex-direction: column; gap: 10px; }
    .note { font-size: 13px; color: var(--green); font-weight: 500; min-height: 18px; }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="container topbar-inner">
      <a class="brand" href="#">
        <span class="brand-mark">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1"><circle cx="12" cy="8" r="4"/><path d="M5 20a7 7 0 0 1 14 0"/></svg>
        </span>
        <span>${safeName}</span>
      </a>
      <nav class="nav-center" aria-label="Portfolio sections">
        <a class="nav-link" href="#projects">Projects</a>
        <a class="nav-link" href="#skills">Skills</a>
        <a class="nav-link" href="#contact">Contact</a>
      </nav>
      <div class="nav-actions">
        <button type="button" class="icon-btn" id="themeBtn" aria-label="Toggle theme">${SUN_MOON_ICON_SVG}</button>
        <a class="btn btn-sm btn-primary" href="#contact">Get in touch</a>
      </div>
    </div>
  </header>

  <main class="container">
    <section class="intro-hero">
      <div>
        <div class="meta-line">
          <span>${safeName}</span>
          <span>&middot;</span>
          <span>Portfolio</span>
          <span>&middot;</span>
          <span>${escapeHtml(brief.domain || "Software & Design")}</span>
        </div>
        <h1>Designing thoughtful products with durable technical craft.</h1>
        <p>${escapeHtml(brief.tagline)}</p>
        <div style="display:flex;flex-wrap:wrap;gap:10px;">
          <a class="btn btn-primary" href="#projects">Selected Projects</a>
          <a class="btn btn-secondary" href="#skills">Technical Skills</a>
        </div>
      </div>

      <aside class="surface-card bio-card" aria-label="Profile details">
        <div class="bio-row"><span>Focus</span><strong>${escapeHtml(brief.domain || "Full-Stack UI")}</strong></div>
        <div class="bio-row"><span>Velocity</span><strong>${escapeHtml(brief.metrics[0].value)}</strong></div>
        <div class="bio-row"><span>Primary Stack</span><strong>${escapeHtml(brief.metrics[1].value)}</strong></div>
        <div class="bio-row"><span>Availability</span><strong>${escapeHtml(brief.metrics[2].value)}</strong></div>
      </aside>
    </section>

    <section id="projects" class="section-block">
      <div class="section-header">
        <div>
          <h2>Selected Projects</h2>
          <p>Production systems and interfaces shaped around clear metrics and real user context.</p>
        </div>
        <div class="seg-tabs" role="tablist" aria-label="Project filter">
          <button type="button" class="seg-btn active" data-proj-filter="all">All Work</button>
          <button type="button" class="seg-btn" data-proj-filter="Systems">Systems</button>
          <button type="button" class="seg-btn" data-proj-filter="Interface">Interface</button>
        </div>
      </div>
      <div class="proj-grid" id="projGrid"></div>
    </section>

    <section id="skills" class="section-block">
      <div class="section-header">
        <div>
          <h2>Capabilities &amp; Skills</h2>
          <p>Core engineering disciplines organized cleanly without decorative clutter.</p>
        </div>
      </div>
      <div class="skills-grid">
        ${skillColumns
          .map(
            (col) => `<article class="surface-card skill-col">
          <h3>${escapeHtml(col.group)}</h3>
          <ul class="skill-list">
            ${col.skills.map((s) => `<li>${escapeHtml(String(s))}</li>`).join("")}
          </ul>
        </article>`
          )
          .join("")}
      </div>
    </section>

    <section id="contact" class="section-block">
      <div class="surface-card contact-card">
        <div>
          <h2 style="font-size:20px;margin-bottom:8px;">Start a Conversation</h2>
          <p style="font-size:14px;color:var(--ink-2);">Share project scope, timelines, or product challenges and I'll reply directly.</p>
        </div>
        <form id="contactForm" class="contact-form">
          <input id="cEmail" class="field-input" type="email" placeholder="you@company.com" required />
          <textarea id="cMsg" class="field-input" placeholder="Tell me about what you are building..." required></textarea>
          <button class="btn btn-primary" type="submit" style="align-self:flex-start;">Send Note</button>
          <div id="cNote" class="note" role="status"></div>
        </form>
      </div>
    </section>

    <footer class="site-footer">
      <span>&copy; ${new Date().getFullYear()} ${safeName}</span>
      <span>Portfolio &middot; Craft &amp; Engineering</span>
    </footer>
  </main>

  ${buildThemeScript(brief.preferDark)}
  <script>
    const PROJECTS = ${JSON.stringify(projects)};
    let projFilter = "all";

    function renderProjects() {
      const visible = projFilter === "all" ? PROJECTS : PROJECTS.filter((p) => p.cat === projFilter);
      document.getElementById("projGrid").innerHTML = visible.map((p) =>
        '<article class="surface-card proj-card">' +
          '<div>' +
            '<div class="meta-line"><span>' + p.cat + '</span><span>&middot;</span><span class="mono">' + p.year + '</span></div>' +
            '<h3>' + p.title + '</h3>' +
            '<p style="margin-top:8px;">' + p.summary + '</p>' +
          '</div>' +
          '<div class="meta-line" style="padding-top:12px;border-top:1px solid var(--line);"><span>' + p.stack + '</span></div>' +
        '</article>'
      ).join("");
    }

    document.querySelectorAll("[data-proj-filter]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-proj-filter]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        projFilter = btn.dataset.projFilter;
        renderProjects();
      });
    });

    document.getElementById("contactForm").addEventListener("submit", (e) => {
      e.preventDefault();
      const email = document.getElementById("cEmail").value.trim();
      document.getElementById("cNote").textContent = "Message received from " + email + " — thanks for reaching out.";
      e.target.reset();
    });

    renderProjects();
  </script>
</body>
</html>`;

  return {
    title: `${personName} — Portfolio`,
    summary: `Built a **Personal Portfolio** for **${personName}** following the **Portfolio Profile** (Introduction Hero, Filterable Projects, Grouped Skills Matrix, and Contact Section) in the Nexora design family.`,
    highlights: [
      "**Portfolio Profile Architecture:** Editorial introduction with profile facts card, interactive `All / Systems / Interface` project filter, structured 3-column skills matrix, and contact composer.",
      "**Personal & Context Connected:** Pulled your name (`" + personName + "`), skills, and topic facts directly from conversation memory and prompt details.",
      "**Zero Pill Clutter & Long-Text Safety:** Uses quiet inline metadata separators (`·`) and safe word-wrapping across all project narratives.",
    ],
    html,
  };
}

export function buildWebsiteByProfile(
  prompt: string,
  context?: WebsiteBuilderContext
): {
  title: string;
  profile: WebsiteProfileKind;
  summary: string;
  highlights: string[];
  html: string;
} | null {
  const brief = extractWebsiteFactBrief(prompt, context);
  if (
    brief.profile === "restaurant" ||
    brief.profile === "calculator" ||
    brief.profile === "todo" ||
    brief.profile === "login"
  ) {
    // Let specialized interactive app builders in pureTsIntelligence handle these with connected brief
    return null;
  }

  if (brief.profile === "dashboard") {
    return { ...buildDashboardHtml(brief), profile: "dashboard" };
  }
  if (brief.profile === "ecommerce") {
    return { ...buildEcommerceHtml(brief), profile: "ecommerce" };
  }
  if (brief.profile === "portfolio") {
    return { ...buildPortfolioHtml(brief), profile: "portfolio" };
  }
  return { ...buildSaasLandingHtml(brief), profile: "saas" };
}
