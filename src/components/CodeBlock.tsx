"use client";

import { useCallback, useMemo, useState, type ReactNode } from "react";
import { Code2, Download, Eye } from "lucide-react";

/* ─────────────────────────────────────────────────────────
 * CODE BLOCK
 * A clean line-numbered code panel with syntax coloring,
 * wrapping lines, and one-click copy.
 * ───────────────────────────────────────────────────────── */

const FILE = "churn.ts";

const CODE_LINES = [
  "export async function churnBatch() {",
  '  const flavor = await getFlavor("pistachio");',
  "  const base = await dairy.fetch({ flavor });",
  '  await freezer.store(base, { temp: "-16C" });',
  "  if (!base.approved) return null;",
  "  return base.gallons;",
  "}",
];

export type CodeBlockLabels = { copy: string; copied: string };

/* light syntax coloring — keywords/imports/conditionals, functions, strings & numbers */
const KEYWORDS = new Set([
  "import",
  "from",
  "export",
  "default",
  "async",
  "function",
  "const",
  "let",
  "var",
  "await",
  "return",
  "if",
  "else",
  "for",
  "while",
  "new",
  "throw",
  "try",
  "catch",
  "null",
  "true",
  "false",
  "undefined",
]);
const TOKEN =
  /("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`[^`]*`|\b\d+(?:\.\d+)?\b|\b(?:import|from|export|default|async|function|const|let|var|await|return|if|else|for|while|new|throw|try|catch|null|true|false|undefined)\b|[A-Za-z_$][\w$]*(?=\s*\())/g;

function highlight(text: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let last = 0;
  let k = 0;
  for (const m of text.matchAll(TOKEN)) {
    const idx = m.index ?? 0;
    const t = m[0];
    if (idx > last) nodes.push(<span key={k++}>{text.slice(last, idx)}</span>);
    let color: string;
    let weight: number | undefined;
    if (/^["'`]/.test(t) || /^\d/.test(t)) color = "var(--orange)";
    else if (KEYWORDS.has(t)) color = "var(--accent-ink)";
    else {
      color = "var(--ink)";
      weight = 500;
    }
    nodes.push(
      <span key={k++} style={{ color, fontWeight: weight }}>
        {t}
      </span>
    );
    last = idx + t.length;
  }
  if (last < text.length) nodes.push(<span key={k++}>{text.slice(last)}</span>);
  return nodes;
}

function FileIcon() {
  return (
    <svg
      aria-hidden
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="shrink-0 text-ink-3"
    >
      <path d="M17.25 6.75 22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3-4.5 16.5" />
    </svg>
  );
}

const DEFAULT_LABELS: CodeBlockLabels = { copy: "Copy", copied: "Copied" };

export type CodeBlockProps = {
  /** The lines shown in the Code view. */
  lines?: string[];
  /** Raw text placed on the clipboard by Copy. Defaults to `lines` joined. */
  code?: string;
  /** Label shown in the header. */
  filename?: string;
  /** Prominent copy strings. */
  labels?: Partial<CodeBlockLabels>;
  /** Called with the copied text after a successful copy. */
  onCopy?: (text: string) => void;
  /** Expand to parent width */
  fill?: boolean;
  /** True while this specific block is actively streaming */
  isStreaming?: boolean;
};

function isHtmlSnippet(filename: string, raw: string): boolean {
  if (/\.(html?|xhtml)$/i.test(filename)) return true;
  const trimmed = raw.trim();
  return /^<!doctype\s+html/i.test(trimmed) || /^<html[\s>]/i.test(trimmed);
}

function buildHtmlSrcDoc(raw: string): string {
  const trimmed = raw.trim();
  if (/^<!doctype\s+html/i.test(trimmed) || /^<html[\s>]/i.test(trimmed)) {
    return raw;
  }
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<style>body{margin:0;padding:16px;font-family:system-ui,-apple-system,sans-serif;}</style>
</head>
<body>${raw}</body>
</html>`;
}

export default function CodeBlock({
  lines = CODE_LINES,
  code,
  filename = FILE,
  labels,
  onCopy,
  fill = false,
  isStreaming = false,
}: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const [manualViewMode, setManualViewMode] = useState<"preview" | "code" | null>(null);
  const text = { ...DEFAULT_LABELS, ...labels };
  const raw = code ?? lines.join("\n");

  const canPreviewHtml = useMemo(() => isHtmlSnippet(filename, raw), [filename, raw]);
  const viewMode: "preview" | "code" = canPreviewHtml
    ? (manualViewMode ?? (isStreaming ? "code" : "preview"))
    : "code";

  const srcDoc = useMemo(
    () => (viewMode === "preview" ? buildHtmlSrcDoc(raw) : ""),
    [viewMode, raw]
  );

  const copy = useCallback(() => {
    navigator.clipboard.writeText(raw).then(() => {
      setCopied(true);
      onCopy?.(raw);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [raw, onCopy]);

  const downloadFile = useCallback(() => {
    const blob = new Blob([raw], {
      type: canPreviewHtml ? "text/html;charset=utf-8" : "text/plain;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename || (canPreviewHtml ? "index.html" : "snippet.txt");
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  }, [raw, canPreviewHtml, filename]);

  return (
    <div
      className={`w-full overflow-hidden rounded-card bg-surface shadow-card ${
        fill ? "" : "max-w-105"
      }`}
    >
      <div className="flex h-11 items-center gap-2 border-b border-line px-4 text-[12.5px]">
        <span className="inline-flex min-w-0 items-center gap-[7px]">
          <FileIcon />
          <span className="truncate font-mono leading-none text-ink">{filename}</span>
        </span>

        <div className="-mr-1 ml-auto flex items-center gap-1">
          {canPreviewHtml && (
            <>
              <button
                type="button"
                aria-label="Download HTML file"
                title="Download HTML"
                onClick={downloadFile}
                className="flex h-7 w-7 items-center justify-center rounded-[6px] text-ink-3 transition-colors duration-100 hover:bg-hover hover:text-ink"
              >
                <Download className="size-3.5" />
              </button>

              <button
                type="button"
                aria-label={viewMode === "preview" ? "View HTML code" : "Show HTML preview"}
                title={viewMode === "preview" ? "View code" : "Show preview"}
                onClick={() =>
                  setManualViewMode(viewMode === "preview" ? "code" : "preview")
                }
                className="flex h-7 items-center gap-1 rounded-[6px] px-2 text-[12px] font-medium text-ink-2 transition-colors duration-100 hover:bg-hover hover:text-ink"
              >
                {viewMode === "preview" ? (
                  <>
                    <Code2 className="size-3.5" />
                    <span>Code</span>
                  </>
                ) : (
                  <>
                    <Eye className="size-3.5" />
                    <span>Preview</span>
                  </>
                )}
              </button>
            </>
          )}

          <button
            type="button"
            aria-label="Copy code"
            onClick={copy}
            className={`flex h-7 items-center gap-1 rounded-[6px] px-2 text-[12px]
              font-medium transition-colors duration-100 hover:bg-hover
              ${copied ? "text-green" : "text-ink-3 hover:text-ink"}`}
          >
            {copied ? (
              <svg
                width="11"
                height="11"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="3"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M20 6L9 17l-5-5" />
              </svg>
            ) : (
              <svg
                width="11"
                height="11"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="9" y="9" width="12" height="12" rx="2.5" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
            )}
            {copied ? text.copied : text.copy}
          </button>
        </div>
      </div>

      {viewMode === "preview" ? (
        <div className="w-full overflow-hidden bg-white">
          <iframe
            title={filename || "HTML Preview"}
            srcDoc={srcDoc}
            sandbox="allow-scripts"
            className="block h-[380px] w-full border-0 bg-white sm:h-[420px]"
          />
        </div>
      ) : (
        <div className="px-4 py-3 font-mono text-[12.5px] leading-[1.65] text-ink-2">
          <div className="flex flex-col">
            {lines.map((line, i) => (
              <code key={i} className="block break-words whitespace-pre-wrap">
                {line === "" ? "\u00A0" : highlight(line)}
              </code>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
