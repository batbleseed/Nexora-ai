import type { MLCEngineInterface, InitProgressReport } from "@mlc-ai/web-llm";
import type { LocalModelTier } from "./offlineStore";

export interface LocalModelPreset {
  tier: LocalModelTier;
  label: string;
  shortLabel: string;
  sizeLabel: string;
  description: string;
  f16Id: string;
  f32Id: string;
  isQwen3Family: boolean;
}

export const LOCAL_MODEL_PRESETS: LocalModelPreset[] = [
  {
    tier: "qwen3.5-4b",
    label: "Qwen 3.5 · 4B (Max Quality)",
    shortLabel: "Qwen 3.5 4B",
    sizeLabel: "~2.5 GB",
    description:
      "Flagship on-device Qwen 3.5 reasoning & synthesis — surpasses Qwen 2.5 locally (needs ~4 GB+ GPU VRAM).",
    f16Id: "Qwen3.5-4B-q4f16_1-MLC",
    f32Id: "Qwen3.5-4B-q4f32_1-MLC",
    isQwen3Family: true,
  },
  {
    tier: "qwen3.5-2b",
    label: "Qwen 3.5 · 2B (Recommended)",
    shortLabel: "Qwen 3.5 2B",
    sizeLabel: "~1.4 GB",
    description:
      "Fast Qwen 3.5 generation with <think> reasoning and strong coding/chat quality — runs smoothly on most WebGPU GPUs.",
    f16Id: "Qwen3.5-2B-q4f16_1-MLC",
    f32Id: "Qwen3.5-2B-q4f32_1-MLC",
    isQwen3Family: true,
  },
  {
    tier: "qwen3-1.7b",
    label: "Qwen 3 · 1.7B (Balanced)",
    shortLabel: "Qwen 3 1.7B",
    sizeLabel: "~1.1 GB",
    description:
      "Compact Qwen 3 model with structured reasoning and fast token throughput.",
    f16Id: "Qwen3-1.7B-q4f16_1-MLC",
    f32Id: "Qwen3-1.7B-q4f32_1-MLC",
    isQwen3Family: true,
  },
  {
    tier: "qwen3.5-0.8b",
    label: "Qwen 3.5 · 0.8B (Compact)",
    shortLabel: "Qwen 3.5 0.8B",
    sizeLabel: "~550 MB",
    description:
      "Lightweight Qwen 3.5 build for modest GPUs and quick downloads.",
    f16Id: "Qwen3.5-0.8B-q4f16_1-MLC",
    f32Id: "Qwen3.5-0.8B-q4f32_1-MLC",
    isQwen3Family: true,
  },
  {
    tier: "smollm2-360m",
    label: "SmolLM2 · 360M (Ultra-Light)",
    shortLabel: "SmolLM2 360M",
    sizeLabel: "~300 MB",
    description:
      "Tiny fallback model for basic short replies on low-VRAM devices.",
    f16Id: "SmolLM2-360M-Instruct-q4f16_1-MLC",
    f32Id: "SmolLM2-360M-Instruct-q4f32_1-MLC",
    isQwen3Family: false,
  },
];

export const DEFAULT_LOCAL_TIER: LocalModelTier = "qwen3.5-2b";

export function getLocalModelPreset(tier?: LocalModelTier): LocalModelPreset {
  return (
    LOCAL_MODEL_PRESETS.find((p) => p.tier === tier) ??
    LOCAL_MODEL_PRESETS.find((p) => p.tier === DEFAULT_LOCAL_TIER)!
  );
}

// Keep legacy constants exported for compatibility if referenced elsewhere
export const LOCAL_MODEL_F16 = LOCAL_MODEL_PRESETS[1].f16Id;
export const LOCAL_MODEL_F32 = LOCAL_MODEL_PRESETS[1].f32Id;

export interface WebGpuCapability {
  supported: boolean;
  hasF16: boolean;
  modelId: string;
  preset: LocalModelPreset;
  reason?: string;
}

export interface LocalLoadProgress {
  progress: number;
  text: string;
}

let activeEngine: MLCEngineInterface | null = null;
let activeModelId: string | null = null;
let loadPromise: Promise<MLCEngineInterface> | null = null;
let loadingModelId: string | null = null;

export async function checkWebGpuCapability(
  tier: LocalModelTier = DEFAULT_LOCAL_TIER
): Promise<WebGpuCapability> {
  const preset = getLocalModelPreset(tier);
  if (typeof navigator === "undefined" || !("gpu" in navigator) || !navigator.gpu) {
    return {
      supported: false,
      hasF16: false,
      modelId: preset.f32Id,
      preset,
      reason:
        "WebGPU is not exposed by this browser. Use a recent Chrome, Edge, or Chromium browser with hardware acceleration enabled.",
    };
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return {
        supported: false,
        hasF16: false,
        modelId: preset.f32Id,
        preset,
        reason:
          "No WebGPU GPU adapter was found on this device or browser session. Enable GPU/hardware acceleration in browser settings.",
      };
    }

    const hasF16 = adapter.features.has("shader-f16");
    return {
      supported: true,
      hasF16,
      modelId: hasF16 ? preset.f16Id : preset.f32Id,
      preset,
    };
  } catch (err) {
    return {
      supported: false,
      hasF16: false,
      modelId: preset.f32Id,
      preset,
      reason:
        err instanceof Error
          ? `WebGPU adapter check failed: ${err.message}`
          : "WebGPU initialization failed in this browser.",
    };
  }
}

export async function isAnyLocalModelCached(
  tier?: LocalModelTier
): Promise<boolean> {
  if (typeof caches === "undefined") return false;
  try {
    const { hasModelInCache, prebuiltAppConfig } = await import("@mlc-ai/web-llm");
    const targets = tier ? [getLocalModelPreset(tier)] : LOCAL_MODEL_PRESETS;
    const checks = await Promise.all(
      targets.flatMap((preset) => [
        hasModelInCache(preset.f16Id, prebuiltAppConfig).catch(() => false),
        hasModelInCache(preset.f32Id, prebuiltAppConfig).catch(() => false),
      ])
    );
    return checks.some(Boolean);
  } catch {
    return false;
  }
}

export async function ensureLocalEngineLoaded(
  tier: LocalModelTier = DEFAULT_LOCAL_TIER,
  onProgress?: (info: LocalLoadProgress) => void
): Promise<MLCEngineInterface> {
  const cap = await checkWebGpuCapability(tier);
  if (!cap.supported) {
    throw new Error(cap.reason || "WebGPU is unavailable in this browser.");
  }

  if (activeEngine && activeModelId === cap.modelId) {
    onProgress?.({ progress: 100, text: `${cap.preset.shortLabel} ready on WebGPU.` });
    return activeEngine;
  }

  if (loadPromise && loadingModelId === cap.modelId) {
    return loadPromise;
  }

  if (activeEngine && activeModelId !== cap.modelId) {
    try {
      await activeEngine.unload();
    } catch {
      // Ignore unload errors when switching local tiers
    }
    activeEngine = null;
    activeModelId = null;
  }

  loadingModelId = cap.modelId;
  loadPromise = (async () => {
    const { CreateMLCEngine } = await import("@mlc-ai/web-llm");
    const engine = await CreateMLCEngine(cap.modelId, {
      initProgressCallback: (report: InitProgressReport) => {
        const pct = Math.max(0, Math.min(100, Math.round((report.progress || 0) * 100)));
        onProgress?.({
          progress: pct,
          text:
            report.text ||
            `Loading ${cap.preset.shortLabel} WebGPU shaders and model weights…`,
        });
      },
    });
    activeEngine = engine;
    activeModelId = cap.modelId;
    return engine;
  })();

  try {
    return await loadPromise;
  } finally {
    loadPromise = null;
    loadingModelId = null;
  }
}

function stripTrailingRepetitionLoop(text: string): {
  cleaned: string;
  loopDetected: boolean;
} {
  if (text.length < 40) return { cleaned: text, loopDetected: false };
  const phraseLoopMatch = text.match(/(.{3,48}?)\1{3,}$/);
  if (phraseLoopMatch && phraseLoopMatch.index !== undefined) {
    const keepEnd = phraseLoopMatch.index + phraseLoopMatch[1].length;
    return { cleaned: text.slice(0, keepEnd).trimEnd(), loopDetected: true };
  }
  const wordLoopMatch = text.match(/(\b\w+(?:\s+\w+){0,5})\s*(?:\1\s*){3,}$/i);
  if (wordLoopMatch && wordLoopMatch.index !== undefined) {
    const keepEnd = wordLoopMatch.index + wordLoopMatch[1].length;
    return { cleaned: text.slice(0, keepEnd).trimEnd(), loopDetected: true };
  }
  return { cleaned: text, loopDetected: false };
}

export async function streamLocalWebGpuChat(opts: {
  prompt: string;
  history: Array<{ role: "user" | "assistant"; content: string }>;
  tier?: LocalModelTier;
  assistantName?: string;
  researchContext?: string;
  language?: string;
  effort?: string;
  onProgress?: (info: LocalLoadProgress) => void;
  onThought?: (delta: string) => void;
  onDelta: (delta: string) => void;
  onReplaceText?: (text: string) => void;
  signal?: AbortSignal;
}): Promise<string> {
  const tier = opts.tier ?? DEFAULT_LOCAL_TIER;
  const preset = getLocalModelPreset(tier);
  const engine = await ensureLocalEngineLoaded(tier, opts.onProgress);
  if (opts.signal?.aborted) {
    return "";
  }

  const languageDirective =
    opts.language && !opts.language.toLowerCase().startsWith("english")
      ? ` Respond in ${opts.language} unless asked otherwise.`
      : "";

  const effortDirective =
    opts.effort === "Low"
      ? " Be concise, direct, and skip unnecessary preamble."
      : opts.effort === "Max Effort"
        ? " Think step by step carefully before answering, and write clear, structured Markdown."
        : " Give a clear, accurate, well-structured Markdown answer.";

  const maxTokens = preset.isQwen3Family
    ? opts.effort === "Low"
      ? 512
      : opts.effort === "Max Effort"
        ? 1600
        : 1024
    : opts.effort === "Low"
      ? 280
      : opts.effort === "Max Effort"
        ? 900
        : 512;

  const temperature = preset.isQwen3Family
    ? opts.effort === "Low"
      ? 0.35
      : opts.effort === "Max Effort"
        ? 0.6
        : 0.5
    : opts.effort === "Low"
      ? 0.25
      : opts.effort === "Max Effort"
        ? 0.5
        : 0.35;

  const assistantIdentity = opts.assistantName || "Nexora Assistant";
  const groundingBlock = opts.researchContext
    ? `\n\nGrounded public reference context retrieved locally for this question (prioritize these facts if relevant, and never copy promotional fluff):\n${opts.researchContext}`
    : "";

  // For Qwen 3 / 3.5, appending /no_think on Low effort gives instant answers without long CoT pauses
  const userContent =
    preset.isQwen3Family && opts.effort === "Low"
      ? `${opts.prompt || "Hello"}\n/no_think`
      : opts.prompt || "Hello";

  const messages: Array<{ role: "system" | "user" | "assistant"; content: string }> = [
    {
      role: "system",
      content: `You are ${assistantIdentity}, powered natively on-device by ${preset.shortLabel} via WebGPU with zero external AI providers.${effortDirective}${languageDirective}${groundingBlock}`,
    },
    ...opts.history.slice(-8).map((turn) => ({
      role: turn.role,
      content: turn.content,
    })),
    {
      role: "user",
      content: userContent,
    },
  ];

  const onAbort = () => {
    try {
      engine.interruptGenerate();
    } catch {
      // Ignore interrupt errors
    }
  };
  opts.signal?.addEventListener("abort", onAbort, { once: true });

  let fullText = "";
  let inThink = false;
  let pending = "";
  let loopStopped = false;

  const pushAnswerChunk = (out: string) => {
    fullText += out;
    const { cleaned, loopDetected } = stripTrailingRepetitionLoop(fullText);
    if (loopDetected) {
      fullText = cleaned;
      loopStopped = true;
      opts.onReplaceText?.(fullText);
      try {
        engine.interruptGenerate();
      } catch {
        // Ignore interrupt errors
      }
      return;
    }
    opts.onDelta(out);
  };

  const emitSplit = (chunkText: string) => {
    pending += chunkText;
    while (pending.length > 0 && !loopStopped) {
      if (!inThink) {
        const openIdx = pending.indexOf("<think>");
        if (openIdx === -1) {
          const safeLen = pending.endsWith("<") ? pending.length - 1 : pending.length;
          if (safeLen > 0) {
            const out = pending.slice(0, safeLen);
            pending = pending.slice(safeLen);
            pushAnswerChunk(out);
          }
          break;
        }
        if (openIdx > 0) {
          const out = pending.slice(0, openIdx);
          pushAnswerChunk(out);
        }
        pending = pending.slice(openIdx + 7);
        inThink = true;
      } else {
        const closeIdx = pending.indexOf("</think>");
        if (closeIdx === -1) {
          const safeLen = pending.endsWith("<") ? pending.length - 1 : pending.length;
          if (safeLen > 0) {
            opts.onThought?.(pending.slice(0, safeLen));
            pending = pending.slice(safeLen);
          }
          break;
        }
        if (closeIdx > 0) {
          opts.onThought?.(pending.slice(0, closeIdx));
        }
        pending = pending.slice(closeIdx + 8);
        inThink = false;
      }
    }
  };

  try {
    const stream = await engine.chat.completions.create({
      messages,
      stream: true,
      temperature,
      top_p: 0.9,
      frequency_penalty: preset.isQwen3Family ? 0.15 : 0.25,
      presence_penalty: 0.05,
      max_tokens: maxTokens,
    });

    for await (const chunk of stream) {
      if (opts.signal?.aborted || loopStopped) break;
      const delta = chunk.choices?.[0]?.delta?.content ?? "";
      if (delta) {
        emitSplit(delta);
      }
    }
    if (pending && !loopStopped) {
      if (inThink) {
        opts.onThought?.(pending);
      } else {
        pushAnswerChunk(pending);
      }
    }
    return fullText;
  } finally {
    opts.signal?.removeEventListener("abort", onAbort);
  }
}

export async function clearLocalModelCache(tier?: LocalModelTier): Promise<void> {
  if (activeEngine) {
    try {
      await activeEngine.unload();
    } catch {
      // Ignore unload failure
    }
    activeEngine = null;
    activeModelId = null;
  }

  try {
    const { deleteModelAllInfoInCache, prebuiltAppConfig } = await import("@mlc-ai/web-llm");
    const targets = tier ? [getLocalModelPreset(tier)] : LOCAL_MODEL_PRESETS;
    await Promise.all(
      targets.flatMap((preset) => [
        deleteModelAllInfoInCache(preset.f16Id, prebuiltAppConfig).catch(() => {}),
        deleteModelAllInfoInCache(preset.f32Id, prebuiltAppConfig).catch(() => {}),
      ])
    );
  } catch {
    // Ignore cache deletion error if module unavailable
  }

  if (typeof caches !== "undefined") {
    await caches.delete("nexora-local-model-v1").catch(() => {});
  }
}
