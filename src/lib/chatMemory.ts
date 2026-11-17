import type { WorkspaceThread } from "../data/workspaceThreads";
import type { AssistantHistoryTurn } from "./pureTsIntelligence";

export type MemoryCategory = "identity" | "preference" | "project" | "fact";

export interface ChatMemoryFact {
  id: string;
  category: MemoryCategory;
  key: string;
  value: string;
  sourceThreadId?: string;
  sourceThreadTitle?: string;
  updatedAt: number;
}

export interface ThreadMemoryDigest {
  threadId: string;
  title: string;
  updatedAt: string;
  summary: string;
  recentUserPrompts: string[];
  lastAssistantSnippet: string;
  tags: string[];
}

export interface MemoryExtractionResult {
  upserts: ChatMemoryFact[];
  forgetTargets: string[];
  clearAll: boolean;
}

const MEMORY_STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
  "for", "from", "have", "how", "i", "in", "is", "it", "last", "me", "my",
  "of", "on", "or", "our", "past", "previous", "recall", "remember", "saved",
  "that", "the", "this", "to", "talk", "talked", "earlier", "before", "we",
  "what", "when", "where", "which", "who", "you", "your", "about", "chat",
  "chats", "conversation", "conversations", "thread", "threads", "memory",
]);

function slugifyId(input: string): string {
  const cleaned = input
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 64);
  return cleaned || `mem_${Date.now()}`;
}

function cleanSnippet(text: string, maxLen = 180): string {
  const collapsed = text
    .replace(/```[\s\S]*?```/g, " [code/preview] ")
    .replace(/[#*_`>]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (collapsed.length <= maxLen) return collapsed;
  return `${collapsed.slice(0, maxLen).trim()}…`;
}

function extractTagsFromText(text: string): string[] {
  const matches = text
    .toLowerCase()
    .replace(/[^a-z0-9\s+#.-]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length >= 3 && !MEMORY_STOPWORDS.has(w));
  const counts = new Map<string, number>();
  for (const w of matches) {
    counts.set(w, (counts.get(w) ?? 0) + 1);
  }
  return Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([w]) => w);
}

export function buildThreadMemoryDigests(
  threads: WorkspaceThread[],
  excludeThreadId?: string | null
): ThreadMemoryDigest[] {
  return threads
    .filter((t) => t.id !== excludeThreadId && t.messages.length > 0)
    .map((t) => {
      const userMsgs = t.messages
        .filter((m) => m.role === "user" && m.content.trim())
        .map((m) => m.content.trim());
      const assistantMsgs = t.messages
        .filter((m) => m.role === "assistant" && m.content.trim() && m.jobStatus !== "error")
        .map((m) => m.content.trim());

      const recentUserPrompts = userMsgs.slice(-4).map((s) => cleanSnippet(s, 140));
      const lastAssistantSnippet =
        assistantMsgs.length > 0
          ? cleanSnippet(assistantMsgs[assistantMsgs.length - 1], 180)
          : "";
      const combined = [t.title, ...recentUserPrompts, lastAssistantSnippet].join(" ");
      const summaryParts: string[] = [];
      if (recentUserPrompts.length > 0) {
        summaryParts.push(`Asked: ${recentUserPrompts.slice(-2).join(" · ")}`);
      }
      if (lastAssistantSnippet) {
        summaryParts.push(`Outcome: ${lastAssistantSnippet}`);
      }
      return {
        threadId: t.id,
        title: t.title || "Untitled chat",
        updatedAt: t.updatedAt || "Recent",
        summary: summaryParts.join(" — ") || t.title,
        recentUserPrompts,
        lastAssistantSnippet,
        tags: extractTagsFromText(combined),
      };
    });
}

const INVALID_IDENTITY_VALUES = new Set([
  "here", "ready", "back", "tired", "happy", "sad", "good", "fine", "great",
  "okay", "ok", "confused", "curious", "testing", "test", "asking", "trying",
  "not sure", "learning", "new", "working", "building", "doing", "using",
  "looking", "sorry", "sure", "done", "online", "offline",
]);

export function extractMemoriesFromMessage(
  prompt: string,
  opts?: { threadId?: string; threadTitle?: string }
): MemoryExtractionResult {
  const clean = prompt.trim();
  const lower = clean.toLowerCase();
  const now = Date.now();
  const upserts: ChatMemoryFact[] = [];
  const forgetTargets: string[] = [];

  if (
    /^(?:please\s+)?(?:clear|reset|wipe|erase|delete|forget)\s+(?:all\s+(?:of\s+)?)?(?:your\s+|my\s+)?(?:memories|memory|everything\s+(?:about\s+me|you\s+know))[\s!?.]*$/i.test(
      lower
    )
  ) {
    return { upserts: [], forgetTargets: [], clearAll: true };
  }

  const forgetMatch = /^(?:please\s+)?forget\s+(?:that\s+|about\s+|my\s+)?(.+?)[.!?]*$/i.exec(
    clean
  );
  if (forgetMatch) {
    forgetTargets.push(forgetMatch[1].trim());
    return { upserts: [], forgetTargets, clearAll: false };
  }

  // Skip extracting facts from questions
  const isPureQuestion =
    /^(?:what|who|where|when|why|how|can|could|would|do|did|is|are)\b/i.test(lower) &&
    !/\bremember\s+that\b/i.test(lower);

  if (!isPureQuestion) {
    // 1. User's Name
    const nameMatch =
      /(?:\bmy\s+name\s+is\b|\bcall\s+me\b|\bi\s+am\s+called\b)\s+([a-zA-Z][a-zA-Z0-9 .'-]{1,40}?)(?=[,.!?;\n]|$|\s+\b(?:and|so|i|my|can|please|remember)\b)/i.exec(
        clean
      ) ??
      /^(?:hi|hello|hey)[,!\s]+i(?:'m|\s+am)\s+([A-Z][a-zA-Z0-9'-]{1,28})(?=[,.!?;\n]|$|\s+\b(?:and|can|please)\b)/.exec(
        clean
      );
    if (nameMatch) {
      const rawName = nameMatch[1].trim().replace(/[,.!?]+$/, "");
      if (
        rawName.length >= 2 &&
        rawName.length <= 40 &&
        !INVALID_IDENTITY_VALUES.has(rawName.toLowerCase())
      ) {
        upserts.push({
          id: "identity_user_name",
          category: "identity",
          key: "Name",
          value: rawName,
          sourceThreadId: opts?.threadId,
          sourceThreadTitle: opts?.threadTitle,
          updatedAt: now,
        });
      }
    }

    // 2. Role / Profession / Location / Brand
    const roleMatch =
      /\b(?:i\s+work\s+as|i\s+am\s+(?:a|an)|i'm\s+(?:a|an))\s+([a-zA-Z][a-zA-Z0-9 /&-]{2,50}?)(?=[,.!?;\n]|$|\s+\b(?:and|in|at|for|who|trying|building)\b)/i.exec(
        clean
      );
    if (roleMatch) {
      const roleVal = roleMatch[1].trim();
      if (roleVal.length >= 3 && !/^(?:user|person|human|test|bot|ai)$/i.test(roleVal)) {
        upserts.push({
          id: "identity_user_role",
          category: "identity",
          key: "Role",
          value: roleVal,
          sourceThreadId: opts?.threadId,
          sourceThreadTitle: opts?.threadTitle,
          updatedAt: now,
        });
      }
    }

    const brandMatch =
      /\bmy\s+(restaurant|business|company|shop|store|brand|app|project|website|startup)\s+(?:name\s+)?is\s+(?:called\s+)?["']?([a-zA-Z0-9][a-zA-Z0-9 &'._-]{1,48})["']?(?=[,.!?;\n]|$|\s+\b(?:and|so|make|build)\b)/i.exec(
        clean
      );
    if (brandMatch) {
      const kind = brandMatch[1].toLowerCase();
      const val = brandMatch[2].trim();
      upserts.push({
        id: `project_${slugifyId(kind)}`,
        category: "project",
        key: `${kind.charAt(0).toUpperCase() + kind.slice(1)} name`,
        value: val,
        sourceThreadId: opts?.threadId,
        sourceThreadTitle: opts?.threadTitle,
        updatedAt: now,
      });
    }

    // 3. Explicit "remember that ..." / "don't forget ..." / "note that ..."
    const explicitRemember =
      /\b(?:remember|don't\s+forget|do\s+not\s+forget|keep\s+in\s+mind|note)\s+(?:that\s+)?([^.!?\n]{4,180})/i.exec(
        clean
      );
    if (explicitRemember) {
      const factText = explicitRemember[1].trim().replace(/[,.!?]+$/, "");
      if (factText.length >= 4 && !/^(?:my\s+name\s+is|call\s+me)\b/i.test(factText)) {
        upserts.push({
          id: `fact_${slugifyId(factText.slice(0, 40))}`,
          category: "fact",
          key: "Note",
          value: factText,
          sourceThreadId: opts?.threadId,
          sourceThreadTitle: opts?.threadTitle,
          updatedAt: now,
        });
      }
    }

    // 4. Preferences ("my favorite X is Y", "i prefer X", "always use X")
    const favMatch =
      /\bmy\s+favorite\s+([a-zA-Z][a-zA-Z0-9 ]{1,24}?)\s+is\s+([^,.!?\n]{2,60})/i.exec(
        clean
      );
    if (favMatch) {
      const subj = favMatch[1].trim();
      const val = favMatch[2].trim();
      upserts.push({
        id: `pref_fav_${slugifyId(subj)}`,
        category: "preference",
        key: `Favorite ${subj}`,
        value: val,
        sourceThreadId: opts?.threadId,
        sourceThreadTitle: opts?.threadTitle,
        updatedAt: now,
      });
    }

    const prefMatch =
      /\b(?:i\s+prefer|i\s+usually\s+use|i\s+always\s+use|i\s+love|i\s+really\s+like)\s+([^,.!?\n]{2,80})/i.exec(
        clean
      );
    if (prefMatch) {
      const val = prefMatch[1].trim();
      if (!/^(?:to\s+ask|to\s+know|if\s+you|you\s+to|when\s+you)\b/i.test(val)) {
        upserts.push({
          id: `pref_${slugifyId(val.slice(0, 36))}`,
          category: "preference",
          key: "Preference",
          value: val,
          sourceThreadId: opts?.threadId,
          sourceThreadTitle: opts?.threadTitle,
          updatedAt: now,
        });
      }
    }
  }

  // Deduplicate by id within this message
  const seen = new Map<string, ChatMemoryFact>();
  for (const item of upserts) {
    seen.set(item.id, item);
  }
  return {
    upserts: Array.from(seen.values()),
    forgetTargets,
    clearAll: false,
  };
}

export function mergeMemoryFacts(
  existing: ChatMemoryFact[],
  incoming: ChatMemoryFact[],
  maxItems = 80
): ChatMemoryFact[] {
  const map = new Map<string, ChatMemoryFact>();
  for (const m of existing) {
    map.set(m.id, m);
  }
  for (const m of incoming) {
    const prev = map.get(m.id);
    if (!prev || m.updatedAt >= prev.updatedAt) {
      map.set(m.id, m);
    }
  }
  return Array.from(map.values())
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, maxItems);
}

export function applyMemoryExtraction(
  existing: ChatMemoryFact[],
  extraction: MemoryExtractionResult
): ChatMemoryFact[] {
  if (extraction.clearAll) return [];
  let next = [...existing];
  if (extraction.forgetTargets.length > 0) {
    const targetsLower = extraction.forgetTargets.map((t) => t.toLowerCase());
    next = next.filter((fact) => {
      const combined = `${fact.key} ${fact.value}`.toLowerCase();
      return !targetsLower.some(
        (t) =>
          combined.includes(t) ||
          (t.includes("name") && fact.id === "identity_user_name")
      );
    });
  }
  if (extraction.upserts.length > 0) {
    next = mergeMemoryFacts(next, extraction.upserts);
  }
  return next;
}

export function tryAnswerMemoryQuery(
  prompt: string,
  opts: {
    memoryFacts?: ChatMemoryFact[];
    threadDigests?: ThreadMemoryDigest[];
    history?: AssistantHistoryTurn[];
    extraction?: MemoryExtractionResult;
    activeThreadTitle?: string;
  }
): {
  markdown: string;
  reasoningNote: string;
  titleHint: string;
  thinkingLines: string[];
} | null {
  const clean = prompt.trim();
  const lower = clean.toLowerCase();
  const facts = opts.memoryFacts ?? [];
  const digests = opts.threadDigests ?? [];
  const history = opts.history ?? [];
  const extraction = opts.extraction ?? extractMemoriesFromMessage(clean);

  // 1. Handle Clear All Memory
  if (extraction.clearAll) {
    return {
      titleHint: "Memory Cleared",
      reasoningNote: "Cleared saved user memories across offline cache and synced storage.",
      thinkingLines: ["Cleared saved workspace memory facts"],
      markdown:
        "Done — I cleared my saved facts and preferences about you. Your existing chat threads remain intact until you delete them.",
    };
  }

  // 2. Handle Forget Specific Fact
  if (extraction.forgetTargets.length > 0) {
    return {
      titleHint: "Memory Updated",
      reasoningNote: `Removed matching memory entries for "${extraction.forgetTargets.join(", ")}".`,
      thinkingLines: [`Removed "${extraction.forgetTargets.join(", ")}" from memory`],
      markdown: `Got it — I removed **${extraction.forgetTargets.join(", ")}** from my saved memory.`,
    };
  }

  // 3. Name Recall ("What's my name?" / "Who am I?")
  if (
    /\b(?:what(?:'s|\s+is)\s+my\s+name|who\s+am\s+i|do\s+you\s+(?:know|remember)\s+my\s+name)\b/i.test(
      lower
    )
  ) {
    const nameFact = facts.find((f) => f.id === "identity_user_name");
    if (nameFact) {
      const extras = facts
        .filter((f) => f.id !== "identity_user_name")
        .slice(0, 3)
        .map((f) => `${f.key}: **${f.value}**`);
      return {
        titleHint: "Memory Recall",
        reasoningNote: `Retrieved persisted identity fact (${nameFact.key} = "${nameFact.value}") from workspace memory.`,
        thinkingLines: ["Queried persisted workspace memory", `Matched identity: ${nameFact.value}`],
        markdown:
          extras.length > 0
            ? `Your name is **${nameFact.value}**. I also remember: ${extras.join(" · ")}.`
            : `Your name is **${nameFact.value}**.`,
      };
    }
    return {
      titleHint: "Memory Recall",
      reasoningNote: "Checked persisted workspace memory; no user name fact stored yet.",
      thinkingLines: ["Queried workspace memory (no name stored yet)"],
      markdown:
        "I don't have your name saved yet — tell me `My name is ...` and I'll remember it across this chat and future sessions.",
    };
  }

  // 4. General Memory Recall ("What do you remember about me?" / "What did we talk about?" / "Previous chats")
  const asksAboutMemories =
    /\b(?:what\s+do\s+you\s+remember|do\s+you\s+remember\s+(?:me|anything|that|what)|list\s+(?:your\s+)?memories|show\s+(?:what\s+you\s+remember|my\s+memory)|what\s+have\s+i\s+told\s+you)\b/i.test(
      lower
    );

  const asksAboutPastChats =
    /\b(?:what\s+did\s+we\s+(?:talk|chat|discuss|build|make|work\s+on)|previous\s+chats?|past\s+chats?|other\s+chats?|earlier\s+chats?|chat\s+history|remember\s+(?:our\s+)?(?:chats?|conversations?)|did\s+we\s+(?:talk|discuss|build|make))\b/i.test(
      lower
    );

  const asksAboutLastTurn =
    /\b(?:what\s+did\s+i\s+(?:just\s+)?(?:ask|say|send|write)|my\s+last\s+(?:question|message|prompt)|what\s+were\s+we\s+(?:just\s+)?(?:talking|discussing))\b/i.test(
      lower
    );

  if (asksAboutLastTurn) {
    const userTurns = history.filter((t) => t.role === "user");
    const lastUser = userTurns[userTurns.length - 1];
    if (lastUser) {
      const prevAssistant = [...history].reverse().find((t) => t.role === "assistant");
      return {
        titleHint: "Current Chat Recall",
        reasoningNote: `Inspected active thread history (${history.length} turns) to recall the immediately preceding request.`,
        thinkingLines: [`Inspected ${history.length} recent turns in active chat`],
        markdown: [
          `In your last message in this chat, you wrote:`,
          "",
          `> ${cleanSnippet(lastUser.content, 260)}`,
          prevAssistant
            ? `\nAnd I replied with: _${cleanSnippet(prevAssistant.content, 200)}_`
            : "",
        ]
          .filter(Boolean)
          .join("\n"),
      };
    }
    if (digests.length > 0) {
      const latest = digests[0];
      return {
        titleHint: "Previous Chat Recall",
        reasoningNote: `No earlier user turns in the active thread; pulled latest saved thread "${latest.title}".`,
        thinkingLines: ["Checked active thread (empty) -> fell back to latest saved chat"],
        markdown: `This is your first message in this thread, so there isn't a previous turn here yet. In your most recent saved chat (**${latest.title}**), we covered:\n\n- ${latest.summary}`,
      };
    }
  }

  if (asksAboutMemories || asksAboutPastChats) {
    const queryWords = extractTagsFromText(clean);
    const matchingDigests =
      queryWords.length > 0
        ? digests
            .map((d) => {
              const hay = `${d.title} ${d.summary} ${d.tags.join(" ")}`.toLowerCase();
              const hits = queryWords.filter((w) => hay.includes(w)).length;
              return { d, hits };
            })
            .sort((a, b) => b.hits - a.hits)
            .map((x) => x.d)
        : digests;

    const currentUsers = history.filter((t) => t.role === "user").slice(-4);
    const sections: string[] = [];

    if (facts.length > 0) {
      sections.push(
        "#### Remembered Facts & Preferences",
        ...facts.slice(0, 10).map((f) => `- **${f.key}:** ${f.value}`)
      );
    }

    if (currentUsers.length > 0) {
      sections.push(
        "",
        `#### Current Chat${opts.activeThreadTitle ? ` (${opts.activeThreadTitle})` : ""}`,
        ...currentUsers.map((m) => `- ${cleanSnippet(m.content, 160)}`)
      );
    }

    if (matchingDigests.length > 0) {
      sections.push(
        "",
        "#### Saved Past Chats",
        ...matchingDigests
          .slice(0, 6)
          .map((d) => `- **${d.title}** _(${d.updatedAt})_: ${d.summary}`)
      );
    }

    if (sections.length === 0) {
      return {
        titleHint: "Workspace Memory",
        reasoningNote: "Checked persisted facts, active conversation, and saved thread digests.",
        thinkingLines: ["Checked persisted facts & thread history (empty)"],
        markdown:
          "I don't have any saved facts or past conversations yet. Anything you share here — like your name, preferences, or projects — is automatically saved and carried across chats.",
      };
    }

    return {
      titleHint: "Workspace Memory Recall",
      reasoningNote: `Retrieved ${facts.length} persisted memory fact(s), ${currentUsers.length} active turn(s), and ${matchingDigests.length} saved chat thread(s).`,
      thinkingLines: [
        `Loaded ${facts.length} persisted memory fact(s)`,
        `Indexed ${matchingDigests.length} saved chat thread(s)`,
      ],
      markdown: sections.join("\n"),
    };
  }

  // 5. Summarize active conversation ("summarize this chat" / "recap our conversation")
  if (
    /\b(summarize|recap)\s+(this\s+)?(chat|conversation|thread|so\s+far)\b/i.test(
      lower
    ) &&
    history.length >= 2
  ) {
    const pairs: string[] = [];
    for (let i = 0; i < history.length; i++) {
      if (history[i].role === "user") {
        const u = cleanSnippet(history[i].content, 140);
        const next = history[i + 1]?.role === "assistant" ? cleanSnippet(history[i + 1].content, 150) : null;
        pairs.push(next ? `- **You:** ${u}\n  - **Nexora:** ${next}` : `- **You:** ${u}`);
      }
    }
    return {
      titleHint: "Conversation Recap",
      reasoningNote: `Summarized ${history.length} turns in the active conversation.`,
      thinkingLines: [`Summarized ${history.length} turns in active chat`],
      markdown: [`### Conversation Recap`, "", ...pairs.slice(-6)].join("\n"),
    };
  }

  // 6. Pure Memory Introduction (only if no separate task is asked in the same prompt!)
  const hasActionTask =
    /\b(make|build|create|write|generate|calculate|solve|explain|how\s+to|what\s+is|compare|summarize|harvest|crawl|train)\b/i.test(
      lower
    );
  if (extraction.upserts.length > 0 && !hasActionTask && clean.length <= 160) {
    const savedList = extraction.upserts.map((u) => `**${u.key}:** ${u.value}`).join(" · ");
    const nameFact = extraction.upserts.find((u) => u.id === "identity_user_name");
    return {
      titleHint: "Saved to Memory",
      reasoningNote: `Extracted and persisted ${extraction.upserts.length} durable memory fact(s) to IndexedDB & workspace sync.`,
      thinkingLines: [`Saved ${extraction.upserts.length} memory fact(s)`],
      markdown: nameFact
        ? `Nice to meet you, **${nameFact.value}**! I've saved that (${savedList}) and will remember it across all your chats.`
        : `Got it — I've remembered that (${savedList}) and will carry it into future responses and chats.`,
    };
  }

  return null;
}
