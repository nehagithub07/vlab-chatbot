import { NextRequest, NextResponse } from "next/server";
import { DataAPIClient } from "@datastax/astra-db-ts";
import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from "@langchain/google-genai";

type Embedder = (text: string) => Promise<number[]>;

async function getEmbedder(): Promise<{
  embed: Embedder;
  dim: number;
  collectionName: string;
}> {
  const {
    GEMINI_API_KEY,
    ASTRA_DB_COLLECTION = "experiment_docs",
    EMBED_PROVIDER = "google",
  } = process.env as Record<string, string>;

  const provider = (EMBED_PROVIDER || "google").toLowerCase();
  if (
    provider === "multimodal" ||
    provider === "google-mm" ||
    provider === "mm" ||
    /multimodal/.test(String(process.env.EMBED_MODEL || ""))
  ) {
    if (!GEMINI_API_KEY) throw new Error("GEMINI_API_KEY is required for multimodalembedding@001.");
    const MM_URL = "https://generativelanguage.googleapis.com/v1beta/models/multimodalembedding@001:embedContent";
    const embed: Embedder = async (text: string) => {
      const res = await fetch(MM_URL + `?key=${encodeURIComponent(GEMINI_API_KEY)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: { parts: [{ text }] } }),
      } as any);
      if (!res.ok) throw new Error(`Multimodal embed failed: ${res.status}`);
      const data: any = await res.json();
      const values: number[] | undefined = data?.embedding?.values;
      if (!Array.isArray(values)) throw new Error("No embedding.values returned");
      return values;
    };
    return { embed, dim: 1408, collectionName: `${ASTRA_DB_COLLECTION}_mm` };
  }
  if (provider === "xenova") {
    const { pipeline } = await import("@xenova/transformers");
    const extractor: any = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    const embed: Embedder = async (text: string) => {
      const output: any = await extractor(text, { pooling: "mean", normalize: true });
      const data: Float32Array = output?.data ?? output?.tolist?.();
      return Array.from(data);
    };
    return { embed, dim: 384, collectionName: `${ASTRA_DB_COLLECTION}_xv` };
  }

  if (!GEMINI_API_KEY) {
    throw new Error(
      "GEMINI_API_KEY is required for google embeddings. Set EMBED_PROVIDER=xenova to use local embeddings."
    );
  }
  const embeddings = new GoogleGenerativeAIEmbeddings({ apiKey: GEMINI_API_KEY, model: "text-embedding-004" });
  const embed: Embedder = async (text: string) => {
    const out: any = await embeddings.embedQuery(text);
    if (Array.isArray(out) && typeof out[0] === "number") return out as number[];
    const vec = out?.data?.[0]?.embedding as number[] | undefined;
    if (!vec) throw new Error("Failed to get embedding vector from Google API response");
    return vec;
  };
  return { embed, dim: 768, collectionName: ASTRA_DB_COLLECTION };
}

async function retrieveContext(query: string) {
  const {
    ASTRA_DB_API_ENDPOINT,
    ASTRA_DB_APPLICATION_TOKEN,
    ASTRA_DB_NAMESPACE,
  } = process.env as Record<string, string>;

  if (!ASTRA_DB_API_ENDPOINT || !ASTRA_DB_APPLICATION_TOKEN || !ASTRA_DB_NAMESPACE) {
    throw new Error("Missing Astra DB env: ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_NAMESPACE");
  }

  const { embed, collectionName } = await getEmbedder();

  const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
  const db = client.db(ASTRA_DB_API_ENDPOINT, { keyspace: ASTRA_DB_NAMESPACE });
  const coll = db.collection(collectionName);

  // Query augmentation
  const augTerms: string[] = ["experiment", "lab"];
  if (/procedure/i.test(query)) augTerms.push("procedure", "steps");
  if (/precaution/i.test(query)) augTerms.push("precaution", "safety");
  if (/(apparatus|equipment)/i.test(query)) augTerms.push("apparatus", "equipment", "setup");
  if (/(objective|aim)/i.test(query)) augTerms.push("objective", "aim");
  if (/(analy[sz]e|analysis|calculation|result)/i.test(query)) augTerms.push("analysis", "calculate", "results");
  if (/(theory|definition|principle)/i.test(query)) augTerms.push("theory", "definition", "principle");
  const augmentedQuery = `${query} ${augTerms.join(" ")}`.trim();
  const vector = await embed(augmentedQuery);

  // Attempt hybrid first, then vector-only with similarity
  let docs: any[] = [];
  try {
    const anyColl: any = coll as any;
    const cur = anyColl.findAndRerank?.({});
    if (cur && typeof cur.sort === "function") {
      const sorted = cur.sort({ $hybrid: { $vector: vector, $lexical: augmentedQuery } });
      const limited = typeof sorted.limit === "function" ? sorted.limit(12) : sorted;
      docs = await limited.toArray();
    }
  } catch {}

  if (!Array.isArray(docs) || docs.length === 0) {
    const cursor = coll.find({}, { limit: 12 }).includeSimilarity?.(true).sort({ $vector: vector });
    docs = await cursor.toArray();
  }

  // Determine top similarity reliably
  const simFromDoc = (d: any): number | null => {
    const cand =
      typeof d?.$similarity === "number"
        ? d.$similarity
        : typeof d?.similarity === "number"
        ? d.similarity
        : typeof d?.score === "number"
        ? d.score
        : typeof d?.$score === "number"
        ? d.$score
        : null;
    return typeof cand === "number" ? cand : null;
  };

  let topSimilarity: number | null = null;
  let topTextSimilarity: number | null = null;
  if (docs?.length) topSimilarity = simFromDoc(docs[0]);
  if (topSimilarity == null) {
    try {
      const cur = coll.find({}, { limit: 1 }).includeSimilarity?.(true).sort({ $vector: vector });
      const vd = await cur.toArray();
      if (vd?.length) topSimilarity = simFromDoc(vd[0]);
    } catch {}
  }
  // Also compute similarity restricted to text docs so web fallback isn't blocked by image-only matches
  try {
    const curText = coll.find({ type: 'text' } as any, { limit: 1 }).includeSimilarity?.(true).sort({ $vector: vector });
    const td = await curText.toArray();
    if (td?.length) topTextSimilarity = simFromDoc(td[0]);
  } catch {}
  if ((docs?.length ?? 0) === 0 && topSimilarity == null) topSimilarity = 0;

  // Build context string and collect image doc paths
  const rawTexts: string[] = [];
  const imagePaths: string[] = [];
  for (const d of docs as any[]) {
    if (typeof d?.text === "string" && d.text.trim()) rawTexts.push(d.text);
    const pth: string | undefined = typeof d?.path === "string" ? d.path : undefined;
    const typ = d?.type;
    const isImg = typ === "image" || (typeof d?.mime === "string" && d.mime.startsWith("image/"));
    if (isImg && pth) imagePaths.push(pth.replace(/\\/g, "/"));
  }
  const context = rawTexts.slice(0, 5).join("\n\n---\n\n");
  return { context, sources: rawTexts.slice(0, 5), images: imagePaths.slice(0, 12), topSimilarity, topTextSimilarity };
}

// Tavily restricted educational search
const ALLOWED_EDU_DOMAINS = ["vlab.co.in", "nptel.ac.in", "wikipedia.org"] as const;
async function tavilyEduSearch(query: string) {
  const { TAVILY_API_KEY } = process.env as Record<string, string>;
  if (!TAVILY_API_KEY) return { webBlob: "", citations: [] as Array<{ title: string; url: string }> };
  try {
    const res = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        api_key: TAVILY_API_KEY,
        query,
        search_depth: "advanced",
        max_results: 6,
        include_answer: false,
        include_raw_content: true,
        include_images: false,
        include_domains: [...ALLOWED_EDU_DOMAINS],
      }),
    } as RequestInit);
    if (!res.ok) throw new Error(`Tavily error ${res.status}`);
    const data: any = await res.json();
    const results: any[] = Array.isArray(data?.results) ? data.results : [];
    const allowed = results.filter((r) => {
      try {
        const host = new URL(r?.url || "").hostname.replace(/^www\./, "");
        return (ALLOWED_EDU_DOMAINS as readonly string[]).some((d) => host.endsWith(d));
      } catch {
        return false;
      }
    });
    const stitched = allowed
      .slice(0, 6)
      .map((r) => `Title: ${r?.title || ""}\nURL: ${r?.url || ""}\nContent: ${(r?.raw_content || r?.content || "").slice(0, 2000)}`)
      .join("\n\n---\n\n");
    const citations = allowed.slice(0, 6).map((r) => ({ title: r?.title || "", url: r?.url || "" }));
    return { webBlob: stitched, citations };
  } catch {
    return { webBlob: "", citations: [] as Array<{ title: string; url: string }> };
  }
}

// Utility: compute resistor color code mapping for a given ohmic value
function computeResistorColorCode(ohms: number) {
  const digitColor = [
    "Black", "Brown", "Red", "Orange", "Yellow",
    "Green", "Blue", "Violet", "Grey", "White",
  ];
  const multiplierColor = [
    "Black", "Brown", "Red", "Orange", "Yellow",
    "Green", "Blue", "Violet", "Grey", "White",
  ];
  if (!isFinite(ohms) || ohms <= 0) return null;
  // 4‑band: 2 significant digits + multiplier + tolerance (default Gold ±5%)
  let p4 = 0;
  let m4 = ohms;
  while (m4 >= 100) { m4 /= 10; p4++; }
  while (m4 < 10) { m4 *= 10; p4--; }
  const d1_4 = Math.floor(m4 / 10);
  const d2_4 = Math.floor(m4 % 10);
  const mult4 = p4;
  if (d1_4 < 0 || d1_4 > 9 || d2_4 < 0 || d2_4 > 9 || mult4 < 0 || mult4 > 9) return null;
  const fourBand = [digitColor[d1_4], digitColor[d2_4], multiplierColor[mult4], "Gold (±5%)"];

  // 5‑band: 3 significant digits + multiplier + tolerance (default Gold ±5%)
  let p5 = 0;
  let m5 = ohms;
  while (m5 >= 1000) { m5 /= 10; p5++; }
  while (m5 < 100) { m5 *= 10; p5--; }
  const d1_5 = Math.floor(m5 / 100) % 10;
  const d2_5 = Math.floor(m5 / 10) % 10;
  const d3_5 = Math.floor(m5 % 10);
  const mult5 = p5;
  if ([d1_5,d2_5,d3_5].some(d => d < 0 || d > 9) || mult5 < 0 || mult5 > 9) return null;
  const fiveBand = [digitColor[d1_5], digitColor[d2_5], digitColor[d3_5], multiplierColor[mult5], "Gold (±5%)"];

  return { fourBand, fiveBand };
}

// Extract one or more ohmic values from a question string (supports k/K/M suffixes)
function extractOhmValues(text: string): number[] {
  const out: number[] = [];
  const seen = new Set<number>();
  const re = /(\d+(?:\.\d+)?)\s*(k|K|M|mega|G|g)?/g;
  for (const m of text.matchAll(re)) {
    const n = parseFloat(m[1]);
    if (!isFinite(n)) continue;
    const suf = (m[2] || '').toLowerCase();
    const factor = suf === 'k' ? 1e3 : suf === 'm' || suf === 'mega' ? 1e6 : suf === 'g' ? 1e9 : 1;
    const val = n * factor;
    if (!seen.has(val)) {
      seen.add(val);
      out.push(val);
    }
  }
  return out;
}

async function summarizeAndAnswer(question: string, astraContext: string, webBlob?: string) {
  const { GEMINI_API_KEY, GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_API_VERSION } =
    process.env as Record<string, string>;
  const apiKey = GOOGLE_API_KEY || GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("Missing Google API key. Set GOOGLE_API_KEY (preferred) or GEMINI_API_KEY in your environment.");
  }

  const instruction = `You are the Virtual Lab Assistant.
Follow these rules strictly:
1) Use Astra DB lab context as the primary source.
2) If a web summary is provided, integrate only relevant details from it; do not contradict Astra.
3) If both sources lack the necessary information, answer exactly: "I don't know."
4) Keep tone academic, helpful, and factual.
5) Be concise and structured with bullet points when appropriate.`;

  const fullPrompt = `${instruction}

Astra Context:
${astraContext || "(none)"}

Web Summary (may be empty):
${webBlob || "(none)"}

Question: ${question}

Answer in Markdown:`;

  const preferred: Array<{ v: string; m: string }> = [];
  if (GEMINI_MODEL && GEMINI_API_VERSION) preferred.push({ v: GEMINI_API_VERSION, m: GEMINI_MODEL });
  if (GEMINI_MODEL) preferred.push({ v: "v1", m: GEMINI_MODEL }, { v: "v1beta", m: GEMINI_MODEL });
  preferred.push(
    { v: "v1", m: "gemini-1.5-flash" },
    { v: "v1", m: "gemini-1.5-flash-8b" },
    { v: "v1", m: "gemini-1.5-pro" },
    { v: "v1beta", m: "gemini-2.0-flash" },
    { v: "v1beta", m: "gemini-1.5-flash" },
    { v: "v1beta", m: "gemini-pro" }
  );
  const seen = new Set<string>();
  const candidates = preferred.filter((c) => {
    const key = `${c.v}|${c.m}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  let lastErr: any = null;
  for (const c of candidates) {
    try {
      const model = new ChatGoogleGenerativeAI({ apiKey, model: c.m, apiVersion: c.v, temperature: 0.2 });
      // Use invoke with a plain string to avoid unsupported message formats
      const msg: any = await model.invoke(fullPrompt as any);
      const content = (msg && (msg.content ?? msg.text)) as any;
      let answer = "";
      if (typeof content === "string") {
        answer = content;
      } else if (Array.isArray(content)) {
        // Gemini may return array parts; join any text fields
        answer = content
          .map((p) =>
            typeof p === "string"
              ? p
              : typeof p?.text === "string"
              ? p.text
              : typeof p?.content === "string"
              ? p.content
              : ""
          )
          .join("")
          .trim();
      } else if (typeof msg === "string") {
        answer = msg;
      }
      return { answer };
    } catch (e) {
      lastErr = e;
    }
  }
  throw new Error(
    `Failed to call Gemini. Tried: ${candidates.map((c) => `${c.v}:${c.m}`).join(", ")} Last error: ${
      (lastErr as any)?.message || lastErr
    }`
  );
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));
    const question: string = body?.question ?? body?.message ?? "";
    if (!question || typeof question !== "string") {
      return NextResponse.json({ error: "Missing 'question' in JSON body" }, { status: 400 });
    }

    // Greetings
    const qNorm = question.trim().toLowerCase();
    const greetRe = /^(hi|hello|hey|hlo|hola|namaste|good\s*(morning|afternoon|evening)|yo|sup)[!.?,\s]*$/i;
    if (greetRe.test(qNorm)) {
      const greeting =
        "Hello! I'm the Virtual Lab IIT Roorkee assistant. Ask about objectives, apparatus, procedure, precautions, or analysis, and I'll help you.";
      return NextResponse.json({ answer: greeting, sources: [] });
    }

    // Retrieve Astra context
    const { context, sources, topSimilarity, topTextSimilarity, images = [] } = await retrieveContext(question);

    // Web search policy
    const forceWeb = /\b(more\s+details?|from\s+web|explanation\s+from\s+web)\b/i.test(question);
    const sim = typeof topTextSimilarity === "number" ? topTextSimilarity : (typeof topSimilarity === 'number' ? topSimilarity : null);
    const thresholdEnv = parseFloat(String(process.env.SEARCH_SIM_THRESHOLD ?? '0.6'));
    const threshold = Number.isFinite(thresholdEnv) ? thresholdEnv : 0.6;
    // Heuristic: if the user asks for a specific resistor color code (digits + 'color code'),
    // and the current text context doesn't include those digits or any color/band details, force web search.
    const wantsColorCode = /(color\s*cod(?:e|ing)|band\s*colou?r)/i.test(question);
    const qDigits = Array.from(question.matchAll(/\d+/g)).map((m) => m[0]);
    const digitsInContext = qDigits.some((d) => new RegExp(`\b${d}\b`).test(context));
    const hasColorTermsInContext = /(color|colour|band)/i.test(context);
    const insufficientMapping = wantsColorCode && (!digitsInContext || !hasColorTermsInContext);
    const shouldSearchWeb = forceWeb || insufficientMapping || (sim != null && sim < threshold);

    let webBlob: string | undefined;
    let webCitations: Array<{ title: string; url: string }> = [];
    if (shouldSearchWeb) {
      // Build a focused query for color-code questions
      const wantsColorCode = /(color\s*cod(?:e|ing)|band\s*colou?r)/i.test(question);
      const qVals = extractOhmValues(question);
      const colorQ = wantsColorCode && qVals.length
        ? `resistor color code ${qVals.map(v=>Math.round(v)).join(' ')} ohm 4-band 5-band`
        : question;
      const web = await tavilyEduSearch(colorQ);
      webBlob = web.webBlob || undefined;
      webCitations = web.citations || [];
    }

    // If user asks for an image, return matched images directly when available
    const asksImage = /(image|photo|figure|diagram|symbol|pic|picture)/i.test(question);
    const toClientPath = (p: string) => {
      const norm = (p || '').replace(/\\/g, '/');
      const i = norm.lastIndexOf('/images/');
      if (i >= 0) return norm.slice(i + 1);
      const file = norm.split('/').pop() || norm;
      return `images/${file}`;
    };
    if (asksImage && images.length) {
      const qtext = question.toLowerCase();
      const preferred = images.filter((p) => {
        const s = p.toLowerCase();
        return (
          (qtext.includes('resistor') && /res(istor)?/.test(s)) ||
          (qtext.includes('capacitor') && /cap(acitor)?/.test(s)) ||
          (qtext.includes('inductor') && /induct/.test(s)) ||
          (qtext.includes('color') || qtext.includes('colour')) && /color|colour|code|guide|chart/.test(s)
        );
      });
      const picks = (preferred.length ? preferred : images).slice(0, 6).map(toClientPath);
      const md = [
        'Here are relevant images:',
        '',
        ...picks.map((r) => `![](/${r})`),
      ].join('\n');
      return NextResponse.json({ answer: md, sources: picks.map((p) => `/${p}`), similarity: sim });
    }

    // If both sources are empty/insufficient, compute color codes if applicable; else fallback
    if (!(context && context.trim()) && !(webBlob && webBlob.trim())) {
      const wantColor = /(resistor)?.*(color\s*cod(?:e|ing)|band\s*colou?r)/i.test(question);
      const qVals = wantColor ? extractOhmValues(question) : [];
      if (qVals.length) {
        const parts: string[] = [];
        for (const val of qVals) {
          const mapping = computeResistorColorCode(val);
          if (mapping) {
            parts.push(
              `- ${Math.round(val)} Ω:\n  - 4‑band: ${mapping.fourBand.join(' - ')}\n  - 5‑band: ${mapping.fiveBand.join(' - ')}`
            );
          }
        }
        if (parts.length) {
          const md = `Resistor color code:\n\n${parts.join('\n')}`;
          return NextResponse.json({ answer: md, sources: [], similarity: sim });
        }
      }
      const fallback = "I don't know.";
      return NextResponse.json({ answer: fallback, sources: [], similarity: sim });
    }

    // Generate final (graceful fallback if model fails)
    let finalAnswer: string;
    try {
      const { answer } = await summarizeAndAnswer(question, context, webBlob);
      finalAnswer = (answer || "").trim();
    } catch {
      finalAnswer = context?.trim()
        ? (context.length > 1200 ? context.slice(0, 1200) + " ..." : context)
        : "I don't know.";
    }
    // If model yielded nothing useful and this is a color‑code query (possibly multiple values), compute deterministically
    const wantColor = /(resistor)?.*(color\s*cod(?:e|ing)|band\s*colou?r)/i.test(question);
    const looksUnknown = (t: string) => /i\s*don'?t\s*know|not\s*(available|present)|no\s*information/i.test((t||"").toLowerCase());
    if (wantColor && (!finalAnswer || looksUnknown(finalAnswer))) {
      const qVals = extractOhmValues(question);
      if (qVals.length) {
        const parts: string[] = [];
        for (const val of qVals) {
          const mapping = computeResistorColorCode(val);
          if (mapping) {
            parts.push(
              `- ${Math.round(val)} Ω:\n  - 4‑band: ${mapping.fourBand.join(' - ')}\n  - 5‑band: ${mapping.fiveBand.join(' - ')}`
            );
          }
        }
        if (parts.length) {
          finalAnswer = `Resistor color code:\n\n${parts.join('\n')}`;
        }
      }
    }
    if (!finalAnswer) finalAnswer = "I don't know.";
    const mergedSources = [...(sources || []), ...webCitations.map((c) => `${c.title} - ${c.url}`)];
    return NextResponse.json({ answer: finalAnswer, sources: mergedSources, similarity: sim });
  } catch (err: any) {
    return NextResponse.json({ error: err?.message ?? "Unexpected error" }, { status: 500 });
  }
}

export const runtime = "nodejs";


  