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

  const provider = EMBED_PROVIDER.toLowerCase();
  if (provider === "xenova") {
    // Local embeddings via Transformers.js (no API key required)
    const { pipeline } = await import("@xenova/transformers");
    const extractor: any = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    const embed: Embedder = async (text: string) => {
      const output: any = await extractor(text, { pooling: "mean", normalize: true });
      const data: Float32Array = output?.data ?? output?.tolist?.();
      return Array.from(data);
    };
    return { embed, dim: 384, collectionName: `${ASTRA_DB_COLLECTION}_xv` };
  }

  // Default to Google Gemini embeddings
  if (!GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY is required for google embeddings. Set EMBED_PROVIDER=xenova to use local embeddings.");
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

  // Heuristic: detect focused requests (procedure / apparatus / objectives / theory)
  const focusRegex = /(procedure|precaution|apparatus|equipment|objective|aim|theory|definition|principle)s?/i;
  const hasFocusKeyword = focusRegex.test(query);
  // Augment query for better recall
  const augTerms: string[] = ["experiment", "lab"];
  if (/procedure/i.test(query)) augTerms.push("procedure", "steps", "step 1", "step 2", "click", "drag");
  if (/precaution/i.test(query)) augTerms.push("precaution", "safety", "warning");
  if (/(apparatus|equipment)/i.test(query)) augTerms.push("apparatus", "equipment", "setup");
  if (/(objective|aim)/i.test(query)) augTerms.push("objective", "aim");
  if (/(analy[sz]e|analysis|calculation|result)/i.test(query)) augTerms.push("analysis", "calculate", "results", "parameters");
  if (/(theory|definition|principle)/i.test(query)) augTerms.push("theory", "definition", "principle");
  const augmentedQuery = `${query} ${augTerms.join(" ")}`.trim();
  const vector = await embed(augmentedQuery);

  // Vector/hybrid similarity search (get a larger pool when focused)
  let docs: any[] = [];
  try {
    const anyColl: any = coll as any;
    const cur = anyColl.findAndRerank?.({});
    if (cur && typeof cur.sort === "function") {
      const sorted = cur.sort({ $hybrid: { $vector: vector, $lexical: augmentedQuery } });
      const limited = typeof sorted.limit === "function" ? sorted.limit(hasFocusKeyword ? 20 : 8) : sorted;
      docs = await limited.toArray();
    }
  } catch {}
  if (!Array.isArray(docs) || docs.length === 0) {
    const cursor = coll
      .find({}, { limit: hasFocusKeyword ? 20 : 8 })
      .includeSimilarity?.(true)
      .sort({ $vector: vector });
    docs = await cursor.toArray();
  }
  const seen = new Set<string>();
  const rawTexts: string[] = (docs as any[])
    .map((d) => (typeof d?.text === "string" ? d.text : JSON.stringify(d)))
    .filter(Boolean) as string[];

  // Prioritize passages that include the focus keyword while preserving original order
  const prioritized: string[] = [];
  const tail: string[] = [];
  for (const s of rawTexts) {
    const t = s.trim();
    if (!t) continue;
    if (seen.has(t)) continue;
    seen.add(t);
    if (hasFocusKeyword && focusRegex.test(t)) prioritized.push(t);
    else tail.push(t);
  }
  const ordered = hasFocusKeyword ? prioritized.concat(tail) : tail.length ? rawTexts : rawTexts;
  const snippets = ordered.slice(0, 5);
  const context = snippets.join("\n\n---\n\n");
  const foundFocused = hasFocusKeyword && prioritized.length > 0;
  const isSparse = !foundFocused && context.length < 200; // prefer context when we found focused passages
  return { context, sources: snippets, isSparse };
}

async function generateAnswer(question: string, context: string, allowGeneral = false) {
  const { GEMINI_API_KEY, GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_API_VERSION } =
    process.env as Record<string, string>;
  const apiKey = GOOGLE_API_KEY || GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      "Missing Google API key. Set GOOGLE_API_KEY (preferred) or GEMINI_API_KEY in your environment."
    );
  }

  const isTheory = /(theory|definition|principle)/i.test(question);
  const instructionOnlyContext = `You are the Virtual Lab IIT Roorkee assistant.
Use only the provided context from the lab materials. If the information is not present, say you don't know.
- Respond strictly to the user's question, point-to-point.
- No headings, no preamble, no notes, no sources, no extra text.
- If listing items, use brief bullet points without quotes.`;
  const instructionWithFallback = `You are the Virtual Lab IIT Roorkee assistant.
Prefer using the provided lab context. If the context is insufficient or missing, still answer concisely using your general domain knowledge of electrical machines and standard lab practice.
- Do not say that the context is missing; do not apologize. Provide the best concise answer instead.
- Respond strictly to the user's question, point-to-point.
- No headings, no preamble, no notes, no sources, no extra text.
- If listing items, use brief bullet points without quotes.`;
  const instructionTheoryOnly = `You are the Virtual Lab IIT Roorkee assistant.
Derive a compact theory note from the context. For each relevant concept (e.g., Resistor, Capacitor, Inductor, Transformer, etc.):
- Render an H3 heading like "### Resistor".
- Under it, write 2–3 short sentences describing the concept and its role in this experiment.
- Use only the provided context; if a concept isn't present, skip it.`;

  const instructionTheoryWithFallback = `You are the Virtual Lab IIT Roorkee assistant.
Compose a compact theory note. For each relevant concept (e.g., Resistor, Capacitor, Inductor, Transformer, etc.):
- Render an H3 heading like "### Resistor".
- Under it, write 2–3 short sentences describing the concept and its role in this experiment.
- Prefer the lab context, but if a concept is missing, use standard domain knowledge to fill in with 2–3 accurate sentences.
- Do not add sources or meta commentary.`;

  const instruction = isTheory
    ? (allowGeneral ? instructionTheoryWithFallback : instructionTheoryOnly)
    : (allowGeneral ? instructionWithFallback : instructionOnlyContext);

  const fullPrompt = `${instruction}\n\nContext (may be empty):\n${context}\n\nQuestion: ${question}\n\nAnswer in Markdown:`;

  // Build a list of (apiVersion, model) candidates to try
  const preferred: Array<{ v: string; m: string }> = [];
  if (GEMINI_MODEL && GEMINI_API_VERSION) preferred.push({ v: GEMINI_API_VERSION, m: GEMINI_MODEL });
  if (GEMINI_MODEL) preferred.push({ v: "v1", m: GEMINI_MODEL }, { v: "v1beta", m: GEMINI_MODEL });
  preferred.push(
    // Common v1 options
    { v: "v1", m: "gemini-1.5-flash" },
    { v: "v1", m: "gemini-1.5-flash-8b" },
    { v: "v1", m: "gemini-1.5-pro" },
    // v1beta options (often more permissive)
    { v: "v1beta", m: "gemini-2.0-flash" },
    { v: "v1beta", m: "gemini-1.5-flash" },
    { v: "v1beta", m: "gemini-pro" }
  );
  // Deduplicate while preserving order
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
      const model = new ChatGoogleGenerativeAI({
        apiKey,
        model: c.m,
        apiVersion: c.v,
        temperature: 0.2,
      });
      const res = await model.invoke(fullPrompt);
      const answer =
        typeof (res as any)?.content === "string"
          ? (res as any).content
          : (res as any)?.lc_kwargs?.content ?? String(res);
      return { answer };
    } catch (e: any) {
      lastErr = e;
      // Try next candidate
    }
  }

  throw new Error(
    `Failed to call Gemini. Tried: ${candidates
      .map((c) => `${c.v}:${c.m}`)
      .join(", ")}. Last error: ${lastErr?.message || lastErr}`
  );
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));
    const question: string = body?.question ?? body?.message ?? "";
    if (!question || typeof question !== "string") {
      return NextResponse.json({ error: "Missing 'question' in JSON body" }, { status: 400 });
    }

    // Friendly greeting handler for salutations like hi/hello
    const qNorm = question.trim().toLowerCase();
    const greetRe = /^(hi|hello|hey|hlo|hola|namaste|good\s*(morning|afternoon|evening)|yo|sup)[!.?,\s]*$/i;
    if (greetRe.test(qNorm)) {
      const greeting =
        "Hello! I’m the Virtual Lab IIT Roorkee assistant. Ask about objectives, apparatus, procedure, precautions, or analysis, and I’ll help you.";
      return NextResponse.json({ answer: greeting, sources: [] });
    }

    const { context, sources, isSparse } = await retrieveContext(question);

    // Allow general fallback by env (default true)
    const allowGeneralEnv = String(process.env.ALLOW_GENERAL_FALLBACK ?? 'true').toLowerCase();
    const allowGeneralDefault = ["1", "true", "yes", "on"].includes(allowGeneralEnv);

    const looksUnknown = (text: string) => {
      const s = (text || "").toLowerCase();
      const patterns = [
        "i don't know",
        "i do not know",
        "not in the context",
        "not present in the context",
        "not available in the context",
        "not mentioned in the context",
        "context does not include",
        "cannot find",
        "can't find",
        "insufficient context",
        "no information",
        "i'm sorry",
        "sorry,",
        "provided context does not include",
        "i don't have the information",
        "i do not have the information",
      ];
      return patterns.some((p) => s.includes(p));
    };

    // First pass: allow general if env says so or retrieval is sparse
    let { answer } = await generateAnswer(question, context, allowGeneralDefault || isSparse);

    // If strict response refuses due to lack of context, retry with general knowledge
    if (looksUnknown(answer)) {
      const retry = await generateAnswer(question, context, true);
      answer = retry.answer;
    }

    return NextResponse.json({ answer, sources });
  } catch (err: any) {
    return NextResponse.json({ error: err?.message ?? "Unexpected error" }, { status: 500 });
  }
}

export const runtime = "nodejs";

