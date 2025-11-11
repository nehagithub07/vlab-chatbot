import fs from "fs";
import path from "path";
import { config as loadEnv } from "dotenv";
import { DataAPIClient } from "@datastax/astra-db-ts";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import crypto from "crypto";
import mammoth from "mammoth";

// Load environment variables from a .env file
(() => {
  const candidates = [
    path.resolve(process.cwd(), "vlab-chatbot/.env"),
    path.resolve(process.cwd(), ".env"),
    path.resolve(__dirname, "../.env"),
  ];
  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) {
        loadEnv({ path: p });
        break;
      }
    } catch {}
  }
})();

const {
  ASTRA_DB_API_ENDPOINT,
  ASTRA_DB_APPLICATION_TOKEN,
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  GEMINI_API_KEY,
} = process.env as Record<string, string>;

// Choose embedding provider
let xenovaExtractor: any = null as any;
type Embedder = (text: string) => Promise<number[]>;
const EMBED_PROVIDER = (process.env.EMBED_PROVIDER || "google").toLowerCase();
const isMultimodal =
  EMBED_PROVIDER === "multimodal" ||
  EMBED_PROVIDER === "google-mm" ||
  EMBED_PROVIDER === "mm" ||
  /multimodal/.test(String((process.env as any).EMBED_MODEL || ""));
const useGoogle = !isMultimodal && EMBED_PROVIDER === "google" && !!GEMINI_API_KEY;
const collectionName = `${ASTRA_DB_COLLECTION}${isMultimodal ? "_mm" : useGoogle ? "" : "_xv"}`;
const embedDimension = isMultimodal ? 1408 : useGoogle ? 768 : 384;

let embed: Embedder;
if (isMultimodal) {
  if (!GEMINI_API_KEY) throw new Error("GEMINI_API_KEY required for multimodalembedding@001");
  const MM_URL = "https://generativelanguage.googleapis.com/v1beta/models/multimodalembedding@001:embedContent";
  embed = async (text: string) => {
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
} else if (useGoogle) {
  const gemini = new GoogleGenerativeAIEmbeddings({ apiKey: GEMINI_API_KEY, model: "embedding-001" });
  embed = async (text: string) => {
    const out: any = await gemini.embedQuery(text);
    if (Array.isArray(out) && typeof out[0] === "number") return out as number[];
    const vec = out?.data?.[0]?.embedding as number[] | undefined;
    if (!vec) throw new Error("Failed to get embedding vector from Google API response");
    return vec;
  };
} else {
  embed = async (text: string) => {
    if (!xenovaExtractor) {
      const { pipeline } = await import("@xenova/transformers");
      xenovaExtractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    }
    const output: any = await xenovaExtractor(text, { pooling: "mean", normalize: true });
    const data: Float32Array = output?.data ?? output?.tolist?.();
    if (!data) throw new Error("Failed to compute local embedding");
    return Array.from(data);
  };
}

// Astra client setup
const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db = client.db(ASTRA_DB_API_ENDPOINT, { keyspace: ASTRA_DB_NAMESPACE });

// Text splitter setup
const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });

// Helpers
const normalize = (s: string) => s.replace(/[\r\t ]+/g, " ").replace(/\s*\n\s*/g, "\n").trim();
const hashId = (prefix: string, s: string) => `${prefix}_${crypto.createHash("sha1").update(s).digest("hex")}`;
const imgMime = (file: string): string | null => {
  const ext = path.extname(file).toLowerCase();
  if (ext === ".png") return "image/png";
  if (ext === ".jpg" || ext === ".jpeg") return "image/jpeg";
  if (ext === ".gif") return "image/gif";
  if (ext === ".webp") return "image/webp";
  if (ext === ".bmp") return "image/bmp";
  if (ext === ".tiff" || ext === ".tif") return "image/tiff";
  if (ext === ".svg") return "image/svg+xml";
  return null;
};
const embedImageMM = async (filePath: string, mime: string): Promise<number[]> => {
  if (!GEMINI_API_KEY) throw new Error("GEMINI_API_KEY required for multimodalembedding@001");
  const MM_URL = "https://generativelanguage.googleapis.com/v1beta/models/multimodalembedding@001:embedContent";
  const dataB64 = fs.readFileSync(filePath).toString("base64");
  const res = await fetch(MM_URL + `?key=${encodeURIComponent(GEMINI_API_KEY)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content: { parts: [{ inline_data: { mime_type: mime, data: dataB64 } }] } }),
  } as any);
  if (!res.ok) throw new Error(`Multimodal image embed failed: ${res.status}`);
  const data: any = await res.json();
  const values: number[] | undefined = data?.embedding?.values;
  if (!Array.isArray(values)) throw new Error("No embedding.values returned for image");
  return values;
};

// Read DOCX (or plain text) file
const readDocxFile = async (filePath: string): Promise<string> => {
  const header = Buffer.alloc(4);
  const fd = fs.openSync(filePath, "r");
  try {
    fs.readSync(fd, header, 0, 4, 0);
  } finally {
    fs.closeSync(fd);
  }
  const isZip = header[0] === 0x50 && header[1] === 0x4b; // 'PK'
  if (isZip) {
    const { value } = await mammoth.extractRawText({ path: filePath });
    return value;
  }
  // Fallback: treat as plain text
  return fs.readFileSync(filePath, "utf8");
};

// Create collection in Astra DB
const createCollection = async (): Promise<void> => {
  try {
    const res = await db.createCollection(collectionName, {
      vector: { dimension: embedDimension, metric: "cosine" },
    });
    console.log("Collection created:", res);
  } catch (err: any) {
    console.log("Collection may already exist:", err?.message || String(err));
  }
};

// Load and store experiment data
const loadAndStoreEmbeddings = async (): Promise<void> => {
  const collection = db.collection(collectionName);
  // Resolve docx path robustly across different working directories
  const docCandidates = [
    path.resolve(process.cwd(), "vlab-chatbot/Experiment-docs.docx"),
    path.resolve(__dirname, "../Experiment-docs.docx"),
  ];
  const docPath = docCandidates.find((p) => fs.existsSync(p));
  if (!docPath) {
    throw new Error(
      "Experiment-docs.docx not found. Place it in vlab-chatbot/ or update the path in scripts/loadDB.ts."
    );
  }
  const docText = await readDocxFile(docPath);

  const chunks = await splitter.splitText(docText);
  console.log(`Split into ${chunks.length} chunks.`);

  let addedText = 0, updatedText = 0, skippedText = 0;
  for (const raw of chunks) {
    const chunk = normalize(raw);
    if (!chunk || chunk.length < 5) { skippedText++; continue; }
    const uid = hashId("txt", chunk);
    const vector = await embed(chunk);
    const doc: any = { uid, type: "text", text: chunk, model: isMultimodal ? "multimodalembedding@001" : useGoogle ? "embedding-001" : "MiniLM-L6-v2", $vector: vector };
    try {
      await collection.insertOne(doc, { documentId: uid } as any);
      addedText++;
    } catch (e: any) {
      try {
        await collection.replaceOne({ uid }, doc, { upsert: true } as any);
        updatedText++;
      } catch {
        skippedText++;
      }
    }
  }

  // Images (only if multimodal enabled)
  let addedImg = 0, updatedImg = 0, skippedImg = 0;
  if (isMultimodal) {
    const imgDirCandidates = [
      path.resolve(process.cwd(), "vlab-chatbot/images"),
      path.resolve(__dirname, "../images"),
    ];
    const imgDir = imgDirCandidates.find((p) => fs.existsSync(p) && fs.statSync(p).isDirectory());
    if (imgDir) {
      for (const name of fs.readdirSync(imgDir)) {
        const full = path.join(imgDir, name);
        const mime = imgMime(full);
        if (!mime) continue;
        try {
          const vec = await embedImageMM(full, mime);
          const rel = path.relative(process.cwd(), full).replace(/\\/g, "/");
          const uid = hashId("img", `${rel}:${fs.statSync(full).size}`);
          const doc: any = { uid, type: "image", path: rel, mime, model: "multimodalembedding@001", $vector: vec };
          try {
            await collection.insertOne(doc, { documentId: uid } as any);
            addedImg++;
          } catch {
            await collection.replaceOne({ uid }, doc, { upsert: true } as any);
            updatedImg++;
          }
        } catch {
          skippedImg++;
        }
      }
    }
  }

  console.log(`Text added: ${addedText}, updated: ${updatedText}, skipped: ${skippedText}`);
  if (isMultimodal) console.log(`Images added: ${addedImg}, updated: ${updatedImg}, skipped: ${skippedImg}`);
  console.log("All embeddings processed.");
};

// Run
(async () => {
  await createCollection();
  await loadAndStoreEmbeddings();
})();

