import fs from "fs";
import path from "path";
import { config as loadEnv } from "dotenv";
import { DataAPIClient } from "@datastax/astra-db-ts";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
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
const useGoogle = EMBED_PROVIDER === "google" && !!GEMINI_API_KEY;
const collectionName = `${ASTRA_DB_COLLECTION}${useGoogle ? "" : "_xv"}`;
const embedDimension = useGoogle ? 768 : 384;

let embed: Embedder;
if (useGoogle) {
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
    console.log("✅ Collection created:", res);
  } catch (err: any) {
    console.log("⚠️ Collection may already exist:", err.message);
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
  console.log(`📘 Split into ${chunks.length} chunks.`);

  for (const chunk of chunks) {
    const vector = await embed(chunk);
    await collection.insertOne({ text: chunk, $vector: vector });
  }

  console.log("✅ All embeddings inserted successfully!");
};

// Run
(async () => {
  await createCollection();
  await loadAndStoreEmbeddings();
})();
