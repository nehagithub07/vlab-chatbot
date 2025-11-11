"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Message = { role: "user" | "assistant"; content: string; sources?: string[] };

const SUGGESTED: string[] = [
  "What is the objectives of this experiment?",
  "List the apparatus and the setup steps.",
];

export default function Home() {
  const normalizeForImages = (text: string): string => {
    if (!text) return "";
    // Convert labeled references like "Symbol: images/Capacitor.png." into an embedded image
    let out = text.replace(
      /\b(Photo|Symbol|Image|Figure|Pic|Picture)\s*:\s*(\/?images\/[\w\-./%]+?)(?=[\s)\]\}.,!?;:]|$)/gi,
      (_m, lbl, pth) => {
        const clean = String(pth).startsWith("/") ? String(pth) : "/" + String(pth);
        return `${lbl}: ![](${clean})`;
      }
    );
    // Convert bare tokens like "... in images/Capacitor.png." to embedded images
    out = out.replace(
      /(^|[\s(])(\/?images\/[\w\-./%]+?)(?=[\s)\]\}.,!?;:]|$)/g,
      (_m, lead, pth) => `${lead}![](${pth.startsWith("/") ? pth : "/" + pth})`
    );
    return out;
  };
  const [messages, setMessages] = React.useState<Message[]>([]);
  const [input, setInput] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const listRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages, loading]);

  const resetChat = () => setMessages([]);

  const ask = async (question: string) => {
    if (!question.trim()) return;
    setLoading(true);
    setMessages((m) => [...m, { role: "user", content: question }]);
    setInput("");
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const json = await res.json();
      if (!res.ok) {
        const msg = json?.error || `Error ${res.status}`;
        setMessages((m) => [...m, { role: "assistant", content: msg }]);
        return;
      }
      const text = json?.answer || "No answer returned.";
      const sources = Array.isArray(json?.sources) ? json.sources : [];
      setMessages((m) => [...m, { role: "assistant", content: text, sources }]);
    } catch (e: any) {
      setMessages((m) => [...m, { role: "assistant", content: `Error: ${e?.message ?? "request failed"}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="mx-auto flex min-h-screen max-w-7xl flex-col gap-0 bg-gray-50">
      {/* Header matching vlab.co.in */}
      <header style={{backgroundColor: '#02263C'}} className="shadow-md border-b-4 border-cyan-900">
        <div className="px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {/* Virtual Labs Logo */}
              <div className="flex items-center gap-4">
                <div className="h-14 w-14 rounded-lg bg-white shadow-lg p-1 flex items-center justify-center">
                  <img 
                    src="https://www.vlab.co.in/images/logo.jpg" 
                    alt="Virtual Labs Logo" 
                    className="h-full w-full object-contain"
                  />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-white tracking-wide">Virtual Lab Assistant</h1>
                  <p className="text-xs text-cyan-100 mt-0.5">IIT Roorkee | Ministry of Education Initiative</p>
                </div>
              </div>
            </div>
            <button
              onClick={resetChat}
              className="flex items-center gap-2 rounded-md bg-white hover:bg-cyan-50 px-4 py-2 text-sm font-semibold shadow-md transition-all duration-200 border border-cyan-200"
              style={{color: '#02263C'}}
            >
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              New Chat
            </button>
          </div>
        </div>
      </header>

      {/* Info banner */}
      <div className="bg-cyan-50 border-b border-cyan-200 px-8 py-3">
        <p className="text-xs font-medium text-center" style={{color: '#02263C'}}>
          An Initiative of Ministry of Education | Under the National Mission on Education through ICT
        </p>
      </div>

      {/* Instructions section */}
      <div className="bg-white border-b border-gray-200 px-8 py-5 shadow-sm">
        <p className="text-sm text-gray-700 font-medium mb-3">
          Ask questions about objectives, apparatus, procedures, or analysis — grounded in lab materials.
        </p>
        <div className="flex flex-wrap gap-2">
          {SUGGESTED.map((q) => (
            <button
              key={q}
              onClick={() => ask(q)}
              className="rounded-md bg-cyan-50 hover:bg-cyan-100 border border-cyan-200 px-3 py-2 text-xs font-medium transition-all duration-200 hover:shadow-sm"
              style={{color: '#02263C'}}
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Chat Section */}
      <section className="flex-1 p-8">
        <div className="h-[calc(100vh-320px)] overflow-hidden rounded-lg border border-gray-300 bg-white shadow-lg">
          <div ref={listRef} className="h-full space-y-4 overflow-y-auto p-6 scroll-smooth">
            {messages.length === 0 && (
              <div className="grid h-full place-items-center">
                <div className="text-center space-y-4 max-w-md">
                  <div className="mx-auto h-20 w-20 rounded-full bg-cyan-100 flex items-center justify-center shadow-md">
                    <svg className="h-10 w-10" style={{color: '#02263C'}} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800">Welcome to Virtual Lab Assistant</h3>
                    <p className="text-sm text-gray-600 mt-2">Select a question above or type your own to begin your learning journey</p>
                  </div>
                </div>
              </div>
            )}
            {messages.map((m, idx) => (
              <div 
                key={idx} 
                className={`flex items-start gap-3 animate-in fade-in slide-in-from-bottom-2 duration-200 ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {m.role === "assistant" && (
                  <div className="grid h-9 w-9 shrink-0 place-items-center rounded-full text-xs font-bold text-white shadow-md" style={{backgroundColor: '#02263C'}}>
                    VL
                  </div>
                )}
                <div
                  className={
                    "max-w-[78%] rounded-lg px-4 py-3 text-sm shadow-sm transition-all duration-200 " +
                    (m.role === "user"
                      ? "text-white"
                      : "bg-gray-50 text-gray-800 border border-gray-200")
                  }
                  style={m.role === "user" ? {backgroundColor: '#02263C'} : {}}
                >
                  {m.role === "assistant" ? (
                    <>
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          img: (props) => (
                            <img {...props} style={{maxWidth: '100%', height: 'auto', borderRadius: 6}} />
                          ),
                          a: (props) => (
                            <a {...props} target="_blank" rel="noreferrer" className="underline" />
                          )
                        }}
                      >
                        {normalizeForImages(m.content)}
                      </ReactMarkdown>
                      {Array.isArray(m.sources) && m.sources.some((s) => /(^|\/)images\//.test(String(s))) && (
                        <div className="mt-2 grid grid-cols-2 gap-2">
                          {m.sources
                            .filter((s) => /(^|\/)images\//.test(String(s)))
                            .slice(0, 6)
                            .map((s, i) => {
                              const src = String(s).startsWith('/') ? String(s) : `/${String(s)}`;
                              return (
                                <img key={i} src={src} alt="" className="rounded border border-gray-200" style={{maxWidth: '100%', height: 'auto'}} />
                              );
                            })}
                        </div>
                      )}
                    </>
                  ) : (
                    <span className="whitespace-pre-wrap">{m.content}</span>
                  )}
                </div>
                {m.role === "user" && (
                  <div className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-gray-600 text-xs font-bold text-white shadow-md">
                    You
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div className="flex items-start gap-3 animate-in fade-in slide-in-from-bottom-2 duration-200">
                <div className="grid h-9 w-9 shrink-0 place-items-center rounded-full text-xs font-bold text-white shadow-md" style={{backgroundColor: '#02263C'}}>
                  VL
                </div>
                <div className="max-w-[78%] rounded-lg bg-gray-50 border border-gray-200 px-4 py-3 text-sm text-gray-700 shadow-sm">
                  <span className="inline-flex items-center gap-1.5">
                    Processing
                    <span className="inline-block h-1.5 w-1.5 animate-bounce rounded-full [animation-delay:-0.3s]" style={{backgroundColor: '#02263C'}}></span>
                    <span className="inline-block h-1.5 w-1.5 animate-bounce rounded-full [animation-delay:-0.15s]" style={{backgroundColor: '#02263C'}}></span>
                    <span className="inline-block h-1.5 w-1.5 animate-bounce rounded-full" style={{backgroundColor: '#02263C'}}></span>
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Input Section */}
        <div className="mt-4 flex items-center gap-3 rounded-lg border border-gray-300 bg-white p-2 shadow-md transition-all duration-200 focus-within:ring-2 focus-within:border-cyan-500" style={{'--tw-ring-color': '#02263C'} as any}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                ask(input);
              }
            }}
            placeholder="Type your lab question here..."
            className="w-full rounded-md border-0 bg-transparent px-4 py-2.5 text-sm text-gray-800 placeholder:text-gray-500 focus:outline-none focus:ring-0"
          />
          <button
            onClick={() => ask(input)}
            disabled={loading || !input.trim()}
            className="grid h-10 w-10 shrink-0 place-items-center rounded-md text-white shadow-sm transition-all duration-200 hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
            style={{backgroundColor: '#02263C'}}
            title="Send message"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="h-5 w-5">
              <path d="M2.01 21 23 12 2.01 3 2 10l15 2-15 2z" />
            </svg>
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-8 py-3 text-center border-t-4 border-cyan-900" style={{backgroundColor: '#02263C'}}>
        <p className="text-xs text-cyan-100">
          Â© 2025 Virtual Labs | National Mission on Education through ICT | Ministry of Education, Government of India
        </p>
      </footer>
    </main>
  );
}

