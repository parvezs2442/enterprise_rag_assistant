import React, { useState, useEffect, useRef } from 'react';
import {
  MessageSquare,
  Scale,
  BarChart3,
  ShieldCheck,
  FileText,
  Send,
  Sparkles,
  Layers,
  ChevronDown,
  ChevronUp,
  Cpu,
  Database,
  CheckCircle2,
  RefreshCw,
  Upload,
  BookOpen,
  ArrowRight,
  ShieldAlert,
  Sliders
} from 'lucide-react';

interface ChunkItem {
  content: string;
  source: string;
  page: number;
  rerank_score?: number | null;
  chunk_id?: number | null;
}

interface CitationInfo {
  cited_tags: string[];
  valid_citations: string[];
  hallucinated_citations: string[];
  citation_precision: number;
  is_citation_valid: boolean;
  metadata_map: Record<string, { source: string; page: number; score: any }>;
}

interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  retrieved_chunks?: ChunkItem[];
  citation_info?: CitationInfo;
  latency_sec?: number;
  timestamp: string;
}

interface DocumentItem {
  filename: string;
  size_kb: number;
  modified: string;
}

interface EvalReport {
  timestamp: string;
  model_evaluated: string;
  metrics_summary: {
    baseline: {
      hit_rate_at_4: number;
      mrr: number;
      avg_faithfulness: number;
      avg_relevance: number;
      abstention_accuracy: number;
      avg_latency_sec: number;
    };
    tuned: {
      hit_rate_at_4: number;
      mrr: number;
      avg_faithfulness: number;
      avg_relevance: number;
      abstention_accuracy: number;
      avg_latency_sec: number;
    };
  };
  test_case_results?: {
    baseline: any[];
    tuned: any[];
  };
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'compare' | 'eval' | 'security' | 'docs'>('chat');

  // Chat State
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      sender: 'assistant',
      text: 'Hello! I am your Enterprise RAG Assistant. I am grounded exclusively on your ingested documents with hybrid retrieval (Dense + BM25), Reciprocal Rank Fusion, and FlashRank reranking. Ask me any question!',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [inputQuery, setInputQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [useRewrite, setUseRewrite] = useState(false);
  const [useHyde, setUseHyde] = useState(false);
  const [useReranker, setUseReranker] = useState(true);
  const [selectedSource, setSelectedSource] = useState<string>('');
  const [expandedChunks, setExpandedChunks] = useState<Record<string, boolean>>({});

  // Documents & Health State
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [healthStatus, setHealthStatus] = useState<any>(null);

  // Compare State
  const [compareQuery, setCompareQuery] = useState('What is RAG and why is it used?');
  const [compareResults, setCompareResults] = useState<any>(null);
  const [compareLoading, setCompareLoading] = useState(false);

  // Eval State
  const [evalReport, setEvalReport] = useState<EvalReport | null>(null);
  const [evalLoading, setEvalLoading] = useState(false);

  // Security Test State
  const [securityResult, setSecurityResult] = useState<any>(null);
  const [securityLoading, setSecurityLoading] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchHealth();
    fetchDocuments();
    fetchEvaluationReport();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchHealth = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) setHealthStatus(await res.json());
    } catch (e) {
      console.error('Health check failed', e);
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch('/api/documents');
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
      }
    } catch (e) {
      console.error('Fetch docs failed', e);
    }
  };

  const fetchEvaluationReport = async () => {
    try {
      const res = await fetch('/api/evaluation');
      if (res.ok) setEvalReport(await res.json());
    } catch (e) {
      console.log('No existing eval report');
    }
  };

  // Submit Question to Chat
  const handleSendMessage = async (queryText?: string) => {
    const q = queryText || inputQuery;
    if (!q.trim() || loading) return;

    const userMsg: ChatMessage = {
      id: String(Date.now()),
      sender: 'user',
      text: q,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMsg]);
    if (!queryText) setInputQuery('');
    setLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          source_filter: selectedSource || null,
          use_rewrite: useRewrite,
          use_hyde: useHyde,
          use_reranker: useReranker
        })
      });

      if (!res.ok) throw new Error('API request failed');
      const data = await res.json();

      const aiMsg: ChatMessage = {
        id: String(Date.now() + 1),
        sender: 'assistant',
        text: data.answer,
        retrieved_chunks: data.retrieved_chunks,
        citation_info: data.citation_info,
        latency_sec: data.latency_sec,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setMessages(prev => [...prev, aiMsg]);
    } catch (error) {
      setMessages(prev => [
        ...prev,
        {
          id: String(Date.now() + 1),
          sender: 'assistant',
          text: 'Error: Could not reach backend API. Make sure backend server is running on port 8000.',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Run Strategy Comparison
  const handleRunCompare = async () => {
    if (!compareQuery.trim() || compareLoading) return;
    setCompareLoading(true);
    try {
      const res = await fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: compareQuery })
      });
      if (res.ok) setCompareResults(await res.json());
    } catch (e) {
      console.error('Compare failed', e);
    } finally {
      setCompareLoading(false);
    }
  };

  // Run Evaluation Benchmark
  const handleRunBenchmark = async () => {
    setEvalLoading(true);
    try {
      const res = await fetch('/api/evaluation/run?max_cases=3', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setEvalReport(data.report);
      }
    } catch (e) {
      console.error('Eval run failed', e);
    } finally {
      setEvalLoading(false);
    }
  };

  // Run Security Test
  const handleRunSecurityTest = async () => {
    setSecurityLoading(true);
    try {
      const res = await fetch('/api/security-test', { method: 'POST' });
      if (res.ok) setSecurityResult(await res.json());
    } catch (e) {
      console.error('Security test failed', e);
    } finally {
      setSecurityLoading(false);
    }
  };

  // Document Upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        await fetchDocuments();
        await fetchHealth();
        alert(`Uploaded and indexed ${file.name} successfully!`);
      }
    } catch (err) {
      alert('Upload failed');
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 font-sans">
      {/* Top Header */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur px-6 py-3.5 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-semibold text-lg leading-tight bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent">
              Enterprise RAG Assistant
            </h1>
            <p className="text-xs text-slate-400">Production Hybrid Search & Citation Engine</p>
          </div>
        </div>

        {/* System Badges */}
        <div className="hidden lg:flex items-center space-x-2 text-xs">
          <span className="px-2.5 py-1 rounded-full bg-slate-800 border border-slate-700/60 text-slate-300 flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            {healthStatus?.models?.chat || 'Gemini Flash Lite'}
          </span>
          <span className="px-2.5 py-1 rounded-full bg-slate-800 border border-slate-700/60 text-slate-300 flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-emerald-400" />
            {healthStatus?.total_chunks_indexed !== undefined
              ? `${healthStatus.total_chunks_indexed} Chunks (Qdrant)`
              : 'Qdrant Vector DB'}
          </span>
          <span className="px-2.5 py-1 rounded-full bg-slate-800 border border-slate-700/60 text-slate-300 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-amber-400" />
            BM25 + FlashRank Rerank
          </span>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center space-x-1 bg-slate-950 p-1 rounded-xl border border-slate-800 text-sm">
          <button
            onClick={() => setActiveTab('chat')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg font-medium transition ${
              activeTab === 'chat' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-white'
            }`}
          >
            <MessageSquare className="w-4 h-4" />
            <span>Assistant</span>
          </button>
          <button
            onClick={() => setActiveTab('compare')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg font-medium transition ${
              activeTab === 'compare' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-white'
            }`}
          >
            <Scale className="w-4 h-4" />
            <span>Comparison</span>
          </button>
          <button
            onClick={() => setActiveTab('eval')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg font-medium transition ${
              activeTab === 'eval' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-white'
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            <span>Evaluation</span>
          </button>
          <button
            onClick={() => setActiveTab('security')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg font-medium transition ${
              activeTab === 'security' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-white'
            }`}
          >
            <ShieldCheck className="w-4 h-4" />
            <span>Security</span>
          </button>
          <button
            onClick={() => setActiveTab('docs')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg font-medium transition ${
              activeTab === 'docs' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-white'
            }`}
          >
            <FileText className="w-4 h-4" />
            <span>Documents ({documents.length})</span>
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 overflow-hidden flex flex-col">
        {/* ===================== TAB 1: CHAT ASSISTANT ===================== */}
        {activeTab === 'chat' && (
          <div className="flex-1 flex flex-col max-w-5xl w-full mx-auto p-4 overflow-hidden">
            {/* Advanced Retrieval Controls Bar */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl px-4 py-2.5 mb-3 flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex items-center space-x-3">
                <span className="text-slate-400 font-medium flex items-center gap-1.5">
                  <Sliders className="w-3.5 h-3.5 text-indigo-400" />
                  Retrieval Pipeline:
                </span>

                {/* Reranker Toggle */}
                <label className="flex items-center space-x-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={useReranker}
                    onChange={e => setUseReranker(e.target.checked)}
                    className="rounded bg-slate-800 border-slate-700 text-indigo-600 focus:ring-0"
                  />
                  <span className={useReranker ? 'text-indigo-400 font-semibold' : 'text-slate-400'}>
                    FlashRank Reranking
                  </span>
                </label>

                {/* Query Rewrite Toggle */}
                <label className="flex items-center space-x-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={useRewrite}
                    onChange={e => setUseRewrite(e.target.checked)}
                    className="rounded bg-slate-800 border-slate-700 text-indigo-600 focus:ring-0"
                  />
                  <span className={useRewrite ? 'text-indigo-400 font-semibold' : 'text-slate-400'}>
                    Query Rewriting
                  </span>
                </label>

                {/* HyDE Toggle */}
                <label className="flex items-center space-x-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={useHyde}
                    onChange={e => setUseHyde(e.target.checked)}
                    className="rounded bg-slate-800 border-slate-700 text-indigo-600 focus:ring-0"
                  />
                  <span className={useHyde ? 'text-indigo-400 font-semibold' : 'text-slate-400'}>
                    HyDE (Hypothetical Doc)
                  </span>
                </label>
              </div>

              {/* Source Filter Dropdown */}
              <div className="flex items-center space-x-2">
                <span className="text-slate-400">Filter Source:</span>
                <select
                  value={selectedSource}
                  onChange={e => setSelectedSource(e.target.value)}
                  className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1 text-slate-300 focus:outline-none focus:border-indigo-500"
                >
                  <option value="">All Ingested Documents</option>
                  {documents.map(doc => (
                    <option key={doc.filename} value={doc.filename}>
                      {doc.filename}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Messages Scroll View */}
            <div className="flex-1 overflow-y-auto space-y-4 pr-2">
              {messages.map(msg => (
                <div
                  key={msg.id}
                  className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
                >
                  <div
                    className={`max-w-3xl rounded-2xl px-5 py-3.5 leading-relaxed text-sm shadow ${
                      msg.sender === 'user'
                        ? 'bg-indigo-600 text-white rounded-br-none'
                        : 'bg-slate-900 border border-slate-800 text-slate-200 rounded-bl-none'
                    }`}
                  >
                    <div className="whitespace-pre-wrap">{msg.text}</div>

                    {/* Citations & Evidence Panel for AI response */}
                    {msg.sender === 'assistant' && msg.retrieved_chunks && msg.retrieved_chunks.length > 0 && (
                      <div className="mt-4 pt-3 border-t border-slate-800 space-y-2">
                        {/* Phase 9 Citation Verification Badge */}
                        <div className="flex items-center justify-between text-xs">
                          <div className="flex items-center space-x-1.5">
                            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                            <span className="font-semibold text-emerald-400">Phase 9 Grounding Verified</span>
                            <span className="text-slate-400">
                              ({msg.citation_info?.valid_citations.length || 0} citations validated)
                            </span>
                          </div>
                          {msg.latency_sec && (
                            <span className="text-slate-500 font-mono text-[11px]">
                              ⚡ {msg.latency_sec}s
                            </span>
                          )}
                        </div>

                        {/* Citation Source Tags */}
                        <div className="flex flex-wrap gap-1.5">
                          {msg.retrieved_chunks.map((c, i) => (
                            <span
                              key={i}
                              className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-md bg-slate-800 text-slate-300 text-xs border border-slate-700/50"
                            >
                              <BookOpen className="w-3 h-3 text-indigo-400" />
                              <span>
                                [DOC-{i + 1}] {c.source} (Page {c.page})
                              </span>
                              {c.rerank_score && (
                                <span className="text-emerald-400 font-mono font-medium">
                                  {c.rerank_score}
                                </span>
                              )}
                            </span>
                          ))}
                        </div>

                        {/* Toggle Context Chunks Accordion */}
                        <button
                          onClick={() =>
                            setExpandedChunks(prev => ({ ...prev, [msg.id]: !prev[msg.id] }))
                          }
                          className="flex items-center space-x-1 text-xs text-indigo-400 hover:text-indigo-300 font-medium pt-1"
                        >
                          <span>{expandedChunks[msg.id] ? 'Hide' : 'Inspect'} Retrieved Context Chunks</span>
                          {expandedChunks[msg.id] ? (
                            <ChevronUp className="w-3.5 h-3.5" />
                          ) : (
                            <ChevronDown className="w-3.5 h-3.5" />
                          )}
                        </button>

                        {expandedChunks[msg.id] && (
                          <div className="mt-2 space-y-2 bg-slate-950 p-3 rounded-xl border border-slate-800">
                            {msg.retrieved_chunks.map((chunk, idx) => (
                              <div key={idx} className="text-xs border-b border-slate-800 last:border-0 pb-2">
                                <div className="flex items-center justify-between text-slate-400 mb-1">
                                  <span className="font-semibold text-slate-300">
                                    [DOC-{idx + 1}] {chunk.source} - Page {chunk.page}
                                  </span>
                                  {chunk.rerank_score && (
                                    <span className="text-emerald-400">Score: {chunk.rerank_score}</span>
                                  )}
                                </div>
                                <p className="text-slate-400 font-mono text-[11px] leading-relaxed">
                                  {chunk.content}
                                </p>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  <span className="text-[11px] text-slate-500 mt-1 px-1">{msg.timestamp}</span>
                </div>
              ))}
              {loading && (
                <div className="flex items-center space-x-2 text-slate-400 text-sm bg-slate-900 border border-slate-800 px-4 py-2.5 rounded-2xl w-fit">
                  <RefreshCw className="w-4 h-4 animate-spin text-indigo-500" />
                  <span>Hybrid retrieval & grounding response...</span>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Quick Prompt Suggestions */}
            {messages.length === 1 && (
              <div className="my-3 flex flex-wrap gap-2 text-xs">
                <span className="text-slate-500 flex items-center gap-1">Try asking:</span>
                {[
                  'What is RAG and why is it used?',
                  'What are the common chunking strategies?',
                  'What is a vector database?',
                  'How do embeddings represent semantic similarity?'
                ].map((s, i) => (
                  <button
                    key={i}
                    onClick={() => handleSendMessage(s)}
                    className="bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 px-2.5 py-1 rounded-lg transition"
                  >
                    "{s}"
                  </button>
                ))}
              </div>
            )}

            {/* Input Bar */}
            <div className="mt-3 pt-2">
              <form
                onSubmit={e => {
                  e.preventDefault();
                  handleSendMessage();
                }}
                className="relative flex items-center"
              >
                <input
                  type="text"
                  value={inputQuery}
                  onChange={e => setInputQuery(e.target.value)}
                  placeholder="Ask a question about your documents..."
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl pl-4 pr-12 py-3 text-sm focus:outline-none focus:border-indigo-500 transition shadow-inner placeholder-slate-500"
                />
                <button
                  type="submit"
                  disabled={!inputQuery.trim() || loading}
                  className="absolute right-2 p-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white rounded-lg transition"
                >
                  <Send className="w-4 h-4" />
                </button>
              </form>
            </div>
          </div>
        )}

        {/* ===================== TAB 2: RETRIEVAL COMPARISON (Phase 8) ===================== */}
        {activeTab === 'compare' && (
          <div className="flex-1 overflow-y-auto p-6 max-w-6xl w-full mx-auto space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <h2 className="text-lg font-semibold flex items-center gap-2 mb-2">
                <Scale className="w-5 h-5 text-indigo-400" />
                Phase 8: Multi-Strategy Retrieval Comparison
              </h2>
              <p className="text-sm text-slate-400 mb-4">
                Compare Dense Semantic Search, BM25 Lexical Keyword Search, Reciprocal Rank Fusion (RRF), and
                Cross-Encoder Reranking side-by-side to understand ranking differences.
              </p>

              <div className="flex gap-2">
                <input
                  type="text"
                  value={compareQuery}
                  onChange={e => setCompareQuery(e.target.value)}
                  placeholder="Enter test query for retrieval..."
                  className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500"
                />
                <button
                  onClick={handleRunCompare}
                  disabled={compareLoading}
                  className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm rounded-xl transition flex items-center gap-2"
                >
                  {compareLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <ArrowRight className="w-4 h-4" />}
                  <span>Compare</span>
                </button>
              </div>
            </div>

            {compareResults && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* 1. Dense Only */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
                  <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-3">
                    <span className="font-semibold text-sm text-indigo-400">1. Dense (Vector)</span>
                    <span className="text-[11px] px-2 py-0.5 rounded bg-indigo-950/80 text-indigo-300 border border-indigo-800">
                      Qdrant
                    </span>
                  </div>
                  <div className="space-y-3 flex-1 overflow-y-auto">
                    {compareResults.dense.map((chunk: ChunkItem, i: number) => (
                      <div key={i} className="bg-slate-950 p-2.5 rounded-lg border border-slate-800/80 text-xs">
                        <div className="text-slate-400 mb-1 font-semibold flex justify-between">
                          <span>Rank #{i + 1}</span>
                          <span>Page {chunk.page}</span>
                        </div>
                        <p className="text-slate-300 line-clamp-4 font-mono text-[11px]">{chunk.content}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* 2. BM25 Only */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
                  <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-3">
                    <span className="font-semibold text-sm text-amber-400">2. Lexical (BM25)</span>
                    <span className="text-[11px] px-2 py-0.5 rounded bg-amber-950/80 text-amber-300 border border-amber-800">
                      Exact Match
                    </span>
                  </div>
                  <div className="space-y-3 flex-1 overflow-y-auto">
                    {compareResults.bm25.map((chunk: ChunkItem, i: number) => (
                      <div key={i} className="bg-slate-950 p-2.5 rounded-lg border border-slate-800/80 text-xs">
                        <div className="text-slate-400 mb-1 font-semibold flex justify-between">
                          <span>Rank #{i + 1}</span>
                          <span>Page {chunk.page}</span>
                        </div>
                        <p className="text-slate-300 line-clamp-4 font-mono text-[11px]">{chunk.content}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* 3. Hybrid RRF */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
                  <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-3">
                    <span className="font-semibold text-sm text-purple-400">3. Hybrid (RRF)</span>
                    <span className="text-[11px] px-2 py-0.5 rounded bg-purple-950/80 text-purple-300 border border-purple-800">
                      Merged
                    </span>
                  </div>
                  <div className="space-y-3 flex-1 overflow-y-auto">
                    {compareResults.hybrid.map((chunk: ChunkItem, i: number) => (
                      <div key={i} className="bg-slate-950 p-2.5 rounded-lg border border-slate-800/80 text-xs">
                        <div className="text-slate-400 mb-1 font-semibold flex justify-between">
                          <span>Rank #{i + 1}</span>
                          <span>Page {chunk.page}</span>
                        </div>
                        <p className="text-slate-300 line-clamp-4 font-mono text-[11px]">{chunk.content}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* 4. FlashRank Reranked */}
                <div className="bg-slate-900 border border-emerald-800/50 rounded-xl p-4 flex flex-col shadow-lg shadow-emerald-950/20">
                  <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-3">
                    <span className="font-semibold text-sm text-emerald-400">4. Reranked</span>
                    <span className="text-[11px] px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-800 font-bold">
                      FlashRank
                    </span>
                  </div>
                  <div className="space-y-3 flex-1 overflow-y-auto">
                    {compareResults.reranked.map((chunk: ChunkItem, i: number) => (
                      <div key={i} className="bg-slate-950 p-2.5 rounded-lg border border-emerald-900/60 text-xs">
                        <div className="text-slate-400 mb-1 font-semibold flex justify-between">
                          <span className="text-emerald-400 font-bold">Rank #{i + 1}</span>
                          <span className="text-emerald-400 font-mono">Score: {chunk.rerank_score}</span>
                        </div>
                        <p className="text-slate-300 line-clamp-4 font-mono text-[11px]">{chunk.content}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ===================== TAB 3: EVALUATION BENCHMARK (Phase 10) ===================== */}
        {activeTab === 'eval' && (
          <div className="flex-1 overflow-y-auto p-6 max-w-5xl w-full mx-auto space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold flex items-center gap-2">
                  <BarChart3 className="w-6 h-6 text-indigo-400" />
                  Phase 10: Evaluation & Measurable Quality
                </h2>
                <p className="text-sm text-slate-400">
                  Automated benchmark evaluation comparing Baseline (Dense-only) vs. Tuned (Hybrid + Reranker)
                  using LLM-as-a-Judge.
                </p>
              </div>

              <button
                onClick={handleRunBenchmark}
                disabled={evalLoading}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm rounded-xl transition flex items-center gap-2"
              >
                {evalLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
                <span>Run Live Benchmark</span>
              </button>
            </div>

            {evalReport && evalReport.metrics_summary && (
              <>
                {/* Metric Summary Cards */}
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                  <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 text-center">
                    <span className="text-xs text-slate-400 block mb-1">Hit Rate @ 4</span>
                    <span className="text-2xl font-bold text-emerald-400">
                      {evalReport.metrics_summary.tuned.hit_rate_at_4}%
                    </span>
                    <span className="text-[10px] text-slate-500 block mt-0.5">
                      Base: {evalReport.metrics_summary.baseline.hit_rate_at_4}%
                    </span>
                  </div>

                  <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 text-center">
                    <span className="text-xs text-slate-400 block mb-1">MRR Score</span>
                    <span className="text-2xl font-bold text-indigo-400">
                      {evalReport.metrics_summary.tuned.mrr}
                    </span>
                    <span className="text-[10px] text-slate-500 block mt-0.5">
                      Base: {evalReport.metrics_summary.baseline.mrr}
                    </span>
                  </div>

                  <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 text-center">
                    <span className="text-xs text-slate-400 block mb-1">Faithfulness</span>
                    <span className="text-2xl font-bold text-emerald-400">
                      {evalReport.metrics_summary.tuned.avg_faithfulness}%
                    </span>
                    <span className="text-[10px] text-slate-500 block mt-0.5">Anti-Hallucination</span>
                  </div>

                  <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 text-center">
                    <span className="text-xs text-slate-400 block mb-1">Relevance</span>
                    <span className="text-2xl font-bold text-purple-400">
                      {evalReport.metrics_summary.tuned.avg_relevance}%
                    </span>
                    <span className="text-[10px] text-slate-500 block mt-0.5">Answer Directness</span>
                  </div>

                  <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 text-center">
                    <span className="text-xs text-slate-400 block mb-1">Abstention Acc.</span>
                    <span className="text-2xl font-bold text-amber-400">
                      {evalReport.metrics_summary.tuned.abstention_accuracy}%
                    </span>
                    <span className="text-[10px] text-slate-500 block mt-0.5">Safe refusal</span>
                  </div>

                  <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 text-center">
                    <span className="text-xs text-slate-400 block mb-1">Avg Latency</span>
                    <span className="text-2xl font-bold text-slate-200">
                      {evalReport.metrics_summary.tuned.avg_latency_sec}s
                    </span>
                    <span className="text-[10px] text-slate-500 block mt-0.5">
                      Base: {evalReport.metrics_summary.baseline.avg_latency_sec}s
                    </span>
                  </div>
                </div>

                {/* Comparison Table */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 overflow-hidden">
                  <h3 className="font-semibold text-sm mb-3">Baseline vs. Tuned Pipeline Comparison</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead>
                        <tr className="border-b border-slate-800 text-xs text-slate-400">
                          <th className="py-2.5 font-medium">Metric</th>
                          <th className="py-2.5 font-medium">Baseline (Dense Only)</th>
                          <th className="py-2.5 font-medium text-emerald-400">Tuned (Hybrid + FlashRank)</th>
                          <th className="py-2.5 font-medium">Impact</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 text-xs">
                        <tr>
                          <td className="py-3 font-medium">Recall / Hit Rate @ 4</td>
                          <td>{evalReport.metrics_summary.baseline.hit_rate_at_4}%</td>
                          <td className="text-emerald-400 font-bold">
                            {evalReport.metrics_summary.tuned.hit_rate_at_4}%
                          </td>
                          <td className="text-slate-400">High relevant passage retrieval</td>
                        </tr>
                        <tr>
                          <td className="py-3 font-medium">Mean Reciprocal Rank (MRR)</td>
                          <td>{evalReport.metrics_summary.baseline.mrr}</td>
                          <td className="text-emerald-400 font-bold">{evalReport.metrics_summary.tuned.mrr}</td>
                          <td className="text-slate-400">Top-1 rank prioritization</td>
                        </tr>
                        <tr>
                          <td className="py-3 font-medium">Faithfulness (Grounding)</td>
                          <td>{evalReport.metrics_summary.baseline.avg_faithfulness}%</td>
                          <td className="text-emerald-400 font-bold">
                            {evalReport.metrics_summary.tuned.avg_faithfulness}%
                          </td>
                          <td className="text-slate-400">Zero outside knowledge hallucination</td>
                        </tr>
                        <tr>
                          <td className="py-3 font-medium">Answer Relevance</td>
                          <td>{evalReport.metrics_summary.baseline.avg_relevance}%</td>
                          <td className="text-emerald-400 font-bold">
                            {evalReport.metrics_summary.tuned.avg_relevance}%
                          </td>
                          <td className="text-slate-400">High answer directness</td>
                        </tr>
                        <tr>
                          <td className="py-3 font-medium">Abstention Accuracy (Safety)</td>
                          <td>{evalReport.metrics_summary.baseline.abstention_accuracy}%</td>
                          <td className="text-emerald-400 font-bold">
                            {evalReport.metrics_summary.tuned.abstention_accuracy}%
                          </td>
                          <td className="text-slate-400">Refuses out-of-domain/injection</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  <div className="mt-4 pt-3 border-t border-slate-800 text-[11px] text-slate-500">
                    Last benchmark run: {evalReport.timestamp} | Model: {evalReport.model_evaluated}
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* ===================== TAB 4: SECURITY & PROMPT INJECTION (Phase 9) ===================== */}
        {activeTab === 'security' && (
          <div className="flex-1 overflow-y-auto p-6 max-w-4xl w-full mx-auto space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <h2 className="text-lg font-semibold flex items-center gap-2 mb-2">
                <ShieldCheck className="w-5 h-5 text-emerald-400" />
                Phase 9: Prompt Injection Defense & Data Isolation
              </h2>
              <p className="text-sm text-slate-400 mb-4">
                Enterprise documents are treated as untrusted data. Context passages are fenced inside
                strict XML tags (<code className="text-indigo-400">&lt;untrusted_document&gt;</code>) to prevent
                instruction hijacking from malicious uploaded documents.
              </p>

              <button
                onClick={handleRunSecurityTest}
                disabled={securityLoading}
                className="px-4 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-sm rounded-xl transition flex items-center gap-2"
              >
                {securityLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <ShieldAlert className="w-4 h-4" />}
                <span>Simulate Adversarial Prompt Injection</span>
              </button>
            </div>

            {securityResult && (
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-sm">Security Audit Result</span>
                  <span
                    className={`px-3 py-1 rounded-full text-xs font-bold border ${
                      securityResult.passed
                        ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                        : 'bg-rose-950 text-rose-400 border-rose-800'
                    }`}
                  >
                    {securityResult.defense_status}
                  </span>
                </div>

                <div className="text-xs space-y-1">
                  <span className="text-slate-400 block">Attacker Payload:</span>
                  <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800 text-rose-400 font-mono">
                    "{securityResult.attack_prompt}"
                  </div>
                </div>

                <div className="text-xs space-y-1">
                  <span className="text-slate-400 block">System Guarded Response:</span>
                  <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800 text-slate-200">
                    {securityResult.response}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ===================== TAB 5: DOCUMENT MANAGEMENT ===================== */}
        {activeTab === 'docs' && (
          <div className="flex-1 overflow-y-auto p-6 max-w-4xl w-full mx-auto space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <h2 className="text-lg font-semibold flex items-center gap-2 mb-2">
                <FileText className="w-5 h-5 text-indigo-400" />
                Enterprise Document Ingestion
              </h2>
              <p className="text-sm text-slate-400 mb-4">
                Upload policy, contract, or technical PDF files. The system automatically chunks the text and
                indexes them into the local Qdrant Vector database and BM25 token index.
              </p>

              {/* Upload Input */}
              <label className="border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-2xl p-8 flex flex-col items-center justify-center cursor-pointer transition bg-slate-950/50">
                <Upload className="w-8 h-8 text-indigo-400 mb-2" />
                <span className="text-sm font-medium text-slate-200">Click to upload new PDF document</span>
                <span className="text-xs text-slate-500 mt-1">Files are saved to backend/data/</span>
                <input type="file" accept=".pdf" onChange={handleFileUpload} className="hidden" />
              </label>
            </div>

            {/* Ingested Documents List */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <h3 className="font-semibold text-sm mb-3">Active Ingested Files</h3>
              {documents.length === 0 ? (
                <p className="text-xs text-slate-500">No documents found.</p>
              ) : (
                <div className="divide-y divide-slate-800">
                  {documents.map((doc, idx) => (
                    <div key={idx} className="py-3 flex items-center justify-between text-xs">
                      <div className="flex items-center space-x-2.5">
                        <FileText className="w-4 h-4 text-indigo-400" />
                        <span className="font-medium text-slate-200">{doc.filename}</span>
                        <span className="text-slate-500">({doc.size_kb} KB)</span>
                      </div>
                      <span className="text-slate-500">{doc.modified}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
