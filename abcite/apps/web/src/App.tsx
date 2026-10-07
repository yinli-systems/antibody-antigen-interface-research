import { FormEvent, useMemo, useState } from "react";

type Citation = {
  chunk_id: string;
  claim: string;
  quote: string;
  modality: string;
  source_url: string;
  source_label: string;
  thumbnail_url: string | null;
  pdb_id: string | null;
  residues: string[];
};

type QueryResponse = {
  trace_id: string;
  status: "answered" | "abstain" | "hitl_pending";
  answer: string;
  citations: Citation[];
  residues: string[];
  hitl_reason: string | null;
  provider: string;
  latency_ms: number;
};

const EVIDENCE_EXAMPLES = [
  "Does trastuzumab on HER2 (PDB 1N8Z) share the pertuzumab domain II epitope?",
  "Using the trastuzumab epitope diagram, is the Fab contacting HER2 domain IV rather than domain II?",
  "Does unpublished antibody ZX-99 overlap PDB 9ZZZ on antigen Q?",
  "Clinical therapeutic decision: should a patient switch from trastuzumab to pertuzumab because epitopes overlap 100%?",
];

const VISUAL_EXAMPLES = [
  "four grayscale micrograph tiles of a wrinkled polymer surface at increasing roughness",
  "seven vertical sample lanes of dark bands beside a rainbow-stained marker column on the left",
  "emerald nuclear speckles against a dark field",
  "flowchart split by dotted silver rules with emerald checkmarks and crimson x-marks",
];

const AUDIO_EXAMPLES = [
  "spoken remark about a viral protein's far-end triplicate top, without reading a slide",
  "which recording describes a mid-mass protein rung that appears only after a chemical kick",
];

const VIDEO_EXAMPLES = [
  "lecture cut from a scarlet similarity grid to a blue volcano of points",
  "two-frame microscopy: first a dark view of emerald dots, then an indigo oval overlay",
];

type Task = "evidence" | "visual_qa" | "audio_qa" | "video_qa" | "unified";

export default function App() {
  const [task, setTask] = useState<Task>("visual_qa");
  const [question, setQuestion] = useState(VISUAL_EXAMPLES[0]);
  const [highStakes, setHighStakes] = useState(false);
  const [resp, setResp] = useState<QueryResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const examples =
    task === "visual_qa"
      ? VISUAL_EXAMPLES
      : task === "audio_qa"
        ? AUDIO_EXAMPLES
        : task === "video_qa"
          ? VIDEO_EXAMPLES
          : EVIDENCE_EXAMPLES;

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const r = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, high_stakes: highStakes, top_k: 6, task }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setResp((await r.json()) as QueryResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : "query failed");
    } finally {
      setBusy(false);
    }
  }

  async function decide(decision: "approve" | "reject") {
    if (!resp) return;
    await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trace_id: resp.trace_id, decision }),
    });
    setResp({
      ...resp,
      status: decision === "approve" ? "answered" : "abstain",
      hitl_reason: null,
    });
  }

  const statusClass = resp?.status ?? "";
  const structureCite = useMemo(
    () => resp?.citations.find((c) => c.modality === "structure") ?? null,
    [resp],
  );

  return (
    <div className="shell">
      <header className="top">
        <h1>AbCite</h1>
        <span>Cite-or-abstain multimodal RAG · text · figure · audio · video · Yin Li 2026-09-10</span>
      </header>

      <form className="panel" onSubmit={onSubmit}>
        <label htmlFor="q">Question</label>
        <textarea id="q" value={question} onChange={(e) => setQuestion(e.target.value)} />
        <label htmlFor="task">Task</label>
        <select
          id="task"
          value={task}
          onChange={(e) => {
            const next = e.target.value as Task;
            setTask(next);
            setQuestion(
              next === "visual_qa"
                ? VISUAL_EXAMPLES[0]
                : next === "audio_qa"
                  ? AUDIO_EXAMPLES[0]
                  : next === "video_qa"
                    ? VIDEO_EXAMPLES[0]
                    : EVIDENCE_EXAMPLES[0],
            );
          }}
        >
          <option value="visual_qa">visual figure QA</option>
          <option value="audio_qa">audio QA</option>
          <option value="video_qa">video QA</option>
          <option value="unified">unified multimodal</option>
          <option value="evidence">text evidence</option>
        </select>
        <label className="flag">
          <input type="checkbox" checked={highStakes} onChange={(e) => setHighStakes(e.target.checked)} />
          Mark high-stakes (forces HITL)
        </label>
        <button className="primary" disabled={busy} type="submit">
          {busy ? "Retrieving…" : "Query"}
        </button>
        {error && <p className="quote">{error}</p>}
        <div className="examples">
          {examples.map((ex) => (
            <button key={ex} type="button" onClick={() => setQuestion(ex)}>
              {ex}
            </button>
          ))}
        </div>
      </form>

      <section className="panel">
        {!resp && <p className="quote">Run a query. Answers without citations are not shown as facts.</p>}
        {resp && (
          <>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span className={`status ${statusClass}`}>{resp.status.replace("_", " ")}</span>
              <span className="quote">
                {resp.provider} · {resp.latency_ms.toFixed(0)} ms · {resp.trace_id.slice(0, 8)}
              </span>
            </div>
            <p className="answer">{resp.answer}</p>
            <h3>Citations</h3>
            <div className="cards">
              {resp.citations.map((c) => (
                <article className="card" key={c.chunk_id}>
                  {c.thumbnail_url ? (
                    <img src={c.thumbnail_url} alt={c.chunk_id} />
                  ) : (
                    <div className="thumb" />
                  )}
                  <div>
                    <strong>{c.chunk_id}</strong> · {c.modality}
                    <p className="quote">{c.quote}</p>
                    <a href={c.source_url} target="_blank" rel="noreferrer">
                      {c.source_label}
                    </a>
                  </div>
                </article>
              ))}
            </div>
          </>
        )}
      </section>

      <aside className="panel">
        <h3>Residues</h3>
        <div className="chips">
          {(resp?.residues ?? []).map((r) => (
            <span className="chip" key={r}>
              {r}
            </span>
          ))}
        </div>
        <h3>Structure</h3>
        {structureCite?.thumbnail_url ? (
          <img src={structureCite.thumbnail_url} alt="structure" style={{ width: "100%", borderRadius: 8 }} />
        ) : (
          <p className="quote">Cited structure snapshots appear here.</p>
        )}
        {structureCite?.pdb_id && (
          <p>
            <a href={`https://www.rcsb.org/structure/${structureCite.pdb_id}`} target="_blank" rel="noreferrer">
              RCSB {structureCite.pdb_id}
            </a>
          </p>
        )}
      </aside>

      {resp?.status === "hitl_pending" && (
        <div className="modal">
          <div className="box">
            <h2>Human-in-the-loop</h2>
            <p>{resp.hitl_reason}</p>
            <p className="quote">High-stakes draft is held until approve or reject.</p>
            <button className="primary" type="button" onClick={() => void decide("approve")}>
              Approve
            </button>
            <button type="button" onClick={() => void decide("reject")}>
              Reject
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
