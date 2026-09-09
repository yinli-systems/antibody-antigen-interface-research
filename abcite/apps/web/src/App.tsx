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

const EXAMPLES = [
  "Does trastuzumab on HER2 (PDB 1N8Z) share the pertuzumab domain II epitope?",
  "Using the trastuzumab epitope diagram, is the Fab contacting HER2 domain IV rather than domain II?",
  "Does unpublished antibody ZX-99 overlap PDB 9ZZZ on antigen Q?",
  "Clinical therapeutic decision: should a patient switch from trastuzumab to pertuzumab because epitopes overlap 100%?",
];

export default function App() {
  const [question, setQuestion] = useState(EXAMPLES[0]);
  const [highStakes, setHighStakes] = useState(false);
  const [resp, setResp] = useState<QueryResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const r = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, high_stakes: highStakes, top_k: 6 }),
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
        <span>Cite-or-abstain copilot · papers · figures · PDB residue maps · Yin Li 2026-09-09</span>
      </header>

      <form className="panel" onSubmit={onSubmit}>
        <label htmlFor="q">Question</label>
        <textarea id="q" value={question} onChange={(e) => setQuestion(e.target.value)} />
        <label className="flag">
          <input type="checkbox" checked={highStakes} onChange={(e) => setHighStakes(e.target.checked)} />
          Mark high-stakes (forces HITL)
        </label>
        <button className="primary" disabled={busy} type="submit">
          {busy ? "Retrieving…" : "Query"}
        </button>
        {error && <p className="quote">{error}</p>}
        <div className="examples">
          {EXAMPLES.map((ex) => (
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
