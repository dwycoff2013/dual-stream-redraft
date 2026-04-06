import React, { useEffect, useMemo, useState } from 'react'
import SplitPane from 'react-split-pane'
import { ArtifactList } from './components/ArtifactList'
import { FrameExplorer } from './components/FrameExplorer'
import { JsonViewer } from './components/JsonViewer'

type Tab = 'generate' | 'solve-task' | 'solve-dataset' | 'kaggle-submit' | 'artifacts'

const API = 'http://127.0.0.1:8765'

export function App() {
  const [tab, setTab] = useState<Tab>('generate')
  const [jobId, setJobId] = useState<string>('')
  const [job, setJob] = useState<any>(null)
  const [form, setForm] = useState<any>({ model: 'gpt2', outdir: 'runs/gui/generate', prompt: '', max_new_tokens: 128, top_k: 5, temperature: 1, top_p: 1 })

  useEffect(() => {
    if (!jobId) return
    const timer = setInterval(async () => {
      const res = await fetch(`${API}/jobs/${jobId}`)
      if (res.ok) {
        const data = await res.json()
        setJob(data)
      }
    }, 1200)
    return () => clearInterval(timer)
  }, [jobId])

  const start = async (endpoint: string, payload: any) => {
    const res = await fetch(`${API}${endpoint}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
    const data = await res.json()
    setJobId(data.job_id)
  }

  const result = useMemo(() => job?.result || {}, [job])

  return (
    <div className="app">
      <aside className="sidebar">{(['generate', 'solve-task', 'solve-dataset', 'kaggle-submit', 'artifacts'] as Tab[]).map((t) => <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>{t}</button>)}</aside>
      <main className="main">
        <header><h1>DualStream Desktop</h1><div className="status">{job ? `${job.status} • ${(job.progress * 100).toFixed(0)}% • ${job.message}` : 'Idle'}</div></header>
        <div className="stepper">
          <span className={job?.status === 'queued' ? 'on' : ''}>Queued</span><span className={job?.status === 'running' ? 'on' : ''}>Running</span><span className={job?.status === 'completed' ? 'on' : ''}>Completed</span><span className={job?.status === 'failed' ? 'on' : ''}>Failed</span>
        </div>
        <SplitPane split="vertical" minSize={360} defaultSize={460}>
          <section className="panel">
            {tab === 'generate' && <>
              <h2>Generate</h2>
              <label>Model<input value={form.model} onChange={(e) => setForm({ ...form, model: e.target.value })} /></label>
              <label>Prompt<textarea value={form.prompt} onChange={(e) => setForm({ ...form, prompt: e.target.value })} /></label>
              <label>Prompt File<input value={form.prompt_file || ''} onChange={(e) => setForm({ ...form, prompt_file: e.target.value })} placeholder="optional"/></label>
              <label>Output Dir<input value={form.outdir} onChange={(e) => setForm({ ...form, outdir: e.target.value })} /></label>
              <div className="grid2">
                <label>max_new_tokens<input type="number" value={form.max_new_tokens} onChange={(e) => setForm({ ...form, max_new_tokens: Number(e.target.value) })} /></label>
                <label>top_k<input type="number" value={form.top_k} onChange={(e) => setForm({ ...form, top_k: Number(e.target.value) })} /></label>
                <label>temperature<input type="number" step="0.1" value={form.temperature} onChange={(e) => setForm({ ...form, temperature: Number(e.target.value) })} /></label>
                <label>top_p<input type="number" step="0.05" value={form.top_p} onChange={(e) => setForm({ ...form, top_p: Number(e.target.value) })} /></label>
              </div>
              {['greedy','include_attn','include_probes','no_heuristics','no_crc32','no_running_hash','offline'].map((k)=><label key={k} className="check"><input type="checkbox" checked={!!form[k]} onChange={(e)=>setForm({...form,[k]:e.target.checked})}/>{k}</label>)}
              <label>seed<input type="number" value={form.seed || ''} onChange={(e)=>setForm({...form,seed:e.target.value?Number(e.target.value):undefined})}/></label>
              <label>probe_pack<input value={form.probe_pack || ''} onChange={(e)=>setForm({...form,probe_pack:e.target.value})}/></label>
              <label>device<input value={form.device || ''} onChange={(e)=>setForm({...form,device:e.target.value})}/></label>
              <label>cache_dir<input value={form.cache_dir || ''} onChange={(e)=>setForm({...form,cache_dir:e.target.value})}/></label>
              <div className="actions"><button disabled={!form.prompt && !form.prompt_file || job?.status==='running'} onClick={() => start('/generate', form)}>Run</button><button onClick={() => jobId && fetch(`${API}/jobs/${jobId}/cancel`, {method:'POST'})}>Cancel</button></div>
            </>}
            {tab === 'solve-task' && <ArcForm title="ARC Solve Task" defaults={{ task:'', outdir:'runs/gui/arc-task' }} onRun={(payload)=>start('/arc/solve-task', payload)} />}
            {tab === 'solve-dataset' && <ArcForm title="ARC Solve Dataset" defaults={{ tasks_dir:'', outdir:'runs/gui/arc-dataset' }} onRun={(payload)=>start('/arc/solve-dataset', payload)} />}
            {tab === 'kaggle-submit' && <ArcForm title="Kaggle Submit" defaults={{ tasks_dir:'', output:'runs/gui/submission.json' }} onRun={(payload)=>start('/arc/kaggle-submit', payload)} />}
            {tab === 'artifacts' && <p>Run a job and inspect files/results here.</p>}
          </section>
          <section className="panel result">
            <h2>Results</h2>
            <ArtifactList artifacts={result.artifacts || []} jobId={jobId} />
            {result.answer_text && <pre>{result.answer_text}</pre>}
            {result.monologue_text && <pre>{result.monologue_text}</pre>}
            {result.frames && <FrameExplorer frames={result.frames} />}
            <JsonViewer title="Meta Summary" value={result.meta_summary} />
            <JsonViewer title="Audit Summary" value={result.audit_summary} />
            <JsonViewer title="Metrics" value={result.metrics} />
            <JsonViewer title="Candidate Rankings" value={result.candidate_rankings} />
            <JsonViewer title="Trace" value={result.trace} />
          </section>
        </SplitPane>
      </main>
    </div>
  )
}

function ArcForm({ title, defaults, onRun }: { title: string; defaults: any; onRun: (payload: any) => void }) {
  const [state, setState] = useState<any>({
    ...defaults,
    max_program_depth: 2,
    max_candidates: 128,
    beam_width: 24,
    diversity_penalty: 0.1,
    emit_candidate_rankings: true,
  })
  return <>
    <h2>{title}</h2>
    {Object.keys(defaults).map((k)=><label key={k}>{k}<input value={state[k]} onChange={(e)=>setState({...state,[k]:e.target.value})}/></label>)}
    <div className="grid2">
      <label>max_program_depth<input type="number" value={state.max_program_depth} onChange={(e)=>setState({...state,max_program_depth:Number(e.target.value)})}/></label>
      <label>max_candidates<input type="number" value={state.max_candidates} onChange={(e)=>setState({...state,max_candidates:Number(e.target.value)})}/></label>
      <label>beam_width<input type="number" value={state.beam_width} onChange={(e)=>setState({...state,beam_width:Number(e.target.value)})}/></label>
      <label>diversity_penalty<input type="number" step="0.05" value={state.diversity_penalty} onChange={(e)=>setState({...state,diversity_penalty:Number(e.target.value)})}/></label>
    </div>
    {['no_trace','no_integrity','emit_candidate_rankings'].map((k)=><label className="check" key={k}><input type="checkbox" checked={!!state[k]} onChange={(e)=>setState({...state,[k]:e.target.checked})}/>{k}</label>)}
    <div className="actions"><button onClick={()=>onRun(state)}>Run</button></div>
  </>
}
