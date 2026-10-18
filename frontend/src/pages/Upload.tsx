import { useRef, useState } from 'react'
import { FileCheck2, FileUp, Zap } from 'lucide-react'
import MatchView from '../components/MatchView'
import { apiPost, type Row } from '../lib/api'

export default function Upload() {
  const [ball, setBall] = useState<File | null>(null)
  const [info, setInfo] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [res, setRes] = useState<Row | null>(null)

  const run = async () => {
    if (!ball || !info) return
    setBusy(true); setErr(''); setRes(null)
    const fd = new FormData()
    fd.append('ball_file', ball); fd.append('info_file', info)
    try { setRes(await apiPost<Row>('/upload', fd)) } catch (e) { setErr(String((e as Error).message)) }
    setBusy(false)
  }

  const onFiles = (files: FileList | null) => {
    Array.from(files ?? []).forEach(f => (f.name.endsWith('_info.csv') ? setInfo(f) : setBall(f)))
  }

  return (
    <div className="page">
      <div className="page-head">
        <h1>Upload a Match</h1>
        <p>Score a brand-new match with the trained win probability model. Upload the Cricsheet ball-by-ball CSV and its matching <code>_info.csv</code>.</p>
      </div>
      <div className="grid g2">
        <Drop label="Ball-by-ball CSV" hint="e.g. 335982.csv" file={ball} onFiles={onFiles} />
        <Drop label="Match info CSV" hint="e.g. 335982_info.csv" file={info} onFiles={onFiles} />
      </div>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', margin: '16px 0 24px', flexWrap: 'wrap' }}>
        <button className="btn" disabled={!ball || !info || busy} onClick={run}><Zap size={16} />{busy ? 'Computing impact…' : 'Compute match impact'}</button>
        {err && <span className="error">{err}</span>}
      </div>
      {res && <MatchView m={res} />}
    </div>
  )
}

function Drop({ label, hint, file, onFiles }: { label: string; hint: string; file: File | null; onFiles: (f: FileList | null) => void }) {
  const [over, setOver] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  return (
    <div className={`dropzone ${over ? 'over' : ''} ${file ? 'ok' : ''}`}
      onClick={() => input.current?.click()}
      onDragOver={e => { e.preventDefault(); setOver(true) }}
      onDragLeave={() => setOver(false)}
      onDrop={e => { e.preventDefault(); setOver(false); onFiles(e.dataTransfer.files) }}>
      <input ref={input} type="file" accept=".csv" multiple hidden onChange={e => onFiles(e.target.files)} />
      {file ? <FileCheck2 size={30} color="var(--good)" /> : <FileUp size={30} color="var(--muted)" />}
      <div style={{ fontWeight: 700, marginTop: 8 }}>{file ? file.name : label}</div>
      <div className="card-sub">{file ? `${(file.size / 1024).toFixed(0)} KB · click to replace` : `Drop or click to choose · ${hint}`}</div>
    </div>
  )
}
