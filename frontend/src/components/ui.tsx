import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { motion } from 'framer-motion'
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'
import { usePalette } from '../lib/theme'
import { num } from '../lib/format'
import type { Row } from '../lib/api'

/* ------------------------------------------------------------ card shells */
export function ChartCard({ title, sub, children, table, className = '', delay = 0 }: {
  title: string; sub?: string; children: ReactNode; table?: Row[]; className?: string; delay?: number
}) {
  const [showTable, setShowTable] = useState(false)
  return (
    <motion.section
      className={`card ${className}`}
      initial={{ opacity: 0, y: 14 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ duration: 0.45, delay }}
    >
      <div className="card-head">
        <div>
          <div className="card-title">{title}</div>
          {sub && <div className="card-sub">{sub}</div>}
        </div>
        {table && table.length > 0 && (
          <button className="ghost-btn" onClick={() => setShowTable(s => !s)}>
            {showTable ? 'Chart' : 'Table'}
          </button>
        )}
      </div>
      {showTable && table ? <DataTable rows={table} /> : children}
    </motion.section>
  )
}

export function DataTable({ rows, max = 50 }: { rows: Row[]; max?: number }) {
  const cols = Object.keys(rows[0] ?? {})
  return (
    <div className="table-wrap" style={{ maxHeight: 320 }}>
      <table className="tbl">
        <thead><tr>{cols.map(c => <th key={c}>{c.replace(/_/g, ' ')}</th>)}</tr></thead>
        <tbody>
          {rows.slice(0, max).map((r, i) => (
            <tr key={i}>{cols.map(c => <td key={c} className={typeof r[c] === 'number' ? 'num' : ''}>
              {typeof r[c] === 'number' ? num(r[c], Number.isInteger(r[c]) ? 0 : 2) : String(r[c] ?? '–')}
            </td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Kpi({ label, value, hint, color }: { label: string; value: ReactNode; hint?: ReactNode; color?: string }) {
  return (
    <div className="card kpi">
      <div className="kpi-label">{color && <span className="kpi-dot" style={{ background: color }} />}{label}</div>
      <div className="kpi-value">{value}</div>
      {hint && <div className="kpi-hint">{hint}</div>}
    </div>
  )
}

export const Loading = () => <div className="loading"><div className="spinner" /></div>
export const ErrorBox = ({ msg }: { msg: string }) => (
  <div className="notice"><span className="error">Could not load data.</span> {msg}. Make sure the backend is running on port 8000.</div>
)
export const Empty = ({ children }: { children: ReactNode }) => <div className="empty">{children}</div>

/* ------------------------------------------------------------ count-up */
export function CountUp({ to, duration = 1400, format = (n: number) => Math.round(n).toLocaleString('en-IN') }: {
  to: number; duration?: number; format?: (n: number) => string
}) {
  const [v, setV] = useState(0)
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { setV(to); return }
    let raf = 0
    const start = performance.now()
    const tick = (now: number) => {
      const p = Math.min(1, (now - start) / duration)
      setV(to * (1 - Math.pow(1 - p, 3)))
      if (p < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [to, duration])
  return <>{format(v)}</>
}

/* ------------------------------------------------------------ tooltip */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function TT({ active, payload, label, fmt, title }: any) {
  if (!active || !payload?.length) return null
  const head = title ?? label ?? payload[0]?.payload?.name ?? payload[0]?.name
  return (
    <div className="tt">
      {head !== undefined && <div className="tt-title">{head}</div>}
      {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
      {payload.map((p: any, i: number) => (
        <div className="tt-row" key={i}>
          <span className="legend-sw" style={{ background: p.color || p.payload?.fill || p.fill }} />
          {p.name}
          <b>{fmt ? fmt(p.value, p) : num(p.value, Number.isInteger(p.value) ? 0 : 1)}</b>
        </div>
      ))}
    </div>
  )
}

/* ------------------------------------------------------------ donut */
export type Slice = { name: string; value: number }

export function Donut({ data, order, size = 170, centerLabel, fmt = (v: number) => num(v), colors }: {
  data: Slice[]; order?: string[]; size?: number; centerLabel?: string; fmt?: (v: number) => string; colors?: string[]
}) {
  const pal = usePalette()
  const rows = useMemo(() => {
    // colour follows the slice's identity (its position in `order`, else in the input), never its rank
    const ord = order ?? (data || []).map(d => d.name)
    const clean = (data || []).filter(d => d && Number(d.value) > 0)
    clean.sort((a, b) => idx(ord, a.name) - idx(ord, b.name))
    return clean.map(d => ({
      ...d,
      value: Number(d.value),
      fill: (colors ?? pal.series)[idx(ord, d.name) % 8],
    }))
  }, [data, order, pal, colors])
  const total = rows.reduce((s, r) => s + r.value, 0)
  if (!total) return <Empty>No data for this breakdown</Empty>
  return (
    <div className="donut-wrap">
      <div style={{ width: size, height: size, position: 'relative', flex: 'none', margin: '0 auto' }}>
        <ResponsiveContainer>
          <PieChart>
            <Pie data={rows} dataKey="value" nameKey="name" innerRadius="62%" outerRadius="100%"
              stroke={pal.surface} strokeWidth={2} startAngle={90} endAngle={-270}
              animationDuration={900} isAnimationActive>
              {rows.map(r => <Cell key={r.name} fill={r.fill} />)}
            </Pie>
            <Tooltip content={<TT fmt={(v: number) => `${fmt(v)} (${((v / total) * 100).toFixed(1)}%)`} />} />
          </PieChart>
        </ResponsiveContainer>
        <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', pointerEvents: 'none', textAlign: 'center' }}>
          <div>
            <div style={{ fontWeight: 800, fontSize: size > 150 ? '1.25rem' : '1rem' }}>{fmt(total)}</div>
            {centerLabel && <div style={{ fontSize: '0.72rem', color: pal.muted }}>{centerLabel}</div>}
          </div>
        </div>
      </div>
      <div className="legend">
        {rows.map(r => (
          <div className="legend-row" key={r.name}>
            <span className="legend-sw" style={{ background: r.fill }} />
            <span className="legend-name">{r.name}</span>
            <span className="legend-val">{fmt(r.value)}</span>
            <span className="legend-pct">{((r.value / total) * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}
const idx = (order: string[], n: string) => { const i = order.indexOf(n); return i < 0 ? order.length : i }

/* ------------------------------------------------------------ gauge */
export function Gauge({ value, max = 100, size = 180, label = 'Impact', color }: {
  value: number; max?: number; size?: number; label?: string; color?: string
}) {
  const pal = usePalette()
  const r = 70, c = Math.PI * r
  const p = Math.max(0, Math.min(1, (value || 0) / max))
  const col = color ?? pal.series[0]
  return (
    <svg width={size} height={size * 0.62} viewBox="0 0 180 112" role="img" aria-label={`${label} ${num(value, 1)} of ${max}`}>
      <path d="M20 95 A70 70 0 0 1 160 95" fill="none" stroke={pal.track} strokeWidth="14" strokeLinecap="round" />
      <motion.path d="M20 95 A70 70 0 0 1 160 95" fill="none" stroke={col} strokeWidth="14" strokeLinecap="round"
        strokeDasharray={c} initial={{ strokeDashoffset: c }} animate={{ strokeDashoffset: c * (1 - p) }}
        transition={{ duration: 1.2, ease: 'easeOut' }} />
      <text x="90" y="82" textAnchor="middle" fontSize="30" fontWeight="800" fill={pal.ink}>{num(value, 1)}</text>
      <text x="90" y="104" textAnchor="middle" fontSize="11" fill={pal.muted}>{label} / {max}</text>
    </svg>
  )
}

/* ------------------------------------------------------------ player combobox */
export function PlayerSelect({ players, value, onChange, placeholder = 'Search player…', label }: {
  players: Row[]; value: string; onChange: (v: string) => void; placeholder?: string; label?: string
}) {
  const [text, setText] = useState(value)
  const [open, setOpen] = useState(false)
  const [hl, setHl] = useState(0)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => setText(value), [value])
  useEffect(() => {
    const h = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false) }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [])
  const list = useMemo(() => {
    const t = text.trim().toLowerCase()
    const base = !t || t === value.toLowerCase() ? players : players.filter(p => String(p.player).toLowerCase().includes(t))
    return base.slice(0, 60)
  }, [players, text, value])
  const pick = (p: string) => { onChange(p); setText(p); setOpen(false) }
  return (
    <div className="field combo" ref={ref}>
      {label && <label>{label}</label>}
      <input className="input" value={text} placeholder={placeholder}
        onFocus={e => { setOpen(true); e.target.select() }}
        onChange={e => { setText(e.target.value); setOpen(true); setHl(0) }}
        onKeyDown={e => {
          if (e.key === 'ArrowDown') { setHl(h => Math.min(h + 1, list.length - 1)); e.preventDefault() }
          if (e.key === 'ArrowUp') { setHl(h => Math.max(h - 1, 0)); e.preventDefault() }
          if (e.key === 'Enter' && list[hl]) pick(list[hl].player)
          if (e.key === 'Escape') setOpen(false)
        }} />
      {open && list.length > 0 && (
        <div className="combo-list">
          {list.map((p, i) => (
            <div key={p.player} className={`combo-item ${i === hl ? 'hl' : ''}`}
              onMouseDown={() => pick(p.player)} onMouseEnter={() => setHl(i)}>
              <span>{p.player}</span><small>{p.role} · {num(p.overall_impact, 1)}</small>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
