import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { ChartCard, Donut, ErrorBox, Kpi, Loading } from '../components/ui'
import { q, useApi, useSeasons, type Row } from '../lib/api'
import { ROLE_ORDER, usePalette } from '../lib/theme'
import { num, pct } from '../lib/format'

const ROLES = ['All', 'Batter', 'Bowler', 'All-rounder']
const COLS: { key: string; label: string; fmt: (v: unknown) => string }[] = [
  { key: 'impact_per_inn', label: 'WPA/inn', fmt: v => `${Number(v) > 0 ? '+' : ''}${num(v, 2)}%` },
  { key: 'value_above_repl', label: 'Value', fmt: v => num(v, 0) },
  { key: 'total_innings', label: 'Inns', fmt: v => num(v) },
  { key: 'total_runs', label: 'Runs', fmt: v => num(v) },
  { key: 'avg_bat_sr', label: 'SR', fmt: v => num(v, 1) },
  { key: 'true_bat_avg', label: 'Avg', fmt: v => num(v, 1) },
  { key: 'total_wickets', label: 'Wkts', fmt: v => num(v) },
  { key: 'avg_economy', label: 'Econ', fmt: v => num(v, 2) },
  { key: 'win_rate', label: 'Win %', fmt: v => pct(v) },
  { key: 'pom_awards', label: 'POM', fmt: v => num(v) },
]

export default function Leaderboard() {
  const pal = usePalette()
  const [season, setSeason] = useState('All')
  const [role, setRole] = useState('All')
  const [minInn, setMinInn] = useState(10)
  const [minRuns, setMinRuns] = useState(0)
  const [minWkts, setMinWkts] = useState(0)
  const [search, setSearch] = useState('')
  const [sort, setSort] = useState<{ key: string; desc: boolean }>({ key: 'display_impact', desc: true })
  const [page, setPage] = useState(0)

  const sl = useSeasons()
  const { data, error, loading } = useApi<Row[]>(`/players?${q({ min_inn: minInn, role_pick: role, season, min_runs: minRuns, min_wkts: minWkts })}`)

  const rows = useMemo(() => {
    const list = (data ?? []).map((r, i): Row => ({ ...r, rank: i + 1 }))
    const s = search.trim().toLowerCase()
    const f = s ? list.filter(r => String(r.player).toLowerCase().includes(s)) : list
    return [...f].sort((a, b) => {
      const va = Number(a[sort.key] ?? -Infinity), vb = Number(b[sort.key] ?? -Infinity)
      return sort.desc ? vb - va : va - vb
    })
  }, [data, search, sort])

  const roleSplit = useMemo(() => ROLE_ORDER.map(r => ({ name: r, value: (data ?? []).filter(p => p.role === r).length })), [data])
  const bands = useMemo(() => {
    const b = [['Elite (75+)', 75, 101], ['Strong (55–75)', 55, 75], ['Average (35–55)', 35, 55], ['Developing (<35)', -1, 35]] as const
    return b.map(([name, lo, hi]) => ({ name, value: (data ?? []).filter(p => p.display_impact >= lo && p.display_impact < hi).length }))
  }, [data])
  const scatter = useMemo(() => ROLE_ORDER.map(r => (data ?? []).filter(p => p.role === r).map(p => ({
    player: p.player, x: Number(p.bat_impact_per_inn ?? 0), y: Number(p.bowl_impact_per_inn ?? 0), z: Number(p.display_impact ?? 0),
  }))), [data])

  const PER = 25
  const pageRows = rows.slice(page * PER, page * PER + PER)
  const maxImpact = Math.max(1, ...rows.map(r => Number(r.display_impact) || 0))
  const th = (key: string, label: string, numCol = true) => (
    <th key={key} className={`sortable ${numCol ? 'num' : ''}`} onClick={() => { setSort(s => ({ key, desc: s.key === key ? !s.desc : true })); setPage(0) }}>
      {label}{sort.key === key ? (sort.desc ? ' ▾' : ' ▴') : ''}
    </th>
  )

  return (
    <div className="page">
      <div className="page-head">
        <h1>Impact Leaderboard</h1>
        <p>Rating = 90% Win Probability Added, split evenly between impact per innings (shrunk for small samples) and career value above a replacement-level player, plus a 10% bonus for Player of the Match rate, captaincy, team wins and playoffs. Seasons and roles use only those innings.</p>
      </div>

      <div className="filters">
        <div className="field"><label>Season</label>
          <select className="select" value={season} onChange={e => { setSeason(e.target.value); setPage(0) }}>
            <option>All</option>{sl.map(s => <option key={s}>{s}</option>)}
          </select>
        </div>
        <div className="field"><label>Role</label>
          <div className="seg">{ROLES.map(r => <button key={r} className={role === r ? 'on' : ''} onClick={() => { setRole(r); setPage(0) }}>{r}</button>)}</div>
        </div>
        <div className="field" style={{ minWidth: 110 }}><label>Min innings</label>
          <input className="input" type="number" min={0} value={minInn} onChange={e => setMinInn(Math.max(0, +e.target.value))} /></div>
        <div className="field" style={{ minWidth: 110 }}><label>Min runs</label>
          <input className="input" type="number" min={0} step={100} value={minRuns} onChange={e => setMinRuns(Math.max(0, +e.target.value))} /></div>
        <div className="field" style={{ minWidth: 110 }}><label>Min wickets</label>
          <input className="input" type="number" min={0} step={5} value={minWkts} onChange={e => setMinWkts(Math.max(0, +e.target.value))} /></div>
        <div className="field" style={{ flex: 1 }}><label>Search</label>
          <input className="input" placeholder="Player name…" value={search} onChange={e => { setSearch(e.target.value); setPage(0) }} /></div>
      </div>

      {error && <ErrorBox msg={error} />}
      {!data && loading && <Loading />}
      {data && (
        <>
          <div className="grid g4">
            <Kpi label="Players" value={num(data.length)} hint="matching filters" />
            <Kpi label="Top player" value={<span style={{ fontSize: '1.2rem' }}>{data[0]?.player ?? '–'}</span>} hint={data[0] ? `Impact ${num(data[0].display_impact, 1)}` : ''} />
            <Kpi label="Median impact" value={num(median(data.map(d => d.display_impact)), 1)} />
            <Kpi label="Total runs · wickets" value={<span style={{ fontSize: '1.3rem' }}>{num(data.reduce((s, d) => s + (d.total_runs || 0), 0))} · {num(data.reduce((s, d) => s + (d.total_wickets || 0), 0))}</span>} />
          </div>

          <div className="grid g3" style={{ marginTop: 16 }}>
            <ChartCard title="Role mix" sub="Of the filtered players" table={roleSplit}>
              <Donut data={roleSplit} order={ROLE_ORDER} centerLabel="players" />
            </ChartCard>
            <ChartCard title="Impact tiers" sub="Players per impact band" table={bands}>
              <Donut data={bands} centerLabel="players" />
            </ChartCard>
            <ChartCard title="Batting vs bowling impact" sub="Win probability added per innings, in percentage points. Bubble size = rating">
              <ResponsiveContainer width="100%" height={210}>
                <ScatterChart margin={{ top: 6, right: 10, bottom: 6, left: -18 }}>
                  <CartesianGrid stroke={pal.grid} />
                  <XAxis type="number" dataKey="x" name="Bat impact" tickLine={false} axisLine={{ stroke: pal.axis }} tick={{ fontSize: 11 }} tickFormatter={v => `${v > 0 ? '+' : ''}${v.toFixed(0)}`} />
                  <YAxis type="number" dataKey="y" name="Bowl impact" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} tickFormatter={v => `${v > 0 ? '+' : ''}${v.toFixed(0)}`} />
                  <ZAxis type="number" dataKey="z" range={[12, 90]} />
                  <Tooltip content={({ active, payload }) => active && payload?.length ? (
                    <div className="tt"><div className="tt-title">{payload[0].payload.player}</div>
                      <div className="tt-row">Batting WPA / inns <b>{num(payload[0].payload.x, 2)}%</b></div>
                      <div className="tt-row">Bowling WPA / inns <b>{num(payload[0].payload.y, 2)}%</b></div>
                      <div className="tt-row">Rating <b>{num(payload[0].payload.z, 1)}</b></div></div>) : null} />
                  {ROLE_ORDER.map((r, i) => <Scatter key={r} name={r} data={scatter[i]} fill={pal.series[i]} fillOpacity={0.75} stroke={pal.surface} strokeWidth={1} />)}
                </ScatterChart>
              </ResponsiveContainer>
              <div className="legend inline">{ROLE_ORDER.map((r, i) => <div className="legend-row" key={r}><span className="legend-sw" style={{ background: pal.series[i], borderRadius: 99 }} /><span className="legend-name">{r}</span></div>)}</div>
            </ChartCard>
          </div>

          <section className="card" style={{ marginTop: 16 }}>
            <div className="card-head"><div><div className="card-title">Rankings</div><div className="card-sub">Click a column to sort, a name to open the profile</div></div></div>
            <div className="table-wrap">
              <table className="tbl">
                <thead><tr>
                  {th('rank', '#')}<th>Player</th><th>Role</th>
                  {th('display_impact', 'Impact', false)}
                  {COLS.map(c => th(c.key, c.label))}
                </tr></thead>
                <tbody>
                  {pageRows.map(r => (
                    <tr key={r.player}>
                      <td className="num">{r.rank}</td>
                      <td><Link className="link" to={`/player?name=${encodeURIComponent(r.player)}`}>{r.player}</Link></td>
                      <td><span className="chip"><span className="legend-sw" style={{ background: pal.series[ROLE_ORDER.indexOf(r.role)] }} />{r.role}</span></td>
                      <td><div className="bar-cell"><div className="bar-track"><div className="bar-fill" style={{ width: `${(r.display_impact / maxImpact) * 100}%` }} /></div><b>{num(r.display_impact, 1)}</b></div></td>
                      {COLS.map(c => <td key={c.key} className="num">{c.key === 'avg_economy' && !r.total_balls_bowled ? '–' : c.fmt(r[c.key])}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {rows.length > PER && (
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', justifyContent: 'flex-end', marginTop: 12 }}>
                <span className="card-sub">{page * PER + 1}–{Math.min(rows.length, (page + 1) * PER)} of {rows.length}</span>
                <button className="ghost-btn" disabled={page === 0} onClick={() => setPage(p => p - 1)}>Prev</button>
                <button className="ghost-btn" disabled={(page + 1) * PER >= rows.length} onClick={() => setPage(p => p + 1)}>Next</button>
              </div>
            )}
          </section>
        </>
      )}
    </div>
  )
}

function median(a: number[]) {
  const s = a.filter(v => v != null).sort((x, y) => x - y)
  return s.length ? s[Math.floor(s.length / 2)] : 0
}
