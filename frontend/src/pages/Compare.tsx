import { useSearchParams } from 'react-router-dom'
import { ArrowLeftRight } from 'lucide-react'
import {
  CartesianGrid, Legend, Line, LineChart, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { ChartCard, Donut, Empty, ErrorBox, Gauge, Loading, PlayerSelect, TT } from '../components/ui'
import { q, useApi, usePlayerList, type Row } from '../lib/api'
import { usePalette } from '../lib/theme'
import { num, pct } from '../lib/format'

const LOWER_BETTER = new Set(['avg_economy'])
const PCT_KEYS = new Set(['win_rate'])

export default function Compare() {
  const [params, setParams] = useSearchParams()
  const a = params.get('a') || 'V Kohli'
  const b = params.get('b') || (a === 'RG Sharma' ? 'V Kohli' : 'RG Sharma')
  const players = usePlayerList()
  const same = a === b
  const { data, error, loading } = useApi<Row>(same ? null : `/compare?${q({ a, b })}`)
  const pal = usePalette()
  const ca = pal.series[0], cb = pal.series[1]
  const set = (k: 'a' | 'b', v: string) => setParams({ a: k === 'a' ? v : a, b: k === 'b' ? v : b })

  return (
    <div className="page">
      <div className="page-head">
        <h1>Player Comparison</h1>
        <p>Pick any two players to compare their impact, skills and head-to-head record.</p>
      </div>
      <div className="filters">
        <PlayerSelect players={players} value={a} onChange={v => set('a', v)} label="Player A" />
        <button className="icon-btn" style={{ marginBottom: 1 }} title="Swap players" aria-label="Swap players" onClick={() => setParams({ a: b, b: a })}><ArrowLeftRight size={16} /></button>
        <PlayerSelect players={players} value={b} onChange={v => set('b', v)} label="Player B" />
      </div>

      {same && <div className="notice">Pick two different players.</div>}
      {error && <ErrorBox msg={error} />}
      {loading && !data && <Loading />}
      {data && !same && (
        <>
          <section className="card vs-banner">
            <Side row={data.a} color={ca} />
            <div className="vs-mid">VS</div>
            <Side row={data.b} color={cb} />
          </section>

          <div className="grid g2" style={{ marginTop: 16 }}>
            <ChartCard title="Skill radar" sub="Percentile vs all rated players (economy inverted)"
              table={data.radar.map((r: Row) => ({ metric: r.metric, [a]: r.p0, [b]: r.p1 }))}>
              <ResponsiveContainer width="100%" height={320}>
                <RadarChart data={data.radar} outerRadius="72%">
                  <PolarGrid stroke={pal.grid} />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
                  <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                  <Radar dataKey="p0" name={a} stroke={ca} fill={ca} fillOpacity={0.25} strokeWidth={2} />
                  <Radar dataKey="p1" name={b} stroke={cb} fill={cb} fillOpacity={0.25} strokeWidth={2} />
                  <Legend wrapperStyle={{ fontSize: 13 }} />
                  <Tooltip content={<TT fmt={(v: number) => `${num(v)}th pct`} />} />
                </RadarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Stat by stat" sub="Brighter value wins the row" table={data.stats.map((s: Row) => ({ stat: s.label, [a]: s.a, [b]: s.b }))}>
              <div className="duel">
                <div className="duel-row" style={{ fontWeight: 700, fontSize: '0.8rem' }}>
                  <span style={{ textAlign: 'right' }}>{a} <span className="kpi-dot" style={{ background: ca, margin: 0 }} /></span><span /><span><span className="kpi-dot" style={{ background: cb }} />{b}</span>
                </div>
                {data.stats.map((s: Row) => {
                  const max = Math.max(Math.abs(s.a), Math.abs(s.b), 1e-9)
                  const better = LOWER_BETTER.has(s.key) ? (s.a === 0 ? 'b' : s.b === 0 ? 'a' : s.a < s.b ? 'a' : 'b') : s.a > s.b ? 'a' : 'b'
                  const tie = s.a === s.b
                  const f = (v: number) => PCT_KEYS.has(s.key) ? pct(v) : num(v, Number.isInteger(v) ? 0 : 1)
                  return (
                    <div className="duel-row" key={s.key}>
                      <div className="duel-side left">
                        <div className="duel-track"><div className="duel-bar" style={{ width: `${(Math.abs(s.a) / max) * 100}%`, background: ca, opacity: tie || better === 'a' ? 1 : 0.45 }} /></div>
                        <span className={`duel-val ${tie || better === 'a' ? 'win' : 'lose'}`}>{f(s.a)}</span>
                      </div>
                      <div className="duel-label">{s.label}</div>
                      <div className="duel-side">
                        <div className="duel-track"><div className="duel-bar" style={{ width: `${(Math.abs(s.b) / max) * 100}%`, background: cb, opacity: tie || better === 'b' ? 1 : 0.45 }} /></div>
                        <span className={`duel-val ${tie || better === 'b' ? 'win' : 'lose'}`}>{f(s.b)}</span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </ChartCard>
          </div>

          <h2 className="section-title">Side-by-side breakdowns</h2>
          {[
            ['Run composition', 'run_composition', 'runs'],
            ['Balls faced', 'balls_faced', 'balls'],
            ['Wickets by phase', 'wicket_phases', 'wickets'],
            ['Team results', 'results', 'matches'],
          ].map(([title, key, unit]) => (
            <div className="grid g2" style={{ marginBottom: 16 }} key={key}>
              <ChartCard title={`${title} · ${a}`} table={data.breakdowns_a[key]}>
                <Donut data={data.breakdowns_a[key]} centerLabel={unit} size={150} />
              </ChartCard>
              <ChartCard title={`${title} · ${b}`} table={data.breakdowns_b[key]}>
                <Donut data={data.breakdowns_b[key]} centerLabel={unit} size={150} />
              </ChartCard>
            </div>
          ))}

          <div className="grid g2">
            <ChartCard title="Impact by season" sub="Average match impact score" table={data.season_trend}>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={data.season_trend} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}>
                  <CartesianGrid vertical={false} stroke={pal.grid} />
                  <XAxis dataKey="season" tickLine={false} axisLine={{ stroke: pal.axis }} tick={{ fontSize: 10 }} />
                  <YAxis domain={[0, 100]} tickLine={false} axisLine={false} />
                  <Tooltip content={<TT fmt={(v: number) => num(v, 1)} />} />
                  <Legend wrapperStyle={{ fontSize: 13 }} />
                  <Line dataKey="a" name={a} stroke={ca} strokeWidth={2} dot={{ r: 3.5, stroke: pal.surface, strokeWidth: 2 }} connectNulls />
                  <Line dataKey="b" name={b} stroke={cb} strokeWidth={2} dot={{ r: 3.5, stroke: pal.surface, strokeWidth: 2 }} connectNulls strokeDasharray="5 3" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
            <ChartCard title="Head-to-head" sub="When one bowled to the other">
              {data.head_to_head.length ? (
                <div className="grid" style={{ gap: 14 }}>
                  {data.head_to_head.map((h: Row) => (
                    <div key={h.batter} className="card" style={{ boxShadow: 'none', background: 'var(--surface-2)' }}>
                      <div style={{ fontWeight: 700, marginBottom: 8 }}>{h.batter} <span style={{ color: 'var(--muted)', fontWeight: 500 }}>batting vs</span> {h.bowler_name}</div>
                      <div className="grid g4" style={{ gap: 8 }}>
                        {[['Balls', h.balls], ['Runs', h.runs], ['Outs', h.dismissals], ['SR', num(h.sr, 1)]].map(([l, v]) => (
                          <div key={l as string}><div className="kpi-label">{l}</div><div style={{ fontWeight: 800, fontSize: '1.3rem' }}>{v}</div></div>
                        ))}
                      </div>
                      <div style={{ marginTop: 10 }}>
                        <Donut size={110} data={[{ name: 'Dots', value: h.dots }, { name: 'Fours', value: h.fours }, { name: 'Sixes', value: h.sixes }, { name: 'Other scoring', value: Math.max(0, h.balls - h.dots - h.fours - h.sixes) }]} centerLabel="balls" />
                      </div>
                    </div>
                  ))}
                </div>
              ) : <Empty>These two have never faced each other as batter and bowler.</Empty>}
            </ChartCard>
          </div>
        </>
      )}
    </div>
  )
}

function Side({ row, color }: { row: Row; color: string }) {
  return (
    <div className="vs-side">
      <Gauge value={row.overall_impact} color={color} label="Impact" size={180} />
      <div className="vs-name"><span className="kpi-dot" style={{ background: color, width: 12, height: 12 }} />{row.player}</div>
      <div className="card-sub">{num(row.total_runs)} runs · {num(row.total_wickets)} wkts · {pct(row.win_rate)} wins</div>
    </div>
  )
}
