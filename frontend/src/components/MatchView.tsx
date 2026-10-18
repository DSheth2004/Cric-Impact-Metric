import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Sparkles } from 'lucide-react'
import {
  Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from 'recharts'
import { ChartCard, Donut, Kpi } from './ui'
import { apiPost, type Row } from '../lib/api'
import { CATEGORY_ORDER, usePalette } from '../lib/theme'
import { abbr, num, short } from '../lib/format'

export default function MatchView({ m }: { m: Row }) {
  const pal = usePalette()
  const teams: string[] = m.teams ?? []
  const teamColor = (t: string) => pal.series[Math.max(0, teams.indexOf(t)) % 8]
  const players: Row[] = m.players ?? []
  const top = players[0]
  const byImpact = [...players].sort((x, y) => y.impact_score - x.impact_score)
  const rows = byImpact.slice(0, 22).map((p): Row => ({ ...p, label: `${short(p.player, 14)} · ${p.role === 'batter' ? 'bat' : 'bowl'}` }))

  return (
    <>
      <div className="grid g4">
        <Kpi label="Fixture" value={<span style={{ fontSize: '1.2rem' }}>{teams.map(abbr).join(' vs ')}</span>} hint={`${m.season ?? ''}`} />
        <Kpi label="Venue" value={<span style={{ fontSize: '1rem' }}>{m.venue || '–'}</span>} />
        <Kpi label="Top performer" value={<span style={{ fontSize: '1.15rem' }}>{top?.player ?? '–'}</span>} hint={top ? `${top.key_stat} · ${num(top.impact_score, 1)}` : ''} />
        <Kpi label="Match winners" value={num(players.filter(p => p.impact_category === 'Match Winner').length)} hint={`of ${players.length} performances`} />
      </div>

      <div className="grid g3" style={{ marginTop: 16 }}>
        <ChartCard title="Impact share by team" sub="Sum of player impact scores" table={m.team_impact}>
          <Donut data={m.team_impact.map((t: Row) => ({ ...t, name: t.name }))} order={teams} fmt={v => num(v)} centerLabel="impact" />
        </ChartCard>
        <ChartCard title="Impact categories" table={m.category_split}>
          <Donut data={m.category_split} order={CATEGORY_ORDER} centerLabel="performances" />
        </ChartCard>
        <ChartCard title="Bat vs ball" sub="Impact share by discipline" table={m.role_impact}>
          <Donut data={m.role_impact.map((r: Row) => ({ ...r, name: r.name === 'batter' ? 'Batting' : 'Bowling' }))} fmt={v => num(v)} centerLabel="impact" colors={[pal.series[2], pal.series[3]]} />
        </ChartCard>
      </div>

      <div className="grid g2" style={{ marginTop: 16 }}>
        <ChartCard title="Player impact" sub="Impact score of each batting and bowling performance" table={rows.map(r => ({ player: r.player, role: r.role, team: r.player_team, stat: r.key_stat, impact: r.impact_score }))}>
          <ResponsiveContainer width="100%" height={Math.max(260, rows.length * 22)}>
            <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 36, left: 0, bottom: 0 }}>
              <XAxis type="number" domain={[0, 100]} hide />
              <YAxis type="category" dataKey="label" width={140} tickLine={false} axisLine={false} interval={0} tick={{ fontSize: 11 }} />
              <Tooltip cursor={{ fill: pal.track }} content={({ active, payload }) => active && payload?.length ? (
                <div className="tt"><div className="tt-title">{payload[0].payload.player}</div>
                  <div className="tt-row">{payload[0].payload.key_stat}</div>
                  <div className="tt-row">Impact <b>{num(payload[0].payload.impact_score, 1)}</b></div>
                  <div className="tt-row">{payload[0].payload.impact_category}</div></div>) : null} />
              <Bar dataKey="impact_score" radius={[0, 4, 4, 0]} barSize={12}
                label={{ position: 'right', fill: pal.ink2, fontSize: 10, formatter: (v: number) => num(v, 0) }}>
                {rows.map((r, i) => <Cell key={i} fill={teamColor(r.player_team)} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <TeamLegend teams={teams} color={teamColor} />
        </ChartCard>
        <ChartCard title="Win Probability Added vs clutch" sub="Each dot is a performance; right = more WPA, up = more clutch">
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart margin={{ top: 8, right: 12, bottom: 8, left: -10 }}>
              <CartesianGrid stroke={pal.grid} />
              <XAxis type="number" dataKey="wpa_score" name="WPA" tickLine={false} axisLine={{ stroke: pal.axis }} tick={{ fontSize: 11 }} tickFormatter={v => v.toFixed(2)} />
              <YAxis type="number" dataKey="clutch_score" name="Clutch" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} tickFormatter={v => v.toFixed(2)} />
              <ReferenceLine x={0} stroke={pal.axis} />
              <ReferenceLine y={0} stroke={pal.axis} />
              <Tooltip content={({ active, payload }) => active && payload?.length ? (
                <div className="tt"><div className="tt-title">{payload[0].payload.player}</div>
                  <div className="tt-row">{payload[0].payload.key_stat}</div>
                  <div className="tt-row">WPA <b>{num(payload[0].payload.wpa_score, 3)}</b></div>
                  <div className="tt-row">Clutch <b>{num(payload[0].payload.clutch_score, 3)}</b></div></div>) : null} />
              {teams.map(t => (
                <Scatter key={t} name={t} data={players.filter(p => p.player_team === t)} fill={teamColor(t)} stroke={pal.surface} strokeWidth={1.5} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
          <TeamLegend teams={teams} color={teamColor} />
        </ChartCard>
      </div>

      <AiSummary m={m} />

      <section className="card" style={{ marginTop: 16 }}>
        <div className="card-head"><div className="card-title">All performances</div></div>
        <div className="table-wrap">
          <table className="tbl">
            <thead><tr><th className="num">Rank</th><th>Player</th><th>Team</th><th>Role</th><th>Key stat</th><th>Category</th><th className="num">WPA</th><th className="num">Impact</th></tr></thead>
            <tbody>{players.map((p, i) => (
              <tr key={i}>
                <td className="num">{p.rank}</td>
                <td><Link className="link" to={`/player?name=${encodeURIComponent(p.player)}`}>{p.player}</Link></td>
                <td><span className="chip"><span className="legend-sw" style={{ background: teamColor(p.player_team) }} />{abbr(p.player_team)}</span></td>
                <td>{p.role}</td><td>{p.key_stat}</td><td>{p.impact_category}</td>
                <td className="num">{num(p.wpa_score, 3)}</td><td className="num"><b>{num(p.impact_score, 1)}</b></td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </section>
    </>
  )
}

function TeamLegend({ teams, color }: { teams: string[]; color: (t: string) => string }) {
  return (
    <div className="legend inline" style={{ marginTop: 8 }}>
      {teams.map(t => <div className="legend-row" key={t}><span className="legend-sw" style={{ background: color(t) }} /><span className="legend-name">{t}</span></div>)}
    </div>
  )
}

function AiSummary({ m }: { m: Row }) {
  const [state, setState] = useState<{ loading?: boolean; error?: string; res?: Row }>({})
  const run = async () => {
    setState({ loading: true })
    try {
      const top = (m.players as Row[]).slice(0, 11).map(p => ({
        player: p.player, team: p.player_team, role: p.role, key_stat: p.key_stat, impact: Math.round(p.impact_score), category: p.impact_category,
      }))
      const res = await apiPost<Row>('/generate_summary', JSON.stringify({ match_id: String(m.match_id), players: top }), true)
      setState({ res })
    } catch (e) { setState({ error: String((e as Error).message) }) }
  }
  return (
    <section className="card" style={{ marginTop: 16 }}>
      <div className="card-head">
        <div><div className="card-title">AI match analyst</div><div className="card-sub">Plain-language explanation of the top performers (Groq LLM)</div></div>
        <button className="btn" onClick={run} disabled={state.loading}><Sparkles size={16} />{state.loading ? 'Analysing…' : 'Generate summary'}</button>
      </div>
      {state.error && <div className="notice"><span className="error">Summary unavailable.</span> {state.error}{state.error.includes('GROQ_API_KEY') && ' Add GROQ_API_KEY=... to a .env file in the project root and restart the backend.'}</div>}
      {state.res && (state.res.parsed?.length ? (
        <div className="table-wrap"><table className="tbl">
          <thead><tr><th>Player</th><th>Role</th><th>Key stat</th><th>Why it mattered</th></tr></thead>
          <tbody>{state.res.parsed.map((r: Row, i: number) => (
            <tr key={i}><td><b>{r.Player}</b></td><td>{r.Role}</td><td>{r['Key Stat']}</td><td style={{ whiteSpace: 'normal', minWidth: 280 }}>{r['Impact Explanation']}</td></tr>
          ))}</tbody>
        </table></div>
      ) : <pre style={{ whiteSpace: 'pre-wrap', margin: 0, fontFamily: 'inherit' }}>{state.res.text}</pre>)}
    </section>
  )
}
