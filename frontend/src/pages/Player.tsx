import { useSearchParams, Link } from 'react-router-dom'
import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { ChartCard, Donut, Empty, ErrorBox, Gauge, Kpi, Loading, PlayerSelect, TT } from '../components/ui'
import { useApi, usePlayerList, type Row } from '../lib/api'
import { CATEGORY_ORDER, ROLE_ORDER, usePalette } from '../lib/theme'
import { abbr, num, pct, short } from '../lib/format'

export default function Player() {
  const [params, setParams] = useSearchParams()
  const name = params.get('name') || 'V Kohli'
  const players = usePlayerList()
  const { data, error, loading } = useApi<Row>(`/players/${encodeURIComponent(name)}`)
  const pal = usePalette()
  const p = data?.player
  const b = data?.breakdowns
  const bowls = (p?.total_balls_bowled ?? 0) > 0
  const captain = (p?.matches_as_captain ?? 0) > 0

  return (
    <div className="page">
      <div className="page-head" style={{ display: 'flex', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <div><h1>Player Profile</h1><p>Every component of a player's impact, broken down.</p></div>
        <PlayerSelect players={players} value={name} onChange={v => setParams({ name: v })} label="Player" />
      </div>

      {error && <ErrorBox msg={error} />}
      {loading && !data && <Loading />}
      {data && p && (
        <>
          <section className="card profile-head">
            <Gauge value={p.overall_impact} label="Overall impact" color={pal.series[ROLE_ORDER.indexOf(p.role)]} size={210} />
            <div>
              <div className="profile-name">{p.player}</div>
              <div className="profile-meta">
                <span className="chip"><span className="legend-sw" style={{ background: pal.series[ROLE_ORDER.indexOf(p.role)] }} />{p.role}</span>
                <span className="chip">Rank #{data.rank} of {data.total_players}</span>
                <span className="chip">{num(p.total_innings)} innings</span>
                {data.teams?.slice(0, 4).map((t: Row) => <span className="chip" key={t.name}>{abbr(t.name)}</span>)}
              </div>
              <div style={{ marginTop: 14 }}>
                <Link className="btn secondary" to={`/compare?a=${encodeURIComponent(p.player)}`}>Compare with another player</Link>
              </div>
            </div>
          </section>

          <div className="grid g4" style={{ marginTop: 16 }}>
            <Kpi label="Runs" value={num(p.total_runs)} hint={`Avg ${num(p.true_bat_avg, 1)} · SR ${num(p.avg_bat_sr, 1)}`} color={pal.series[0]} />
            <Kpi label="Wickets" value={num(p.total_wickets)} hint={bowls ? `Econ ${num(p.avg_economy, 2)} · Dot ${num(p.dot_pct_bowled, 1)}%` : 'Does not bowl'} color={pal.series[1]} />
            <Kpi label="Win rate" value={pct(p.win_rate)} hint={`${num(p.total_wins)} wins`} color={pal.series[2]} />
            <Kpi label="Player of the Match" value={num(p.pom_awards)} hint={captain ? `Captain in ${num(p.matches_as_captain)} matches` : 'Never captained'} color={pal.series[3]} />
          </div>

          <h2 className="section-title">Batting</h2>
          <div className="grid g3">
            <ChartCard title="Run composition" sub="Runs from boundaries vs running" table={b.run_composition}>
              <Donut data={b.run_composition} centerLabel="runs" />
            </ChartCard>
            <ChartCard title="Balls faced" sub="Dot balls vs scoring shots" table={b.balls_faced}>
              <Donut data={b.balls_faced} centerLabel="balls" />
            </ChartCard>
            <ChartCard title="Dismissals" sub="Innings ended out vs not out" table={b.dismissals}>
              <Donut data={b.dismissals} centerLabel="innings" />
            </ChartCard>
          </div>

          <h2 className="section-title">Bowling</h2>
          {bowls ? (
            <div className="grid g3">
              <ChartCard title="Balls by phase" sub="Powerplay (1-6), middle, death (16-20)" table={b.bowling_phases}>
                <Donut data={b.bowling_phases} centerLabel="balls" />
              </ChartCard>
              <ChartCard title="Wickets by phase" table={b.wicket_phases}>
                <Donut data={b.wicket_phases} centerLabel="wickets" />
              </ChartCard>
              <ChartCard title="Balls bowled" sub="Dot balls vs balls scored off" table={b.balls_bowled}>
                <Donut data={b.balls_bowled} centerLabel="balls" />
              </ChartCard>
            </div>
          ) : <div className="card"><Empty>{p.player} has not bowled in the IPL.</Empty></div>}

          <h2 className="section-title">Impact & results</h2>
          <div className="grid g3">
            <ChartCard title="Skill percentiles" sub="Percentile vs all rated players (economy inverted)" table={data.radar.map((r: Row) => ({ metric: r.metric, percentile: r.p0 }))}>
              <ResponsiveContainer width="100%" height={250}>
                <RadarChart data={data.radar} outerRadius="72%">
                  <PolarGrid stroke={pal.grid} />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
                  <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                  <Radar dataKey="p0" name={p.player} stroke={pal.series[0]} fill={pal.series[0]} fillOpacity={0.3} strokeWidth={2} animationDuration={900} />
                  <Tooltip content={<TT fmt={(v: number) => `${num(v)}th pct`} />} />
                </RadarChart>
              </ResponsiveContainer>
            </ChartCard>
            <ChartCard title="Match impact categories" sub="How each performance was rated" table={data.categories}>
              <Donut data={data.categories} order={CATEGORY_ORDER} centerLabel="performances" />
            </ChartCard>
            <ChartCard title="Results" sub={captain ? 'Team results, and as captain' : 'Team results in matches played'} table={captain ? [...b.results, ...b.captaincy.map((r: Row) => ({ ...r, name: `Captain: ${r.name}` }))] : b.results}>
              <Donut data={b.results} centerLabel="matches" size={captain ? 130 : 170} />
              {captain && <div style={{ marginTop: 14 }}><Donut data={b.captaincy} centerLabel="as captain" size={130} colors={[pal.series[2], pal.series[3]]} /></div>}
            </ChartCard>
          </div>

          <div className="grid g2" style={{ marginTop: 16 }}>
            <ChartCard title="Recent innings impact" sub={`Last 15 ${p.role === 'Bowler' ? 'bowling' : 'batting'} innings, impact score per innings`} table={data.trend.map((r: Row) => ({ date: r.start_date, opposition: r.opposition, impact: r.impact_score }))}>
              {data.trend.length ? (
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart data={data.trend} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}>
                    <CartesianGrid vertical={false} stroke={pal.grid} />
                    <XAxis dataKey="start_date" tickLine={false} axisLine={{ stroke: pal.axis }} tick={{ fontSize: 10 }} tickFormatter={d => String(d).slice(2, 7)} />
                    <YAxis domain={[0, 100]} tickLine={false} axisLine={false} />
                    <Tooltip content={({ active, payload }) => active && payload?.length ? (
                      <div className="tt"><div className="tt-title">{payload[0].payload.start_date} vs {abbr(payload[0].payload.opposition)}</div>
                        <div className="tt-row">Impact <b>{num(payload[0].payload.impact_score, 1)}</b></div>
                        <div className="tt-row">{payload[0].payload.wickets !== undefined ? `${payload[0].payload.wickets}/${payload[0].payload.runs_conceded}` : `${payload[0].payload.runs} runs (${payload[0].payload.balls_faced} balls)`}</div></div>) : null} />
                    <Line type="monotone" dataKey="impact_score" name="Impact" stroke={pal.series[0]} strokeWidth={2} dot={{ r: 4, stroke: pal.surface, strokeWidth: 2 }} activeDot={{ r: 6 }} />
                  </LineChart>
                </ResponsiveContainer>
              ) : <Empty>No innings data</Empty>}
            </ChartCard>
            <ChartCard title="Impact by season" sub="Average match impact score per season" table={data.season_impact}>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={data.season_impact} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
                  <CartesianGrid vertical={false} stroke={pal.grid} />
                  <XAxis dataKey="season" tickLine={false} axisLine={{ stroke: pal.axis }} tick={{ fontSize: 10 }} />
                  <YAxis domain={[0, 100]} tickLine={false} axisLine={false} />
                  <Tooltip cursor={{ fill: pal.track }} content={<TT fmt={(v: number) => num(v, 1)} />} />
                  <Bar dataKey="avg_impact" name="Avg impact" fill={pal.series[0]} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          <div className="grid g2" style={{ marginTop: 16 }}>
            <ChartCard title="Batting partners" sub="Total partnership runs together" table={data.partners}>
              {data.partners.length ? <HBar data={data.partners} y="partner" x="runs" color={pal.series[2]} label="Runs" /> : <Empty>No partnership data</Empty>}
            </ChartCard>
            <ChartCard title="Bowlers dominated" sub="Highest dominance index, min 12 balls faced" table={data.dominated_bowlers.map((r: Row) => ({ bowler: r.bowler_name, balls: r.balls, runs: r.runs, outs: r.dismissals, sr: r.sr }))}>
              {data.dominated_bowlers.length ? <HBar data={data.dominated_bowlers.map((r: Row) => ({ ...r, sr: Number(r.sr) }))} y="bowler_name" x="sr" color={pal.series[0]} label="Strike rate" /> : <Empty>Not enough head-to-head data</Empty>}
            </ChartCard>
          </div>

          <div className="grid g2" style={{ marginTop: 16 }}>
            <ChartCard title="Nemesis bowlers" sub="Bowlers who dismissed this batter most" table={data.nemesis_bowlers.map((r: Row) => ({ bowler: r.bowler_name, balls: r.balls, runs: r.runs, outs: r.dismissals }))}>
              {data.nemesis_bowlers.length ? <HBar data={data.nemesis_bowlers} y="bowler_name" x="dismissals" color={pal.series[7]} label="Dismissals" /> : <Empty>Not enough head-to-head data</Empty>}
            </ChartCard>
            <ChartCard title="Best match performances" table={data.best_matches}>
              <div className="table-wrap">
                <table className="tbl">
                  <thead><tr><th>Season</th><th>vs</th><th>Key stat</th><th className="num">Impact</th></tr></thead>
                  <tbody>{data.best_matches.map((m: Row, i: number) => (
                    <tr key={i}><td>{m.season}</td><td>{abbr(m.opposition)}</td><td>{m.key_stat}</td><td className="num"><b>{num(m.impact_score, 1)}</b></td></tr>
                  ))}</tbody>
                </table>
              </div>
            </ChartCard>
          </div>
          {bowls && data.dominated_batters.length > 0 && (
            <div className="grid g2" style={{ marginTop: 16 }}>
              <ChartCard title="Batters dismissed most" sub="As a bowler, min 12 balls" table={data.dominated_batters.map((r: Row) => ({ batter: r.batter, balls: r.balls, runs: r.runs, outs: r.dismissals }))}>
                <HBar data={data.dominated_batters} y="batter" x="dismissals" color={pal.series[1]} label="Dismissals" />
              </ChartCard>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export function HBar({ data, x, y, color, label }: { data: Row[]; x: string; y: string; color: string; label: string }) {
  const pal = usePalette()
  return (
    <ResponsiveContainer width="100%" height={Math.max(170, data.length * 30)}>
      <BarChart data={data.map(d => ({ ...d, _y: short(String(d[y]), 16) }))} layout="vertical" margin={{ top: 0, right: 36, left: 0, bottom: 0 }}>
        <XAxis type="number" hide />
        <YAxis type="category" dataKey="_y" width={112} tickLine={false} axisLine={false} interval={0} tick={{ fontSize: 12 }} />
        <Tooltip cursor={{ fill: pal.track }} content={<TT fmt={(v: number) => num(v, Number.isInteger(v) ? 0 : 1)} />} />
        <Bar dataKey={x} name={label} radius={[0, 4, 4, 0]} barSize={14} animationDuration={800}
          label={{ position: 'right', fill: pal.ink2, fontSize: 11, formatter: (v: number) => num(v, Number.isInteger(v) ? 0 : 1) }}>
          {data.map((_, i) => <Cell key={i} fill={color} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
