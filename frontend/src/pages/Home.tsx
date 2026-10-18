import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowRight, Activity, Database, BrainCircuit, Scale, Gauge as GaugeIcon } from 'lucide-react'
import {
  Bar, BarChart, CartesianGrid, Cell, LabelList, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from 'recharts'
import WinProbReplay from '../components/WinProbReplay'
import { ChartCard, CountUp, Donut, ErrorBox, Loading, TT } from '../components/ui'
import { prefetch, useApi, type Row } from '../lib/api'
import { CATEGORY_ORDER, ROLE_ORDER, usePalette } from '../lib/theme'
import { abbr, compact, num, pct, short } from '../lib/format'

export default function Home() {
  const { data, error } = useApi<Row>('/overview')
  const storyReq = useApi<Row>('/story')
  const story = storyReq.data
  const pal = usePalette()
  const [leader, setLeader] = useState<'top_batters' | 'top_bowlers' | 'top_allrounders'>('top_batters')
  const [pick, setPick] = useState(0)
  const t = data?.totals
  const m: Row | undefined = story?.model
  const featured: Row[] = useMemo(() => story?.featured ?? [], [story])
  // warm every featured replay so switching tabs is instant
  useEffect(() => { featured.forEach(f => prefetch(`/matches/${f.match_id}/worm`)) }, [featured])
  const rvi: Row[] = useMemo(() => story?.runs_vs_impact ?? [], [story])
  const labelled = useMemo(() => {
    const byWpa = [...rvi].sort((a, b) => b.wpa_per_inn - a.wpa_per_inn)
    const byRuns = [...rvi].sort((a, b) => b.total_runs - a.total_runs)
    return new Set([...byWpa.slice(0, 3), ...byWpa.slice(-1), ...byRuns.slice(0, 3)].map(r => r.player))
  }, [rvi])

  return (
    <div className="page">
      <section className="hero">
        <div className="hero-copy">
          <motion.div className="hero-kicker" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <Activity size={14} /> CricHeroes Impact Metric · IPL 2008 – 2025
          </motion.div>
          <motion.h1 initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            Runs tell you what happened.<br /><em>Impact</em> tells you who won it.
          </motion.h1>
          <motion.p initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            A machine-learning model prices each side's chance of winning before every one of{' '}
            {m ? <b>{num(m.deliveries)}</b> : 'the'} IPL deliveries. Every ball moves that number, and the batter
            and bowler are credited with the move. That is Win Probability Added, the backbone of this impact metric.
          </motion.p>
          <motion.div className="hero-ctas" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
            <Link className="btn" to="/leaderboard">Explore rankings <ArrowRight size={16} /></Link>
            <Link className="btn secondary" to="/compare">Compare players</Link>
            <Link className="btn secondary" to="/matches">Replay any match</Link>
          </motion.div>
        </div>

        <div className="hero-replay">
          <div className="replay-tabs" role="tablist">
            {featured.map((f, i) => (
              <button key={f.match_id} role="tab" aria-selected={pick === i} className={pick === i ? 'on' : ''} onClick={() => setPick(i)}>
                <b>{f.title}</b><span>{abbr(f.team1)} v {abbr(f.team2)}</span>
              </button>
            ))}
          </div>
          {featured[pick]
            ? <>
              <div className="replay-blurb">{featured[pick].blurb}</div>
              <WinProbReplay key={featured[pick].match_id} matchId={featured[pick].match_id} />
            </>
            : storyReq.error
              ? <div className="replay-empty replay-error">
                  <b>Match replays could not load.</b>
                  <span>{storyReq.error}. Restart the backend so it picks up the latest code, then refresh this page.</span>
                </div>
              : <div className="replay-empty"><div className="spinner" /></div>}
          <div className="replay-caption">Real model output for a real match, replayed ball by ball. Hover the chart to inspect any delivery.</div>
        </div>

        <div className="hero-stats">
          {[
            ['Deliveries priced', m?.deliveries], ['Matches analysed', t?.matches],
            ['Players rated', t?.players], ['Seasons', t?.seasons],
          ].map(([l, v]) => (
            <div className="hero-stat" key={l as string}>
              <div className="v">{v ? <CountUp to={v as number} /> : '–'}</div>
              <div className="l">{l}</div>
            </div>
          ))}
        </div>
      </section>

      <h2 className="section-title">How it works</h2>
      <div className="grid g4 steps">
        {[
          [Database, 'Ball-by-ball data', `${m ? num(m.deliveries) : '278k'} deliveries from ${m ? num(m.matches) : '1,169'} IPL matches, enriched with score, wickets, required rate, phase and venue.`],
          [BrainCircuit, 'Win probability model', `Gradient boosting reads ${m ? m.features.length : 13} match-state features and predicts the batting side's chance of winning after every ball. Built-in rules stop a wicket from ever helping the batting side.`],
          [Scale, 'Win Probability Added', 'Each ball’s change in win probability is credited to the batter and, with the sign flipped, to the bowler. High-leverage moments weigh more.'],
          [GaugeIcon, 'Player rating', 'Impact per innings and career value above a replacement-level player, weighted equally, make up 90% of a 0 to 100 rating. Player of the Match rate, captaincy, team wins and playoffs add up to 10%.'],
        ].map(([Icon, h, d], i) => {
          const I = Icon as typeof Database
          return (
            <motion.div key={h as string} className="card step" initial={{ opacity: 0, y: 12 }} whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }} transition={{ delay: i * 0.08 }}>
              <div className="step-n"><I size={16} /></div>
              <h3><span className="step-k">{i + 1}</span>{h as string}</h3>
              <p>{d as string}</p>
            </motion.div>
          )
        })}
      </div>

      <div className="grid story-grid" style={{ marginTop: 16 }}>
        <ChartCard title="Runs are not the same as impact" sub="Batters with 40+ innings. Right = more career runs. Up = more win probability added per innings."
          table={rvi.map(r => ({ player: r.player, runs: r.total_runs, innings: r.bat_innings, wpa_per_innings_pct: Number(r.wpa_per_inn.toFixed(2)) }))}>
          <ResponsiveContainer width="100%" height={340}>
            <ScatterChart margin={{ top: 26, right: 30, bottom: 8, left: -6 }}>
              <CartesianGrid stroke={pal.grid} />
              <XAxis type="number" dataKey="total_runs" name="Career runs" tickLine={false} axisLine={{ stroke: pal.axis }} tickFormatter={v => compact(v)} />
              <YAxis type="number" dataKey="wpa_per_inn" name="WPA per innings" tickLine={false} axisLine={false} tickFormatter={v => `${v > 0 ? '+' : ''}${v.toFixed(1)}%`} />
              <ReferenceLine y={0} stroke={pal.axis} strokeDasharray="4 4" />
              <Tooltip content={({ active, payload }) => active && payload?.length ? (
                <div className="tt"><div className="tt-title">{payload[0].payload.player}</div>
                  <div className="tt-row">Career runs <b>{num(payload[0].payload.total_runs)}</b></div>
                  <div className="tt-row">Innings <b>{num(payload[0].payload.bat_innings)}</b></div>
                  <div className="tt-row">Win prob added / innings <b>{payload[0].payload.wpa_per_inn > 0 ? '+' : ''}{payload[0].payload.wpa_per_inn.toFixed(2)}%</b></div></div>) : null} />
              {ROLE_ORDER.map((role, i) => (
                <Scatter key={role} name={role} data={rvi.filter(r => r.role === role)} fill={pal.series[i]} fillOpacity={0.8} stroke={pal.surface} strokeWidth={1}>
                  <LabelList dataKey="player" content={(props: Row) => labelled.has(props.value)
                    ? <text x={props.x + 8} y={props.y - 6} fontSize={11} fill={pal.ink2}>{props.value}</text> : null} />
                </Scatter>
              ))}
            </ScatterChart>
          </ResponsiveContainer>
          <div className="legend inline">{ROLE_ORDER.filter(r => rvi.some(x => x.role === r)).map(r => <div className="legend-row" key={r}><span className="legend-sw" style={{ background: pal.series[ROLE_ORDER.indexOf(r)], borderRadius: 99 }} /><span className="legend-name">{r}</span></div>)}</div>
        </ChartCard>

        <section className="card model-card">
          <div className="card-title">Model card</div>
          <div className="card-sub">Win probability model behind every number on this site</div>
          {m ? (
            <>
              <dl className="mc-list">
                <div><dt>Algorithm</dt><dd>Gradient boosting with monotonic rules</dd></div>
                <div><dt>Data</dt><dd>{num(m.matches)} matches · {num(m.deliveries)} balls</dd></div>
                <div><dt>Scoring</dt><dd>Every match predicted by a model that never saw it (5-fold)</dd></div>
              </dl>
              <div className="mc-metric">
                <div className="mc-row"><span>AUC</span><b>{m.auc.toFixed(3)}</b></div>
                <div className="mc-track"><span style={{ width: `${((m.auc - 0.5) / 0.5) * 100}%` }} /></div>
                <div className="mc-hint">0.5 is a coin flip, 1.0 is perfect</div>
              </div>
              <div className="mc-metric">
                <div className="mc-row"><span>Brier score</span><b>{m.brier.toFixed(3)}</b></div>
                <div className="mc-track"><span style={{ width: `${(1 - m.brier / 0.25) * 100}%` }} /></div>
                <div className="mc-hint">Lower is better. Always guessing 50% scores 0.250</div>
              </div>
              {m.validation && (
                <div className="mc-compare">
                  <div className="mc-compare-title">Before and after the model upgrade</div>
                  <table className="tbl">
                    <thead><tr><th>Check</th><th className="num">Old</th><th className="num">New</th></tr></thead>
                    <tbody>
                      <tr><td>AUC, same honest test</td><td className="num">{m.validation.v1.crossfit_auc.toFixed(3)}</td><td className="num"><b>{m.validation.v2.crossfit_auc.toFixed(3)}</b></td></tr>
                      <tr><td>Wickets that help the batting side</td><td className="num">{pct(m.validation.v1.per_ball.wickets_raising_batting_wp, 1)}</td><td className="num"><b>{pct(m.validation.v2.per_ball.wickets_raising_batting_wp, 1)}</b></td></tr>
                      <tr><td>Boundaries that hurt the batting side</td><td className="num">{pct(m.validation.v1.per_ball.boundaries_lowering_batting_wp, 1)}</td><td className="num"><b>{pct(m.validation.v2.per_ball.boundaries_lowering_batting_wp, 1)}</b></td></tr>
                      <tr><td>Top WPA player was Player of the Match</td><td className="num">{pct(m.validation.v1.player_of_match.top_wpa_is_pom, 0)}</td><td className="num"><b>{pct(m.validation.v2.player_of_match.top_wpa_is_pom, 0)}</b></td></tr>
                    </tbody>
                  </table>
                </div>
              )}
              <div className="mc-features">
                {m.features.map((f: string) => <span className="chip" key={f}>{f.replace(/_/g, ' ')}</span>)}
              </div>
            </>
          ) : <Loading />}
        </section>
      </div>

      <h2 className="section-title">League at a glance</h2>
      {error && <ErrorBox msg={error} />}
      {!data && !error && <Loading />}
      {data && (
        <>
          <div className="grid g3">
            <ChartCard title="Player roles" sub="Classified from batting vs bowling innings" table={data.role_split}>
              <Donut data={data.role_split} order={ROLE_ORDER} centerLabel="players" />
            </ChartCard>
            <ChartCard title="Impact categories" sub="All match performances" table={data.category_split} delay={0.05}>
              <Donut data={data.category_split} order={CATEGORY_ORDER} centerLabel="performances" fmt={v => compact(v)} />
            </ChartCard>
            <ChartCard title="Where the runs come from" sub="Career runs of every rated player" table={data.boundary_split} delay={0.1}>
              <Donut data={data.boundary_split} centerLabel="runs" fmt={v => compact(v)} />
            </ChartCard>
          </div>

          <div className="grid g2" style={{ marginTop: 16 }}>
            <ChartCard title="Impact score distribution" sub="Number of players in each overall-impact band" table={data.histogram}>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={data.histogram} margin={{ top: 8, right: 8, left: -14, bottom: 0 }}>
                  <CartesianGrid vertical={false} stroke={pal.grid} />
                  <XAxis dataKey="range" tickLine={false} axisLine={{ stroke: pal.axis }} />
                  <YAxis tickLine={false} axisLine={false} allowDecimals={false} />
                  <Tooltip cursor={{ fill: pal.track }} content={<TT />} />
                  <Bar dataKey="count" name="Players" fill={pal.series[0]} radius={[4, 4, 0, 0]} animationDuration={900} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
            <ChartCard title="Top 10 by overall impact" sub="Coloured by role" table={data.top_overall.map((r: Row) => ({ player: r.player, role: r.role, impact: r.value }))}>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={data.top_overall} layout="vertical" margin={{ top: 0, right: 30, left: 0, bottom: 0 }}>
                  <XAxis type="number" domain={[0, 100]} hide />
                  <YAxis type="category" dataKey="player" width={110} tickLine={false} axisLine={false} interval={0} tick={{ fontSize: 12 }} />
                  <Tooltip cursor={{ fill: pal.track }} content={<TT fmt={(v: number) => num(v, 2)} />} />
                  <Bar dataKey="value" name="Impact" radius={[0, 4, 4, 0]} barSize={14} animationDuration={900}
                    label={{ position: 'right', fill: pal.ink2, fontSize: 11, formatter: (v: number) => num(v, 1) }}>
                    {data.top_overall.map((r: Row) => <Cell key={r.player} fill={pal.series[ROLE_ORDER.indexOf(r.role)]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="legend inline" style={{ marginTop: 6 }}>
                {ROLE_ORDER.map((r, i) => <div className="legend-row" key={r}><span className="legend-sw" style={{ background: pal.series[i] }} /><span className="legend-name">{r}</span></div>)}
              </div>
            </ChartCard>
          </div>

          <div className="grid g2" style={{ marginTop: 16 }}>
            <ChartCard title="Matches per season" table={data.season_trend}>
              <ResponsiveContainer width="100%" height={230}>
                <BarChart data={data.season_trend} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
                  <CartesianGrid vertical={false} stroke={pal.grid} />
                  <XAxis dataKey="season" tickLine={false} axisLine={{ stroke: pal.axis }} tick={{ fontSize: 11 }} interval="preserveStartEnd" />
                  <YAxis tickLine={false} axisLine={false} />
                  <Tooltip cursor={{ fill: pal.track }} content={<TT />} />
                  <Bar dataKey="matches" name="Matches" fill={pal.series[0]} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
            <ChartCard title="Average performance impact per season" sub="Mean match impact score of all performances">
              <ResponsiveContainer width="100%" height={230}>
                <LineChart data={data.season_trend} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}>
                  <CartesianGrid vertical={false} stroke={pal.grid} />
                  <XAxis dataKey="season" tickLine={false} axisLine={{ stroke: pal.axis }} tick={{ fontSize: 11 }} interval="preserveStartEnd" />
                  <YAxis tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                  <Tooltip content={<TT fmt={(v: number) => num(v, 1)} />} />
                  <Line type="monotone" dataKey="avg_impact" name="Avg impact" stroke={pal.series[1]} strokeWidth={2}
                    dot={{ r: 4, strokeWidth: 2, stroke: pal.surface }} activeDot={{ r: 6 }} />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          <div className="grid g2" style={{ marginTop: 16 }}>
            <ChartCard title="Franchise impact" sub="Average match impact score of a franchise's players" table={data.team_impact}>
              <ResponsiveContainer width="100%" height={Math.max(260, data.team_impact.length * 22)}>
                <BarChart data={data.team_impact.map((r: Row) => ({ ...r, team: abbr(r.player_team) }))} layout="vertical" margin={{ top: 0, right: 34, left: 0, bottom: 0 }}>
                  <XAxis type="number" hide domain={[0, 'dataMax']} />
                  <YAxis type="category" dataKey="team" width={52} tickLine={false} axisLine={false} interval={0} tick={{ fontSize: 12 }} />
                  <Tooltip cursor={{ fill: pal.track }} content={({ active, payload }) => active && payload?.length ? (
                    <div className="tt"><div className="tt-title">{payload[0].payload.player_team}</div>
                      <div className="tt-row">Avg impact <b>{num(payload[0].payload.avg_impact, 1)}</b></div>
                      <div className="tt-row">Match-winning displays <b>{num(payload[0].payload.match_winners)}</b></div></div>) : null} />
                  <Bar dataKey="avg_impact" fill={pal.series[2]} radius={[0, 4, 4, 0]} barSize={12}
                    label={{ position: 'right', fill: pal.ink2, fontSize: 11, formatter: (v: number) => num(v, 1) }} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
            <ChartCard title="Specialist leaders" sub="Min 30 innings in the discipline">
              <div className="seg" style={{ marginBottom: 12 }}>
                {([['top_batters', 'Batters'], ['top_bowlers', 'Bowlers'], ['top_allrounders', 'All-rounders']] as const).map(([k, l]) => (
                  <button key={k} className={leader === k ? 'on' : ''} onClick={() => setLeader(k)}>{l}</button>
                ))}
              </div>
              <div className="table-wrap">
                <table className="tbl">
                  <thead><tr><th>#</th><th>Player</th><th className="num">Runs</th><th className="num">Wkts</th><th>Overall impact</th></tr></thead>
                  <tbody>
                    {data[leader].map((r: Row, i: number) => (
                      <tr key={r.player}>
                        <td>{i + 1}</td>
                        <td><Link className="link" to={`/player?name=${encodeURIComponent(r.player)}`}>{short(r.player, 20)}</Link></td>
                        <td className="num">{num(r.total_runs)}</td>
                        <td className="num">{num(r.total_wickets)}</td>
                        <td><div className="bar-cell"><div className="bar-track"><div className="bar-fill" style={{ width: `${r.overall_impact}%` }} /></div>{num(r.overall_impact, 1)}</div></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </ChartCard>
          </div>
        </>
      )}
    </div>
  )
}
