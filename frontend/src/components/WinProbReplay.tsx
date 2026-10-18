import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Pause, Play, SkipBack, SkipForward } from 'lucide-react'
import { useApi, type Row } from '../lib/api'
import { abbr, num } from '../lib/format'

/**
 * Ball-by-ball replay of the model's win probability for one real match.
 * The line is the chasing side's chance of winning; shading shows who is ahead.
 * The five biggest single-ball swings are numbered as they happen.
 */

const W = 1000, H = 380, PL = 54, PR = 14, PT = 18, PB = 30
const DURATION = 14000 // ms for a full replay at 1x
const SPEEDS = [0.5, 1, 2, 4]

type Theme = { team1: string; team2: string; ink: string; muted: string; grid: string; surface: string }
const DARK: Theme = { team1: '#3987e5', team2: '#d95926', ink: '#f0efec', muted: '#9a988f', grid: 'rgba(255,255,255,0.08)', surface: '#0f1512' }

export default function WinProbReplay({ matchId, autoplay = true, theme = DARK, compact = false }: {
  matchId: number; autoplay?: boolean; theme?: Theme; compact?: boolean
}) {
  const { data, error } = useApi<Row>(`/matches/${matchId}/worm`)
  const [prog, setProg] = useState(autoplay ? 0 : 1)
  const [playing, setPlaying] = useState(autoplay)
  const [speed, setSpeed] = useState(1)
  const [hover, setHover] = useState<number | null>(null)
  const raf = useRef(0)
  const box = useRef<HTMLDivElement>(null)

  // restart when the match changes
  useEffect(() => { setProg(autoplay ? 0 : 1); setPlaying(autoplay); setHover(null) }, [matchId, autoplay])

  useEffect(() => {
    if (!playing || !data) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { setProg(1); setPlaying(false); return }
    let last = performance.now()
    const tick = (now: number) => {
      const dt = Math.max(0, now - last); last = now
      setProg(p => {
        const n = Math.min(1, Math.max(0, p + (dt * speed) / DURATION))
        if (n >= 1) setPlaying(false)
        return n
      })
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [playing, data, speed])

  const g = useMemo(() => {
    if (!data) return null
    const pts: Row[] = data.points
    const n = pts.length
    const x = (i: number) => PL + (i / n) * (W - PL - PR)
    const y = (wp: number) => PT + (1 - wp) * (H - PT - PB)
    const xy = pts.map(p => [x(p.i), y(p.wp)] as [number, number])
    xy.push([x(n), y(data.final_wp)])
    return { pts, n, x, y, xy, mid: y(0.5) }
  }, [data])

  if (error) return <div className="replay-empty">Replay unavailable: {error}</div>
  if (!data || !g) return <div className="replay-empty"><div className="spinner" /></div>

  const t1 = abbr(data.team1), t2 = abbr(data.team2)
  const k = Math.max(0, Math.min(g.n - 1, Math.floor(prog * g.n))) // last revealed ball index
  const done = prog >= 1
  const shown = g.xy.slice(0, Math.min(k + 1, g.xy.length) + (done ? 1 : 0))
  const line = shown.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join('')
  const last = shown[shown.length - 1]
  const area = shown.length > 1 ? `${line}L${last[0]},${g.mid}L${shown[0][0]},${g.mid}Z` : ''
  const cur: Row = done ? { ...g.pts[g.n - 1], wp: data.final_wp } : g.pts[Math.min(k, g.n - 1)]
  const focus: Row = hover !== null ? g.pts[hover] : cur
  const wp2 = focus.wp, wp1 = 1 - focus.wp
  const moments: Row[] = data.moments
  const reached = (m: Row) => done || m.i <= k
  const wickets = g.pts.filter(p => p.ev === 'W' && (done || p.i <= k))
  const brkX = g.x(data.innings_break)
  const maxWpa = Math.max(0.01, ...data.top_wpa.map((r: Row) => r.wpa))

  const onMove = (e: React.MouseEvent) => {
    const r = box.current?.getBoundingClientRect()
    if (!r) return
    const vx = ((e.clientX - r.left) / r.width) * W
    const i = Math.round(((vx - PL) / (W - PL - PR)) * g.n)
    setHover(i >= 0 && i < g.n && (done || i <= k) ? i : null)
  }
  const ballAt = (e: React.MouseEvent) => {
    const r = box.current?.getBoundingClientRect()
    if (!r) return null
    const vx = ((e.clientX - r.left) / r.width) * W
    return Math.max(0, Math.min(g.n, Math.round(((vx - PL) / (W - PL - PR)) * g.n)))
  }
  // seek to a ball index (n = the final result)
  const seek = (i: number) => { setHover(null); setProg(i >= g.n ? 1 : i / g.n) }
  const togglePlay = () => {
    if (done) { setProg(0); setPlaying(true) } else setPlaying(p => !p)
  }
  const pos = done ? g.n : k
  const posLabel = pos >= g.n ? 'Result' : `${g.pts[pos].inn === 1 ? '1st inns' : 'Chase'} ${g.pts[pos].over}`

  return (
    <div className={`replay ${compact ? 'compact' : ''}`}>
      <div className="replay-main">
        <div className="replay-board">
          <TeamOdds name={data.team1} short={t1} wp={wp1} color={theme.team1} score={focus.inn === 1 ? focus.score : data.score1} batting={focus.inn === 1} />
          <div className="replay-mid">
            <div className="replay-over">{done && hover === null ? 'Result' : `${focus.inn === 1 ? '1st inns' : 'Chase'} · over ${focus.over}`}</div>
            <div className="replay-bar"><span style={{ width: `${wp1 * 100}%`, background: theme.team1 }} /><span style={{ width: `${wp2 * 100}%`, background: theme.team2 }} /></div>
            <div className="replay-sub">Win probability</div>
          </div>
          <TeamOdds name={data.team2} short={t2} wp={wp2} color={theme.team2} score={focus.inn === 2 ? focus.score : (k >= data.innings_break || done ? data.score2 : 'yet to bat')} batting={focus.inn === 2} right />
        </div>

        <div className="replay-chart" ref={box} onMouseMove={onMove} onMouseLeave={() => setHover(null)}
          onClick={e => { const i = ballAt(e); if (i !== null) seek(i) }} title="Click to jump to this ball">
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Win probability through ${data.team1} versus ${data.team2}`}>
            <defs>
              <clipPath id={`top${matchId}`}><rect x={0} y={0} width={W} height={g.mid} /></clipPath>
              <clipPath id={`bot${matchId}`}><rect x={0} y={g.mid} width={W} height={H} /></clipPath>
            </defs>
            {[0, 0.25, 0.5, 0.75, 1].map(v => (
              <g key={v}>
                <line x1={PL} x2={W - PR} y1={g.y(v)} y2={g.y(v)} stroke={v === 0.5 ? theme.muted : theme.grid} strokeDasharray={v === 0.5 ? '4 4' : undefined} />
                <text x={PL - 8} y={g.y(v) + 4} textAnchor="end" fontSize={12} fill={theme.muted}>{`${Math.round(Math.max(v, 1 - v) * 100)}%`}</text>
              </g>
            ))}
            <text x={PL + 6} y={PT + 12} fontSize={12} fontWeight={700} fill={theme.team2}>{t2} favoured ↑</text>
            <text x={PL + 6} y={H - PB - 6} fontSize={12} fontWeight={700} fill={theme.team1}>{t1} favoured ↓</text>
            <line x1={brkX} x2={brkX} y1={PT} y2={H - PB} stroke={theme.muted} strokeDasharray="2 4" />
            <text x={(PL + brkX) / 2} y={H - 10} textAnchor="middle" fontSize={12} fill={theme.muted}>{t1} bat · {data.score1}</text>
            <text x={(brkX + W - PR) / 2} y={H - 10} textAnchor="middle" fontSize={12} fill={theme.muted}>{t2} chase · {data.score2}</text>

            {area && <>
              <path d={area} fill={theme.team2} opacity={0.28} clipPath={`url(#top${matchId})`} />
              <path d={area} fill={theme.team1} opacity={0.28} clipPath={`url(#bot${matchId})`} />
            </>}
            <path d={line} fill="none" stroke={theme.ink} strokeWidth={2.2} strokeLinejoin="round" />
            {wickets.map(p => <circle key={p.i} cx={g.x(p.i)} cy={H - PB + 1} r={3} fill={theme.ink} opacity={0.7} />)}
            {moments.filter(reached).map((m, idx) => {
              const mx = g.x(m.i), my = g.y(m.wp)
              const col = m.swing > 0 ? theme.team2 : theme.team1
              return (
                <g key={m.i} className="replay-pop">
                  <circle cx={mx} cy={my} r={11} fill={col} stroke={theme.surface} strokeWidth={2.5} />
                  <text x={mx} y={my + 4} textAnchor="middle" fontSize={11} fontWeight={800} fill="#fff">{moments.indexOf(m) + 1 || idx + 1}</text>
                </g>
              )
            })}
            {!done && last && <circle cx={last[0]} cy={last[1]} r={5} fill={theme.ink}><animate attributeName="r" values="5;8;5" dur="0.8s" repeatCount="indefinite" /></circle>}
            {hover !== null && <>
              <line x1={g.x(hover)} x2={g.x(hover)} y1={PT} y2={H - PB} stroke={theme.ink} opacity={0.4} />
              <circle cx={g.x(hover)} cy={g.y(g.pts[hover].wp)} r={5} fill={theme.ink} stroke={theme.surface} strokeWidth={2} />
            </>}
          </svg>
        </div>

        <div className="replay-ticker">
          {focus.ev && !(done && hover === null) && <span className={`ev ev-${focus.ev === 'W' ? 'w' : 'b'}`}>{focus.ev === 'W' ? 'WICKET' : focus.ev}</span>}
          <span className="tick-text">{done && hover === null ? `${data.winner} ${data.margin}` : `${focus.over} · ${focus.text}`}</span>
        </div>

        <div className="replay-controls">
          <div className="ctl-buttons">
            <button className="icon-btn" onClick={() => { setPlaying(false); seek(0) }} aria-label="Go to first ball" title="Go to first ball"><SkipBack size={15} /></button>
            <button className="icon-btn play" onClick={togglePlay} aria-label={playing ? 'Pause' : done ? 'Replay from start' : 'Play'} title={playing ? 'Pause (Space)' : 'Play (Space)'}>
              {playing ? <Pause size={16} /> : <Play size={16} />}
            </button>
            <button className="icon-btn" onClick={() => { setPlaying(false); seek(g.n) }} aria-label="Jump to result" title="Jump to result"><SkipForward size={15} /></button>
          </div>
          <div className="ctl-seek">
            <div className="seek-track">
            <div className="seek-marks" aria-hidden>
              <span className="mark brk" style={{ left: `${(data.innings_break / g.n) * 100}%` }} />
              {moments.map((m, i) => (
                <span key={m.i} className="mark mom" style={{ left: `${(m.i / g.n) * 100}%`, background: m.swing > 0 ? theme.team2 : theme.team1 }} title={`${i + 1}. ${m.text}`} />
              ))}
            </div>
            <input type="range" min={0} max={g.n} step={1} value={pos}
              style={{ ['--fill' as string]: `${(pos / g.n) * 100}%` }}
              onChange={e => seek(Number(e.target.value))}
              onKeyDown={e => { if (e.key === ' ') { e.preventDefault(); togglePlay() } }}
              aria-label="Seek to ball" aria-valuetext={posLabel} />
            </div>
            <div className="seek-label">{posLabel}</div>
          </div>
          <div className="seg speed" role="group" aria-label="Playback speed">
            {SPEEDS.map(s => <button key={s} className={speed === s ? 'on' : ''} onClick={() => setSpeed(s)} aria-pressed={speed === s}>{s}x</button>)}
          </div>
        </div>
      </div>

      <aside className="replay-side">
        <div className="side-block">
          <div className="side-title">Turning points</div>
          <ol className="moments">
            {moments.map((m, i) => (
              <li key={m.i} className={reached(m) ? 'on' : ''} onMouseEnter={() => reached(m) && setHover(m.i)} onMouseLeave={() => setHover(null)}>
                <span className="m-num" style={{ background: m.swing > 0 ? theme.team2 : theme.team1 }}>{i + 1}</span>
                <span className="m-body">
                  <b>{m.inn === 1 ? '1st' : 'Chase'} {m.over}</b> {m.text}
                  <em>{m.swing > 0 ? t2 : t1} +{Math.round(Math.abs(m.swing) * 100)}%</em>
                </span>
              </li>
            ))}
          </ol>
        </div>
        <div className="side-block">
          <div className="side-title">Who won it · win probability added</div>
          {data.top_wpa.map((r: Row) => (
            <Link to={`/player?name=${encodeURIComponent(r.player)}`} className="wpa-row" key={r.player}>
              <span className="wpa-name">{r.player}<small>{abbr(r.team)}</small></span>
              <span className="wpa-track"><span style={{ width: `${(r.wpa / maxWpa) * 100}%`, background: r.team === data.team1 ? theme.team1 : theme.team2 }} /></span>
              <span className="wpa-val">+{num(r.wpa * 100, 0)}%</span>
            </Link>
          ))}
          {data.player_of_match && <div className="pom">Official Player of the Match: <b>{data.player_of_match}</b></div>}
        </div>
      </aside>
    </div>
  )
}

function TeamOdds({ name, short, wp, color, score, batting, right }: {
  name: string; short: string; wp: number; color: string; score: string; batting: boolean; right?: boolean
}) {
  return (
    <div className={`team-odds ${right ? 'right' : ''}`}>
      <div className="team-name"><span className="team-dot" style={{ background: color }} />{short}{batting && <span className="bat-badge">batting</span>}</div>
      <div className="team-wp">{Math.round(wp * 100)}%</div>
      <div className="team-score" title={name}>{score}</div>
    </div>
  )
}
