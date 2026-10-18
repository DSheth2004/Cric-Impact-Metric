import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import MatchView from '../components/MatchView'
import WinProbReplay from '../components/WinProbReplay'
import { ErrorBox, Loading } from '../components/ui'
import { q, useApi, useSeasons, type Row } from '../lib/api'

export default function Matches() {
  const [params, setParams] = useSearchParams()
  const seasons = useSeasons()
  const [season, setSeason] = useState(params.get('season') || '')
  const [team, setTeam] = useState('All')
  useEffect(() => { if (!season && seasons.length) setSeason(seasons[seasons.length - 1]) }, [season, seasons])

  const list = useApi<Row[]>(season ? `/matches?${q({ season, team: 'All' })}` : null)
  const teams = useMemo(() => [...new Set((list.data ?? []).flatMap(m => m.teams as string[]))].sort(), [list.data])
  const matches = useMemo(() => (list.data ?? []).filter(m => team === 'All' || (m.teams as string[]).includes(team)), [list.data, team])
  const id = params.get('id') || (matches[0] ? String(matches[0].match_id) : '')
  const detail = useApi<Row>(id ? `/matches/${id}` : null)

  return (
    <div className="page">
      <div className="page-head">
        <h1>Match Explorer</h1>
        <p>Pick any IPL match to see who swung it, how impact split across teams, and an AI explanation of the key performances.</p>
      </div>
      <div className="filters">
        <div className="field"><label>Season</label>
          <select className="select" value={season} onChange={e => { setSeason(e.target.value); setTeam('All'); setParams({}) }}>
            {seasons.map(s => <option key={s}>{s}</option>)}
          </select>
        </div>
        <div className="field"><label>Team</label>
          <select className="select" value={team} onChange={e => { setTeam(e.target.value); setParams({}) }}>
            <option>All</option>{teams.map(t => <option key={t}>{t}</option>)}
          </select>
        </div>
        <div className="field" style={{ flex: 1, minWidth: 260 }}><label>Match ({matches.length})</label>
          <select className="select" value={id} onChange={e => setParams({ id: e.target.value })}>
            {matches.map(m => <option key={m.match_id} value={m.match_id}>{m.label}</option>)}
          </select>
        </div>
      </div>
      {(list.error || detail.error) && <ErrorBox msg={(list.error || detail.error)!} />}
      {(list.loading || detail.loading) && !detail.data && <Loading />}
      {id && (
        <section className="card" style={{ marginBottom: 16 }}>
          <div className="card-head"><div>
            <div className="card-title">Win probability, ball by ball</div>
            <div className="card-sub">The model's chance of each side winning after every delivery. Press replay to watch the match unfold.</div>
          </div></div>
          <WinProbReplay key={id} matchId={Number(id)} autoplay={false} />
        </section>
      )}
      {detail.data && <MatchView key={id} m={detail.data} />}
    </div>
  )
}
