export const num = (v: unknown, d = 0): string => {
  const n = Number(v)
  if (v === null || v === undefined || Number.isNaN(n)) return '–'
  return n.toLocaleString('en-IN', { maximumFractionDigits: d, minimumFractionDigits: d })
}
export const pct = (v: unknown, d = 0): string => {
  const n = Number(v)
  if (v === null || v === undefined || Number.isNaN(n)) return '–'
  return `${(n * 100).toFixed(d)}%`
}
export const compact = (v: number): string =>
  v >= 1e5 ? `${(v / 1e5).toFixed(1)}L` : v >= 1e3 ? `${(v / 1e3).toFixed(1)}k` : String(Math.round(v))
export const short = (s: string, n = 18) => (s && s.length > n ? s.slice(0, n - 1) + '…' : s)

const TEAM_ABBR: Record<string, string> = {
  'Chennai Super Kings': 'CSK', 'Mumbai Indians': 'MI', 'Royal Challengers Bangalore': 'RCB',
  'Royal Challengers Bengaluru': 'RCB', 'Kolkata Knight Riders': 'KKR', 'Sunrisers Hyderabad': 'SRH',
  'Rajasthan Royals': 'RR', 'Delhi Capitals': 'DC', 'Delhi Daredevils': 'DD', 'Kings XI Punjab': 'KXIP',
  'Punjab Kings': 'PBKS', 'Deccan Chargers': 'DCG', 'Gujarat Titans': 'GT', 'Lucknow Super Giants': 'LSG',
  'Pune Warriors': 'PWI', 'Rising Pune Supergiant': 'RPS', 'Rising Pune Supergiants': 'RPS',
  'Gujarat Lions': 'GL', 'Kochi Tuskers Kerala': 'KTK',
}
export const abbr = (t: string) => (t ? TEAM_ABBR[t] ?? t.split(' ').map(w => w[0]).join('') : '')
