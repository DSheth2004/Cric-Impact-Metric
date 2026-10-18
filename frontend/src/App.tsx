import { NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { GitCompare, Home as HomeIcon, Moon, Sun, Trophy, Upload as UploadIcon, User, Swords } from 'lucide-react'
import { useEffect } from 'react'
import { useTheme } from './lib/theme'
import Home from './pages/Home'
import Leaderboard from './pages/Leaderboard'
import Player from './pages/Player'
import Compare from './pages/Compare'
import Matches from './pages/Matches'
import Upload from './pages/Upload'

const LINKS = [
  { to: '/', label: 'Home', icon: HomeIcon },
  { to: '/leaderboard', label: 'Leaderboard', icon: Trophy },
  { to: '/player', label: 'Player Profile', icon: User },
  { to: '/compare', label: 'Compare', icon: GitCompare },
  { to: '/matches', label: 'Match Explorer', icon: Swords },
  { to: '/upload', label: 'Upload Match', icon: UploadIcon },
]

export default function App() {
  const { mode, toggle } = useTheme()
  const loc = useLocation()
  useEffect(() => { window.scrollTo(0, 0) }, [loc.pathname])
  return (
    <>
      <header className="nav">
        <NavLink to="/" className="brand">
          <svg className="brand-ball" viewBox="0 0 32 32" aria-hidden>
            <circle cx="16" cy="16" r="14" fill="#c8102e" />
            <path d="M7 9c5 4 5 10 0 14M25 9c-5 4-5 10 0 14" stroke="#fff" strokeWidth="1.6" fill="none" strokeDasharray="2 2" />
          </svg>
          <span>CricImpact</span>
        </NavLink>
        <nav className="nav-links">
          {LINKS.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} end={to === '/'} title={label} aria-label={label}>
              <Icon size={15} /> <span className="nav-label">{label}</span>
            </NavLink>
          ))}
        </nav>
        <button className="icon-btn" onClick={toggle} aria-label="Toggle theme" title="Toggle theme">
          {mode === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
        </button>
      </header>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/leaderboard" element={<Leaderboard />} />
        <Route path="/player" element={<Player />} />
        <Route path="/compare" element={<Compare />} />
        <Route path="/matches" element={<Matches />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="*" element={<div className="page"><div className="empty">Page not found</div></div>} />
      </Routes>
    </>
  )
}
