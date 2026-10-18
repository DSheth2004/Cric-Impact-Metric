import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

type Mode = 'dark' | 'light'

// Validated categorical palette: fixed slot order, separate steps per mode.
const SERIES = {
  light: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'],
  dark: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
}
const CHROME = {
  light: { surface: '#fcfcfb', ink: '#0b0b0b', ink2: '#52514e', muted: '#76746e', grid: '#e1e0d9', axis: '#c3c2b7', track: '#f0efec' },
  dark: { surface: '#151a17', ink: '#f0efec', ink2: '#c3c2b7', muted: '#8f8d86', grid: '#2c2f2c', axis: '#3c403b', track: '#262b28' },
}
export const STATUS = { good: '#0ca30c', warning: '#fab219', serious: '#ec835a', critical: '#d03b3b' }

function makePalette(mode: Mode) {
  return { mode, series: SERIES[mode], ...CHROME[mode] }
}
export type Palette = ReturnType<typeof makePalette>

const Ctx = createContext<{ mode: Mode; toggle: () => void; pal: Palette }>(null!)

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<Mode>(() => {
    try { return (localStorage.getItem('ci-theme') as Mode) || 'dark' } catch { return 'dark' }
  })
  useEffect(() => {
    document.documentElement.dataset.theme = mode
    try { localStorage.setItem('ci-theme', mode) } catch { /* ignore */ }
  }, [mode])
  const value = useMemo(() => ({
    mode, pal: makePalette(mode), toggle: () => setMode(m => (m === 'dark' ? 'light' : 'dark')),
  }), [mode])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export const useTheme = () => useContext(Ctx)
// eslint-disable-next-line react-refresh/only-export-components
export const usePalette = () => useContext(Ctx).pal

/** Fixed colour order per entity so a filter never repaints survivors. */
export const ROLE_ORDER = ['Batter', 'Bowler', 'All-rounder']
export const CATEGORY_ORDER = ['Match Winner', 'High Impact', 'Neutral', 'Low Impact']
