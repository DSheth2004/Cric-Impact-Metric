import { useEffect, useState } from 'react'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Row = Record<string, any>

// Backend address. Empty in local dev (Vite proxies /api to the local backend);
// in production set VITE_API_URL to the Render URL, e.g. https://cric-impact-api.onrender.com
const API_BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/+$/, '')

async function parse<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = r.statusText
    try { msg = (await r.json()).detail || msg } catch { /* ignore */ }
    throw new Error(msg)
  }
  return (await r.json()).data as T
}

export async function apiGet<T = Row>(path: string): Promise<T> {
  return parse<T>(await fetch(`${API_BASE}/api${path}`))
}

export async function apiPost<T = Row>(path: string, body: BodyInit, json = false): Promise<T> {
  return parse<T>(await fetch(`${API_BASE}/api${path}`, {
    method: 'POST',
    body,
    headers: json ? { 'Content-Type': 'application/json' } : undefined,
  }))
}

const cache = new Map<string, unknown>()
const inflight = new Map<string, Promise<unknown>>()

/** Fetch once and share: concurrent and repeat requests for the same path reuse one response. */
function cachedGet<T>(path: string): Promise<T> {
  if (cache.has(path)) return Promise.resolve(cache.get(path) as T)
  let p = inflight.get(path) as Promise<T> | undefined
  if (!p) {
    p = apiGet<T>(path).then(d => { cache.set(path, d); return d }).finally(() => inflight.delete(path))
    inflight.set(path, p)
  }
  return p
}

/** Warm the cache in the background; errors are ignored. */
export function prefetch(path: string) { cachedGet(path).catch(() => undefined) }

export function useApi<T = Row>(path: string | null) {
  const [data, setData] = useState<T | null>(path ? (cache.get(path) as T) ?? null : null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  useEffect(() => {
    if (!path) { setData(null); return }
    let alive = true
    if (cache.has(path)) { setData(cache.get(path) as T); setError(null); return }
    setLoading(true); setError(null)
    cachedGet<T>(path)
      .then(d => { if (alive) setData(d) })
      .catch(e => { if (alive) setError(String(e.message || e)) })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [path])
  return { data, error, loading }
}

export function useSeasons(): string[] {
  return useApi<string[]>('/seasons').data ?? []
}

export function usePlayerList(): Row[] {
  return useApi<Row[]>('/player-list').data ?? []
}

export const q =(params: Record<string, string | number>) =>
  Object.entries(params).map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`).join('&')
