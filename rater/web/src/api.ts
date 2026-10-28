export type Item = {
  item_id: string
  clip_id: string
  instruction: string
  video_url: string
  position: number
  total: number
  score?: number
}

export type Next = Item | { done: true }

export async function getNext(rater: string): Promise<Next> {
  const r = await fetch(`/api/next?rater=${encodeURIComponent(rater)}`)
  if (!r.ok) throw new Error(`next failed (${r.status})`)
  return r.json()
}

export async function getPrev(rater: string, item_id?: string): Promise<Item | { none: true }> {
  const q = item_id ? `&item_id=${encodeURIComponent(item_id)}` : ''
  const r = await fetch(`/api/prev?rater=${encodeURIComponent(rater)}${q}`)
  if (!r.ok) throw new Error(`prev failed (${r.status})`)
  return r.json()
}

async function post(path: string, body: object): Promise<void> {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  // 409 means already rated elsewhere (e.g. a second tab); the caller loads the next item.
  if (!r.ok && r.status !== 409) throw new Error(`request failed (${r.status})`)
}

export const rate = (rater: string, item_id: string, score: number, override = false) =>
  post('/api/rate', { rater, item_id, score, override })

export const flag = (rater: string, item_id: string) => post('/api/flag', { rater, item_id })
