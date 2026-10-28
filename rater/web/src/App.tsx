import { useCallback, useEffect, useRef, useState } from 'react'
import { flag, getNext, getPrev, rate, type Item } from './api'

const FRAME = 1 / 30
const STEP = 0.5
const NAME_KEY = 'rater_name'
const AUTOPLAY_DELAY_MS = 2000

// rater names ignore case and show as "Shay" (matches rater.server.canonical)
function canonical(name: string): string {
  const s = name.trim()
  return s.slice(0, 1).toUpperCase() + s.slice(1).toLowerCase()
}

function loadName(): string {
  try {
    return localStorage.getItem(NAME_KEY) ?? ''
  } catch {
    return ''
  }
}

function saveName(name: string) {
  try {
    localStorage.setItem(NAME_KEY, name)
  } catch {
    // storage unavailable; the name lasts for this page load only
  }
}

function NamePrompt({ onSubmit }: { onSubmit: (name: string) => void }) {
  const [value, setValue] = useState('')
  const go = () => value.trim() && onSubmit(value.trim())
  return (
    <div className="page">
      <div className="center">
        <h1>Your name</h1>
        <input
          autoFocus
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && go()}
        />
        <button onClick={go}>Start</button>
      </div>
    </div>
  )
}

function Rater({ rater }: { rater: string }) {
  const [item, setItem] = useState<Item | null>(null)
  const [done, setDone] = useState(false)
  const [error, setError] = useState('')
  const videoRef = useRef<HTMLVideoElement>(null)
  const busy = useRef(false)

  const loadNext = useCallback(async () => {
    try {
      const n = await getNext(rater)
      if ('done' in n) {
        setItem(null)
        setDone(true)
      } else {
        setItem(n)
        setError('')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'load failed')
    }
  }, [rater])

  useEffect(() => {
    void loadNext()
  }, [loadNext])

  // B steps back to the previously scored item; a score typed there overrides the old one
  const loadPrev = useCallback(async () => {
    if (busy.current) return
    try {
      const p = await getPrev(rater, item?.item_id)
      if ('none' in p) setError('No earlier rating to change')
      else {
        setItem(p)
        setError('')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'load failed')
    }
  }, [rater, item])

  const submit = useCallback(
    async (act: () => Promise<void>) => {
      if (busy.current) return
      busy.current = true
      try {
        await act()
        await loadNext()
      } catch (e) {
        setError(e instanceof Error ? e.message : 'request failed')
      } finally {
        busy.current = false
      }
    },
    [loadNext],
  )

  useEffect(() => {
    if (!item) return
    // Time to read the instruction before the clip starts.
    const t = setTimeout(() => void videoRef.current?.play().catch(() => {}), AUTOPLAY_DELAY_MS)
    return () => clearTimeout(t)
  }, [item])

  useEffect(() => {
    if (!item) return
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return
      const v = videoRef.current
      if (e.key === ' ') {
        e.preventDefault()
        if (v) void (v.paused ? v.play() : v.pause())
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault()
        if (!v) return
        const d = (e.shiftKey ? FRAME : STEP) * (e.key === 'ArrowLeft' ? -1 : 1)
        v.pause()
        v.currentTime = Math.min(Math.max(v.currentTime + d, 0), v.duration || 0)
      } else if (/^[1-5]$/.test(e.key)) {
        void submit(() => rate(rater, item.item_id, Number(e.key), item.score !== undefined))
      } else if (e.key === 'f' || e.key === 'F') {
        void submit(() => flag(rater, item.item_id))
      } else if (e.key === 'b' || e.key === 'B') {
        void loadPrev()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [item, rater, submit, loadPrev])

  if (done) {
    return (
      <div className="page">
        <div className="center">
          <h1>All done</h1>
          <div className="progress">Thank you, {rater}.</div>
        </div>
      </div>
    )
  }
  if (!item) return <div className="page"><div className="error">{error}</div></div>

  return (
    <div className="page">
      <h1 className="instruction">{item.instruction}</h1>
      <video
        key={item.item_id}
        ref={videoRef}
        className="video"
        src={item.video_url}
        muted
        playsInline
        preload="auto"
        onError={() => setError('Video failed to load: press F to flag')}
      />
      <div className="progress">
        {item.position} / {item.total}
        {item.score !== undefined && <> &nbsp; changing rating {item.score}</>}
      </div>
      <div className="error">{error}</div>
      <div className="legend">
        <kbd>1</kbd>-<kbd>5</kbd> score &nbsp; <kbd>F</kbd> broken &nbsp; <kbd>B</kbd> back &nbsp; <kbd>Space</kbd> play/pause &nbsp;
        <kbd>←</kbd> <kbd>→</kbd> 0.5 s &nbsp; <kbd>Shift</kbd>+<kbd>←</kbd> <kbd>→</kbd> 1 frame
      </div>
    </div>
  )
}

export default function App() {
  const [name, setName] = useState(() => canonical(loadName()))
  if (!name) {
    return (
      <NamePrompt
        onSubmit={(raw) => {
          const n = canonical(raw)
          saveName(n)
          setName(n)
        }}
      />
    )
  }
  return <Rater rater={name} />
}
