import React, { useMemo, useState } from 'react'

export function FrameExplorer({ frames }: { frames: any[] }) {
  const [index, setIndex] = useState(0)
  const [filter, setFilter] = useState('')
  const filtered = useMemo(() => frames.filter((f) => JSON.stringify(f).toLowerCase().includes(filter.toLowerCase())), [frames, filter])
  const frame = filtered[index]
  if (!frames?.length) return null
  return <div>
    <h3>Frame Explorer</h3>
    <input placeholder="Search frames" value={filter} onChange={(e) => { setFilter(e.target.value); setIndex(0) }} />
    <input type="range" min={0} max={Math.max(filtered.length - 1, 0)} value={Math.min(index, Math.max(filtered.length - 1, 0))} onChange={(e) => setIndex(Number(e.target.value))} />
    <div>{filtered.length ? `Frame ${index + 1}/${filtered.length}` : 'No frames'}</div>
    {frame && <pre>{JSON.stringify(frame, null, 2)}</pre>}
  </div>
}
