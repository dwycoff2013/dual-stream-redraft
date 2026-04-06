import React, { useState } from 'react'

export function JsonViewer({ title, value }: { title: string; value: unknown }) {
  const [open, setOpen] = useState(false)
  if (!value) return null
  const text = JSON.stringify(value, null, 2)
  return <div>
    <button onClick={() => setOpen(!open)}>{open ? '▼' : '▶'} {title}</button>
    {open && <div><button onClick={() => navigator.clipboard.writeText(text)}>Copy</button><pre>{text}</pre></div>}
  </div>
}
