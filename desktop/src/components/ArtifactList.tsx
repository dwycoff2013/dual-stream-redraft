import React from 'react'

export function ArtifactList({ artifacts, jobId }: { artifacts: any[]; jobId: string }) {
  if (!artifacts?.length) return null
  return <div>
    <h3>Artifacts</h3>
    <table><thead><tr><th>Name</th><th>Path</th><th>Size</th><th>Actions</th></tr></thead><tbody>
      {artifacts.map((a) => <tr key={a.relative_path}><td>{a.name}</td><td>{a.relative_path}</td><td>{a.size}</td><td><a href={`http://127.0.0.1:8765/artifacts/${jobId}/file/${a.relative_path}`} target="_blank">Open</a></td></tr>)}
    </tbody></table>
  </div>
}
