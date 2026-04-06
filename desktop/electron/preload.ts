import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('desktopApi', {
  revealPath: (path: string) => ipcRenderer.invoke('reveal-path', path)
})
