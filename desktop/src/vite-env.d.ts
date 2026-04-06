declare global {
  interface Window {
    desktopApi?: { revealPath: (path: string) => Promise<void> }
  }
}
export {}
