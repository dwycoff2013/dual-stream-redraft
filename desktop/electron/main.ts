import { app, BrowserWindow, ipcMain, shell } from 'electron'
import path from 'node:path'
import { spawn, ChildProcess } from 'node:child_process'

let pyProc: ChildProcess | null = null

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 940,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  })
  win.loadURL(process.env.VITE_DEV_SERVER_URL ?? `file://${path.join(__dirname, '../dist/index.html')}`)
}

function startPythonApi() {
  pyProc = spawn('python', ['-m', 'uvicorn', 'dualstream.api:app', '--host', '127.0.0.1', '--port', '8765'])
  pyProc.stdout?.on('data', (d) => console.log(`[py] ${d}`))
  pyProc.stderr?.on('data', (d) => console.error(`[py] ${d}`))
}

app.whenReady().then(() => {
  startPythonApi()
  createWindow()
  app.on('activate', () => BrowserWindow.getAllWindows().length === 0 && createWindow())
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => {
  pyProc?.kill()
})

ipcMain.handle('reveal-path', async (_event, p: string) => {
  await shell.showItemInFolder(p)
})
