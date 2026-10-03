export function readDarkMode(): boolean {
  try {
    return window.localStorage.getItem('darkMode') === 'true';
  } catch {
    return false;
  }
}

export function saveDarkMode(darkMode: boolean): void {
  try {
    window.localStorage.setItem('darkMode', String(darkMode));
  } catch {
    // A browser can block storage; the current session still changes theme.
  }
}
