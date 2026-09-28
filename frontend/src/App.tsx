import React, { lazy, Suspense, useState, useEffect, useRef, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Activity,
  FileSearch,
  BarChart3,
  Gauge,
  Info,
  Sun,
  Moon,
  Menu,
  X,
  Heart
} from 'lucide-react';

// Components
import Dashboard from './pages/Dashboard';
import ModelComparison from './pages/ModelComparison';
import About from './pages/About';
import ErrorBoundary from './components/ErrorBoundary';
import results from './data/results.json';

const SignalExplorer = lazy(() => import('./pages/SignalExplorer'));
const ExplainabilityDashboard = lazy(() => import('./pages/ExplainabilityDashboard'));
const CalibrationPanel = lazy(() => import('./pages/CalibrationPanel'));

// The calibration section is optional; only show it once the experiment has produced it.
const hasCalibration = Boolean((results as any).calibration);

// Navigation items
const navItems = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/signals', icon: Activity, label: 'Signal Explorer' },
  { path: '/explain', icon: FileSearch, label: 'Explainability' },
  { path: '/models', icon: BarChart3, label: 'Model Comparison' },
  ...(hasCalibration ? [{ path: '/calibration', icon: Gauge, label: 'Calibration' }] : []),
  { path: '/about', icon: Info, label: 'About' },
];

// Sidebar component
const Sidebar: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  darkMode: boolean;
  toggleDarkMode: () => void;
}> = ({ isOpen, onClose, darkMode, toggleDarkMode }) => {
  const location = useLocation();
  const ThemeIcon = darkMode ? Sun : Moon;
  const sidebarRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!isOpen) return;
    const sidebar = sidebarRef.current;
    const controls = sidebar?.querySelectorAll<HTMLElement>('a[href], button');
    controls?.[0]?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (!window.matchMedia('(max-width: 1023px)').matches) return;
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
      } else if (event.key === 'Tab' && controls?.length) {
        const first = controls[0];
        const last = controls[controls.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-20 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <aside
        id="navigation"
        ref={sidebarRef}
        className={`
          fixed top-0 left-0 z-30 h-full w-64
          bg-primary-700 dark:bg-gray-900
          transform transition-transform duration-300 ease-in-out
          lg:visible lg:translate-x-0 lg:static
          ${isOpen ? 'visible translate-x-0' : 'invisible -translate-x-full'}
        `}
      >
        {/* Logo */}
        <div className="flex items-center justify-between p-4 border-b border-primary-600 dark:border-gray-700">
          <div className="flex items-center space-x-2">
            <Heart className="w-8 h-8 text-red-400" />
            <span className="text-xl font-bold text-white">CalmSense</span>
          </div>
          <button
            onClick={onClose}
            aria-label="Close navigation menu"
            className="lg:hidden text-white hover:text-gray-300"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="p-4 space-y-2">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            const Icon = item.icon;

            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={onClose}
                aria-current={isActive ? 'page' : undefined}
                className={`
                  flex items-center space-x-3 px-4 py-3 rounded-lg
                  transition-colors duration-200
                  ${isActive
                    ? 'bg-accent-500 text-white'
                    : 'text-gray-300 hover:bg-primary-600 dark:hover:bg-gray-800'
                  }
                `}
              >
                <Icon className="w-5 h-5" />
                <span className="font-medium">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Dark mode toggle */}
        <div className="absolute bottom-4 left-4 right-4">
          <button
            onClick={toggleDarkMode}
            className="flex items-center justify-center w-full px-4 py-3
                       bg-primary-600 dark:bg-gray-800 rounded-lg
                       text-white hover:bg-primary-500 dark:hover:bg-gray-700
                       transition-colors duration-200"
          >
            <ThemeIcon className="w-5 h-5 mr-2" />
            <span>{darkMode ? 'Light Mode' : 'Dark Mode'}</span>
          </button>
        </div>
      </aside>
    </>
  );
};

// Header component
const Header: React.FC<{
  onMenuClick: () => void;
  menuOpen: boolean;
  menuButtonRef: React.Ref<HTMLButtonElement>;
}> = ({ onMenuClick, menuOpen, menuButtonRef }) => {
  return (
    <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between px-4 py-3">
        <button
          ref={menuButtonRef}
          onClick={onMenuClick}
          aria-label="Open navigation menu"
          aria-expanded={menuOpen}
          aria-controls="navigation"
          className="lg:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
        >
          <Menu className="w-6 h-6 text-gray-600 dark:text-gray-300" />
        </button>

        <div className="flex items-center space-x-4">
          <div className="hidden sm:flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400">
            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
            <span>Precomputed research results</span>
          </div>
        </div>
      </div>
    </header>
  );
};

// Main App component
const App: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const menuButtonRef = useRef<HTMLButtonElement>(null);
  const [darkMode, setDarkMode] = useState(() => {
    try {
      return localStorage.getItem('darkMode') === 'true';
    } catch {
      return false;
    }
  });

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
    try {
      localStorage.setItem('darkMode', String(darkMode));
    } catch {}
  }, [darkMode]);

  const toggleDarkMode = () => setDarkMode(!darkMode);
  const closeSidebar = useCallback(() => {
    setSidebarOpen(false);
    menuButtonRef.current?.focus();
  }, []);

  return (
    <Router basename={import.meta.env.BASE_URL}>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <div className="flex h-screen overflow-hidden">
          {/* Sidebar */}
          <Sidebar
            isOpen={sidebarOpen}
            onClose={closeSidebar}
            darkMode={darkMode}
            toggleDarkMode={toggleDarkMode}
          />

          {/* Main content */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <Header
              onMenuClick={() => setSidebarOpen(true)}
              menuOpen={sidebarOpen}
              menuButtonRef={menuButtonRef}
            />

            <main className="flex-1 overflow-y-auto p-4 lg:p-6">
              <Suspense fallback={<p role="status" className="text-gray-600 dark:text-gray-300">Loading charts…</p>}>
                <Routes>
                  <Route path="/" element={<ErrorBoundary><Dashboard /></ErrorBoundary>} />
                  <Route path="/signals" element={<ErrorBoundary><SignalExplorer /></ErrorBoundary>} />
                  <Route path="/explain" element={<ErrorBoundary><ExplainabilityDashboard /></ErrorBoundary>} />
                  <Route path="/models" element={<ErrorBoundary><ModelComparison /></ErrorBoundary>} />
                  {hasCalibration && (
                    <Route path="/calibration" element={<ErrorBoundary><CalibrationPanel /></ErrorBoundary>} />
                  )}
                  <Route path="/about" element={<ErrorBoundary><About /></ErrorBoundary>} />
                </Routes>
              </Suspense>
            </main>

            {/* Footer */}
            <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-4 py-3">
              <div className="text-center text-sm text-gray-500 dark:text-gray-400">
                CalmSense v0.1.0 | Multimodal Stress Detection System
              </div>
            </footer>
          </div>
        </div>
      </div>
    </Router>
  );
};

export default App;
