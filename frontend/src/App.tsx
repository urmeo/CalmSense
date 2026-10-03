import React, { lazy, Suspense, useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

// Components
import Dashboard from './pages/Dashboard';
import ModelComparison from './pages/ModelComparison';
import About from './pages/About';
import ErrorBoundary from './components/ErrorBoundary';
import Sidebar from './components/layout/Sidebar';
import Header from './components/layout/Header';
import results from './data/results';

const SignalExplorer = lazy(() => import('./pages/SignalExplorer'));
const ExplainabilityDashboard = lazy(() => import('./pages/ExplainabilityDashboard'));
const CalibrationPanel = lazy(() => import('./pages/CalibrationPanel'));

// The calibration section is optional; only show it once the experiment has produced it.
const hasCalibration = Boolean((results as any).calibration);

// Main App component
const App: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });

  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const toggleDarkMode = () => setDarkMode(!darkMode);

  return (
    <Router basename={import.meta.env.BASE_URL}>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <div className="flex h-screen overflow-hidden">
          {/* Sidebar */}
          <Sidebar
            hasCalibration={hasCalibration}
            isOpen={sidebarOpen}
            onClose={() => setSidebarOpen(false)}
            darkMode={darkMode}
            toggleDarkMode={toggleDarkMode}
          />

          {/* Main content */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <Header onMenuClick={() => setSidebarOpen(true)} />

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
                CalmSense v1.0.0 | Multimodal Stress Detection System
              </div>
            </footer>
          </div>
        </div>
      </div>
    </Router>
  );
};

export default App;
