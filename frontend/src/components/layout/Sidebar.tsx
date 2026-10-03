import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Activity,
  FileSearch,
  BarChart3,
  Gauge,
  Info,
  Sun,
  Moon,
  X,
  Heart,
} from 'lucide-react';

// Sidebar component
const Sidebar: React.FC<{
  hasCalibration: boolean;
  isOpen: boolean;
  onClose: () => void;
  darkMode: boolean;
  toggleDarkMode: () => void;
}> = ({ hasCalibration, isOpen, onClose, darkMode, toggleDarkMode }) => {
  // Navigation items
  const navItems = [
    { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/signals', icon: Activity, label: 'Signal Explorer' },
    { path: '/explain', icon: FileSearch, label: 'Explainability' },
    { path: '/models', icon: BarChart3, label: 'Model Comparison' },
    ...(hasCalibration ? [{ path: '/calibration', icon: Gauge, label: 'Calibration' }] : []),
    { path: '/about', icon: Info, label: 'About' },
  ];

  const location = useLocation();
  const ThemeIcon = darkMode ? Sun : Moon;

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
        className={`
          fixed top-0 left-0 z-30 h-full w-64
          bg-primary-700 dark:bg-gray-900
          transform transition-transform duration-300 ease-in-out
          lg:translate-x-0 lg:static
          ${isOpen ? 'translate-x-0' : '-translate-x-full'}
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

export default Sidebar;
