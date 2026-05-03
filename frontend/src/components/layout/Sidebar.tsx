import { NavLink } from 'react-router-dom';

const navItems = [
  { to: '/', label: 'Dashboard', icon: '~' },
  { to: '/races', label: 'Races', icon: '~' },
  { to: '/dogs', label: 'Dogs', icon: '~' },
  { to: '/features', label: 'Features', icon: '~' },
  { to: '/training', label: 'Training Lab', icon: '~' },
  { to: '/predictions', label: 'Predictions', icon: '~' },
  { to: '/bankroll', label: 'Bankroll', icon: '~' },
  { to: '/scraping', label: 'Scraping', icon: '~' },
];

interface SidebarProps {
  open: boolean;
  onClose: () => void;
}

export default function Sidebar({ open, onClose }: SidebarProps) {
  return (
    <aside
      className={`
        fixed inset-y-0 left-0 z-50 w-56 bg-gray-900 text-gray-300 flex flex-col shrink-0 min-h-screen
        transition-transform duration-200 ease-in-out
        md:static md:translate-x-0
        ${open ? 'translate-x-0' : '-translate-x-full'}
      `}
    >
      <div className="px-5 py-5 border-b border-gray-700 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-white tracking-tight">
            Greyhound Predictor
          </h1>
          <p className="text-xs text-gray-500 mt-0.5">Irish Race Analytics</p>
        </div>
        <button
          onClick={onClose}
          className="md:hidden p-1 text-gray-400 hover:text-white"
          aria-label="Close menu"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      <nav className="flex-1 py-3">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            onClick={onClose}
            className={({ isActive }) =>
              `flex items-center px-5 py-2.5 text-sm transition-colors ${
                isActive
                  ? 'bg-gray-800 text-white border-r-2 border-blue-500'
                  : 'hover:bg-gray-800 hover:text-white'
              }`
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
      <div className="px-5 py-3 border-t border-gray-700 text-xs text-gray-500">
        v0.1.03
      </div>
    </aside>
  );
}
