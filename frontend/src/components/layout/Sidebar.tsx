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

export default function Sidebar() {
  return (
    <aside className="w-56 bg-gray-900 text-gray-300 flex flex-col shrink-0 min-h-screen">
      <div className="px-5 py-5 border-b border-gray-700">
        <h1 className="text-lg font-bold text-white tracking-tight">
          Greyhound Predictor
        </h1>
        <p className="text-xs text-gray-500 mt-0.5">Irish Race Analytics</p>
      </div>
      <nav className="flex-1 py-3">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
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
        v0.1.0
      </div>
    </aside>
  );
}
