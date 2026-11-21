import { hrefFor, type Mode } from "../lib/router";
import { useUi } from "../lib/store";

const MODES: { mode: Mode; label: string }[] = [
  { mode: "watch", label: "Watch" },
  { mode: "fold", label: "Fold" },
  { mode: "ai", label: "AI" },
];

export function Header({ mode }: { mode: Mode }) {
  const setLibraryOpen = useUi((s) => s.setLibraryOpen);
  return (
    <header className="header">
      <button className="icon-button menu-button" onClick={() => setLibraryOpen(true)} aria-label="Open the menu" title="Menu">
        <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
          <path d="M3.5 6h15M3.5 11h15M3.5 16h15" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </svg>
      </button>
      <span className="wordmark">
        <span className="wordmark-text">Foldr</span>
        <svg className="wordmark-corner" viewBox="0 0 14 14" aria-hidden="true">
          <path d="M0 0h14L0 14z" fill="var(--kami)" />
          <path d="M14 0v14H0z" fill="var(--accent)" />
        </svg>
      </span>
      <nav className="modes" aria-label="Mode">
        {MODES.map((m) => (
          <a key={m.mode} href={hrefFor(m.mode)} className="mode" aria-current={m.mode === mode ? "page" : undefined}>
            {m.label}
          </a>
        ))}
      </nav>
    </header>
  );
}
