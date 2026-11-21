import { useEffect, useRef } from "react";
import { hrefFor, type Route } from "../lib/router";
import { useUi } from "../lib/store";
import { GITHUB_URL, GitHubMark } from "../ui/GitHubMark";

export function Header({ route }: { route: Route }) {
  const setLibraryOpen = useUi((s) => s.setLibraryOpen);
  const mode = route.mode;
  // the Fold tab returns to whatever was last open in Fold
  const lastFold = useRef("#/fold/crane");
  useEffect(() => {
    if (route.mode === "fold") lastFold.current = route.own ? hrefFor("fold") : hrefFor("fold", route.example);
  }, [route]);
  const MODES = [
    { mode: "fold", label: "Fold", href: route.mode === "fold" ? window.location.hash || lastFold.current : lastFold.current },
    { mode: "ai", label: "AI", href: hrefFor("ai") },
  ];
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
          <path d="M0 0h14L0 14z" fill="var(--paper-back)" />
          <path d="M14 0v14H0z" fill="var(--paper)" />
        </svg>
      </span>
      <nav className="modes" aria-label="Mode">
        {MODES.map((m) => (
          <a key={m.mode} href={m.href} className="mode" aria-current={m.mode === mode ? "page" : undefined}>
            {m.label}
          </a>
        ))}
      </nav>
      <a className="github-link header-github" href={GITHUB_URL} target="_blank" rel="noopener">
        <GitHubMark />
        <span>GitHub</span>
      </a>
    </header>
  );
}
