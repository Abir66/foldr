export function RunMark({ passed }: { passed: boolean }) {
  return passed ? (
    <span className="run-mark is-pass">
      <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6.5 5 9.5 10 3" /></svg>
      Folded it
    </span>
  ) : (
    <span className="run-mark is-fail">
      <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 3l6 6M9 3 3 9" /></svg>
      Didn't match
    </span>
  );
}
