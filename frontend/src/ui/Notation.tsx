// Origami diagram notation: valley = dashed, mountain = dash-dot, turn over = loop arrow.
const OP_WORDS: Record<string, string> = {
  valley: "Valley fold",
  mountain: "Mountain fold",
  flip: "Turn over",
  rotate: "Turn the paper",
  squash: "Squash fold",
  petal: "Petal fold",
  inside_reverse: "Inside reverse fold",
  outside_reverse: "Outside reverse fold",
  unfold: "Unfold",
};

export function opWords(op: string): string {
  return OP_WORDS[op] ?? op.replace(/_/g, " ");
}

const stroke = { stroke: "currentColor", strokeWidth: 1.6, fill: "none", strokeLinecap: "round" as const };

export function OpBadge({ op }: { op: string }) {
  let body: JSX.Element;
  switch (op) {
    case "valley":
      body = <line x1="3" y1="9" x2="29" y2="9" {...stroke} strokeDasharray="4 3" />;
      break;
    case "mountain":
      body = <line x1="3" y1="9" x2="29" y2="9" {...stroke} strokeDasharray="5 2.5 1 2.5" />;
      break;
    case "flip":
      body = (
        <g {...stroke}>
          <path d="M8 13a8 6 0 1 1 16 0" />
          <path d="M21 10l3 3 3-3" />
        </g>
      );
      break;
    case "rotate":
      body = (
        <g {...stroke}>
          <path d="M22 5a7 7 0 1 0 2 6" />
          <path d="M19 4h4v4" />
        </g>
      );
      break;
    case "squash":
      body = (
        <g {...stroke}>
          <path d="M6 15 16 3l10 12" strokeDasharray="3.5 2.5" />
          <path d="M16 9v6" />
        </g>
      );
      break;
    case "petal":
      body = (
        <g {...stroke}>
          <path d="M6 4l10 11 10-11" strokeDasharray="3.5 2.5" />
          <path d="M6 4h20" />
        </g>
      );
      break;
    case "inside_reverse":
    case "outside_reverse":
      body = (
        <g {...stroke}>
          <line x1="3" y1="12" x2="22" y2="12" strokeDasharray="5 2.5 1 2.5" />
          <path d="M22 12c5 0 6-6 2-8" />
          <path d="M22 2l2 2-2.5 1.5" />
        </g>
      );
      break;
    default:
      body = <line x1="3" y1="9" x2="29" y2="9" {...stroke} />;
  }
  return (
    <span className="op-badge" title={opWords(op)}>
      <svg width="32" height="18" viewBox="0 0 32 18" aria-hidden="true" focusable="false">
        {body}
      </svg>
      <span className="visually-hidden">{opWords(op)}</span>
    </span>
  );
}

/** Diagram-style circled step number. */
export function StepNum({ n, current }: { n: number; current?: boolean }) {
  return (
    <span className={current ? "step-num is-current" : "step-num"} aria-hidden="true">
      {n}
    </span>
  );
}
