// Silhouettes of the finished examples (SVG files written by the example build) and the blank squares.
import { exampleFileUrl, getExample } from "../lib/examples";

type Face = { d: string; tone: "front" | "back" | "white" };

const P = (pts: string, tone: Face["tone"] = "front"): Face => ({ d: `M${pts}Z`, tone });

const SHAPES: Record<string, Face[]> = {
  blank: [P("18,18 82,18 82,82 18,82", "white")],
  blank_blue: [P("18,18 82,18 82,82 18,82")],
};

const FILL = { front: "var(--kami)", back: "var(--kami-deep)", white: "var(--sheet)" } as const;

export function Thumb({ name, size = 56, className }: { name: string; size?: number; className?: string }) {
  const example = getExample(name);
  if (example) {
    return <img className={className} src={exampleFileUrl(example, "thumb")} width={size} height={size} alt="" aria-hidden="true" />;
  }
  const faces = SHAPES[name] ?? SHAPES.blank;
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 100 100" aria-hidden="true" focusable="false">
      {faces.map((f, i) => (
        <path key={i} d={f.d} fill={FILL[f.tone]} stroke="var(--ink)" strokeOpacity={0.55} strokeWidth={1.1} strokeLinejoin="round" />
      ))}
    </svg>
  );
}
