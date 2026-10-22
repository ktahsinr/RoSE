/** The IRIS mark: a stylised iris of an eye. While a run is in progress the
 *  pupil dilates and contracts (CSS animation, disabled by
 *  prefers-reduced-motion); it settles when the answer arrives. */

export default function Logo({
  busy = false,
  size = 36,
}: {
  busy?: boolean;
  size?: number;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      role="img"
      aria-label={busy ? "IRIS logo, run in progress" : "IRIS logo"}
      className="shrink-0"
    >
      <circle
        cx="24"
        cy="24"
        r="21"
        fill="none"
        stroke="var(--accent)"
        strokeWidth="2.5"
      />
      <g stroke="var(--accent)" strokeWidth="1.6" opacity="0.75">
        <line x1="24" y1="5" x2="24" y2="12" />
        <line x1="24" y1="36" x2="24" y2="43" />
        <line x1="5" y1="24" x2="12" y2="24" />
        <line x1="36" y1="24" x2="43" y2="24" />
        <line x1="10.6" y1="10.6" x2="15.5" y2="15.5" />
        <line x1="32.5" y1="32.5" x2="37.4" y2="37.4" />
        <line x1="37.4" y1="10.6" x2="32.5" y2="15.5" />
        <line x1="15.5" y1="32.5" x2="10.6" y2="37.4" />
      </g>
      <circle
        cx="24"
        cy="24"
        r="12"
        fill="none"
        stroke="var(--amber)"
        strokeWidth="3"
      />
      <circle
        cx="24"
        cy="24"
        r="5.2"
        fill="var(--fg)"
        className={busy ? "iris-pupil-busy" : undefined}
      />
      <circle cx="26.5" cy="21.5" r="1.8" fill="var(--bg)" />
    </svg>
  );
}
