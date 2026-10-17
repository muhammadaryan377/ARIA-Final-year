import { useId } from "react";

function AriaLogo({
  size = 22,
  mono = false,
  variant = "default",
  className = "",
  title = "ARIA",
}) {
  const gradId = `aria-logo-grad-${useId().replace(/[^a-zA-Z0-9]/g, "")}`;

  const isMarkOnly = variant === "mark-only";
  const isFullLockup = variant === "full-lockup";

  return (
    <svg
      className={`aria-logo ${isMarkOnly ? "aria-logo--mark-only" : ""} ${isFullLockup ? "aria-logo--full-lockup" : ""} ${className}`}
      width={isFullLockup ? "auto" : size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      role="img"
      aria-label={title}
      focusable="false"
      style={isFullLockup ? { width: "auto", height: size } : {}}
    >
      <defs>
        <linearGradient
          id={gradId}
          x1="16"
          y1="52"
          x2="48"
          y2="12"
          gradientUnits="userSpaceOnUse"
        >
          <stop offset="0%" stopColor="var(--aria-accent, #5B4FE9)" />
          <stop offset="100%" stopColor="var(--aria-accent-2, #7C5CE0)" />
        </linearGradient>
        <linearGradient
          id={`${gradId}-subtle`}
          x1="16"
          y1="52"
          x2="48"
          y2="12"
          gradientUnits="userSpaceOnUse"
        >
          <stop offset="0%" stopColor="var(--aria-accent, #5B4FE9)" stopOpacity="0.15" />
          <stop offset="100%" stopColor="var(--aria-accent-2, #7C5CE0)" stopOpacity="0.1" />
        </linearGradient>
      </defs>

      {/* Background glow for mark-only variant */}
      {!isFullLockup && (
        <circle
          cx="32"
          cy="32"
          r="28"
          fill={mono ? "none" : `url(#${gradId}-subtle)`}
          opacity="0.6"
        />
      )}

      {/* Node/Network monogram - 4 nodes forming an "A" shape */}
      <g
        stroke={mono ? "currentColor" : `url(#${gradId})`}
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      >
        {/* Left leg - 3 nodes connected */}
        <path d="M18 44 L24 32 L22 20 L20 8" />
        {/* Right leg - 3 nodes connected */}
        <path d="M46 44 L40 32 L42 20 L44 8" />
        {/* Crossbar connecting the legs */}
        <path d="M22 24 L42 24" />
      </g>

      {/* Nodes - circles at key connection points */}
      <g fill={mono ? "currentColor" : `url(#${gradId})`}>
        {/* Top of left leg */}
        <circle cx="20" cy="8" r="3.5" />
        {/* Top of right leg */}
        <circle cx="44" cy="8" r="3.5" />
        {/* Left crossbar end */}
        <circle cx="22" cy="24" r="3" />
        {/* Right crossbar end */}
        <circle cx="42" cy="24" r="3" />
        {/* Lower left node */}
        <circle cx="20" cy="20" r="2.5" opacity="0.7" />
        {/* Lower right node */}
        <circle cx="44" cy="20" r="2.5" opacity="0.7" />
        {/* Base left */}
        <circle cx="18" cy="44" r="3" opacity="0.5" />
        {/* Base right */}
        <circle cx="46" cy="44" r="3" opacity="0.5" />
      </g>

      {/* Full lockup: wordmark next to mark */}
      {isFullLockup && (
        <g transform="translate(72, 0)">
          <text
            x="0"
            y="38"
            fontFamily="var(--font-sans, Inter, system-ui, sans-serif)"
            fontSize="24"
            fontWeight="600"
            letterSpacing="0.02em"
            fill={mono ? "currentColor" : "var(--aria-text, #0F172A)"}
          >
            ARIA
          </text>
          <text
            x="0"
            y="52"
            fontFamily="var(--font-sans, Inter, system-ui, sans-serif)"
            fontSize="10"
            fontWeight="400"
            letterSpacing="0.08em"
            fill="var(--aria-text-muted, #64748B)"
          >
            INTELLIGENT ANALYTICS
          </text>
        </g>
      )}
    </svg>
  );
}

export default AriaLogo;