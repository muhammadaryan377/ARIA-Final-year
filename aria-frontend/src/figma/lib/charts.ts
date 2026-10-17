// Shared premium palette: primary blue/indigo first, then secondary
// violet / teal / cyan / green, then warm neutrals for overflow.
// Muted professional tones — never neon or rainbow. All charts and KPIs
// consume these tokens (via colorFor/colorForName/accentForIndex); no page
// hardcodes its own hex ramp.
export const CATEGORY_COLORS = [
  "#2F6FAD", // brand blue
  "#6366F1", // indigo
  "#14B8A6", // teal
  "#059669", // green
  "#8B5CF6", // violet
  "#06B6D4", // cyan
  "#C98A1B", // bronze
  "#B85C4D", // terracotta
  "#5B7C99", // slate
  "#D9A441", // gold
];

/** Semantic status colors — success/warning/danger/neutral (premium, muted). */
export const SEMANTIC_COLORS = {
  success: "#059669",
  warning: "#D97706",
  danger: "#DC2626",
  neutral: "#64748B",
  info: "#2F6FAD",
} as const;

/** KPI accent pairs (gradient from→to + soft wash). Values still come 1:1
 *  from the backend; these only dress the card. */
export const KPI_ACCENTS = [
  { from: "#2F6FAD", to: "#1E4E79", soft: "rgba(47,111,173,0.12)" },
  { from: "#6366F1", to: "#4338CA", soft: "rgba(99,102,241,0.12)" },
  { from: "#14B8A6", to: "#0F766E", soft: "rgba(20,184,166,0.12)" },
  { from: "#059669", to: "#047857", soft: "rgba(5,150,105,0.12)" },
  { from: "#8B5CF6", to: "#6D28D9", soft: "rgba(139,92,246,0.12)" },
  { from: "#06B6D4", to: "#0E7490", soft: "rgba(6,182,212,0.12)" },
  { from: "#D97706", to: "#B45309", soft: "rgba(217,119,6,0.12)" },
  { from: "#DC2626", to: "#B91C1C", soft: "rgba(220,38,38,0.10)" },
];

/** Accent for the nth KPI card. Honors a backend-provided hex when valid. */
export function accentForIndex(index: number, customColor?: string | null) {
  const custom =
    typeof customColor === "string" && /^#[0-9a-fA-F]{3,8}$/.test(customColor.trim())
      ? customColor.trim()
      : null;
  if (custom) return { from: custom, to: custom, soft: `${custom}1F` };
  return KPI_ACCENTS[((index % KPI_ACCENTS.length) + KPI_ACCENTS.length) % KPI_ACCENTS.length];
}

export const colorFor = (index: number): string =>
  CATEGORY_COLORS[((index % CATEGORY_COLORS.length) + CATEGORY_COLORS.length) % CATEGORY_COLORS.length];

/** Stable color per category name so legends stay consistent across renders. */
export const colorForName = (name: string): string => {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = (hash * 31 + name.charCodeAt(i)) >>> 0;
  }
  return colorFor(hash);
};
