import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    // NOTE: scoped to JS/JSX — the default espree parser cannot parse
    // TypeScript syntax. TS correctness is enforced by `tsc --noEmit`
    // (`npm run typecheck`, also run as part of `npm run build`).
    // To lint .ts/.tsx as well, add devDependency `typescript-eslint`
    // and set its parser/languageOptions here.
    files: ['**/*.{js,jsx}'],
    extends: [
      js.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: { ...globals.browser, process: "readonly" },
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    rules: {
      // Backend-error propagation pattern: intentional re-throw with new message.
      "preserve-caught-error": "off",
    },
  },
  {
    // SUPPLY-CHAIN / HYGIENE (non-blocking, visibility-only):
    // Reuses the exact same rule set as the JS/JSX entry above — no strict
    // TS rules enabled here. The default espree parser cannot parse TS
    // syntax, so findings on .ts/.tsx are informational only: run via
    // `npm run lint:ts` (always exits 0). TS correctness stays gated by
    // `tsc --noEmit` (`npm run typecheck`, also part of `npm run build`).
    // To enforce TS lint, add devDependency `typescript-eslint` and set its
    // parser/languageOptions here (blocked until TS7 parser support lands).
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: { ...globals.browser, process: "readonly" },
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    rules: {
      // Backend-error propagation pattern: intentional re-throw with new message.
      "preserve-caught-error": "off",
    },
  },
])
