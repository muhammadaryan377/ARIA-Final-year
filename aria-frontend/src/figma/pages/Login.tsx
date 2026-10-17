import { useState } from "react";
import { authenticateLocalUser, registerLocalUser } from "../../utils/authStorage";
import AriaLogo from "../components/AriaLogo";

interface LoginProps {
  onLogin: () => void;
}

export default function Login({ onLogin }: LoginProps) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (email.trim() === "" || password.trim() === "") {
      setError("Please enter your email and password.");
      return;
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) {
      setError("Please enter a valid email address.");
      return;
    }

    setLoading(true);
    try {
      if (mode === "register") {
        if (password !== confirmPassword) {
          setError("Passwords do not match");
          setLoading(false);
          return;
        }
        if (password.length < 8) {
          setError("Password must be at least 8 characters");
          setLoading(false);
          return;
        }
        const reg = await registerLocalUser(email, password);
        if (!reg.success) {
          setError(reg.message || "Could not create your account.");
          setLoading(false);
          return;
        }
        // Registration establishes the cookie session: auto-login.
        onLogin();
        return;
      }

      const result = await authenticateLocalUser(email, password);
      if (!result.success) {
        setError(result.message || "Unable to sign in.");
        if (result.code === "USER_NOT_FOUND") {
          setMode("register");
        }
        return;
      }
      onLogin();
    } catch (err) {
      setError("ARIA could not sign you in. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page login-page--signin">
      <div className="login-shell">
        {/* ─── LEFT: ARIA VISUAL AREA ─── */}
        <aside className="login-showcase" aria-hidden="true">
          <div className="login-showcase-brand">
            <AriaLogo size={40} />
            <div>
              <strong>ARIA</strong>
              <span>Intelligent Business Analytics</span>
            </div>
          </div>

          <div className="login-showcase-center">
            <div className="login-showcase-copy">
              <h3>
                Turn your data
                <br />
                into smarter decisions.
              </h3>
              <p>
                Ask business questions in plain language, explore your data,
                and uncover meaningful insights with ARIA.
              </p>
            </div>

            <div className="login-stage" aria-hidden="true">
              <div className="login-focal">
                <span className="login-focal-ring" />
                <span className="login-focal-symbol">✦</span>
                <span className="login-focal-signal login-focal-signal--in" />
                <span className="login-focal-signal login-focal-signal--out" />
              </div>

              <div className="login-kpi">
                <span className="login-float-label">Intelligence</span>
                <strong className="login-kpi-value">ARIA Agents</strong>
                <span className="login-kpi-delta">Ask · Explore · Decide</span>
              </div>

              <div className="login-chart">
                <span className="login-float-label">Schema Discovery</span>
                <svg
                  className="login-chart-svg"
                  viewBox="0 0 140 44"
                  preserveAspectRatio="none"
                  aria-hidden="true"
                >
                  <polyline
                    className="login-chart-line"
                    points="0,36 26,30 48,32 74,15 100,19 140,6"
                  />
                  <circle className="login-chart-dot" cx="0" cy="36" r="3" />
                </svg>
              </div>

              <div className="login-insight">
                <span className="login-float-tag">ARIA</span>
                <span className="login-insight-text">
                  Natural-language analytics
                </span>
              </div>
            </div>
          </div>

          <div className="login-showcase-bottom">
            <ul className="login-showcase-features">
              <li>Connect your databases &amp; files</li>
              <li>Explore your schema automatically</li>
              <li>Ask questions without SQL</li>
            </ul>
            <div className="login-showcase-footer">
              Secure intelligent analytics · ARIA
            </div>
          </div>
        </aside>

        {/* ─── RIGHT: LOGIN / REGISTER AREA ─── */}
        <div className="login-card">
          <div className="login-card-brand">
            <span className="login-card-mark">
              <AriaLogo mono size={20} />
            </span>
            <span className="login-card-wordmark">ARIA</span>
          </div>

          <div className="login-heading">
            <h2>{mode === "login" ? "Welcome back" : "Create account"}</h2>
            <p>
              {mode === "login"
                ? "Sign in to your ARIA workspace"
                : "Get started with ARIA Intelligence"}
            </p>
          </div>

          <form className="login-form" onSubmit={handleSubmit} noValidate>
            {error && (
              <p className="login-error" role="alert">{error}</p>
            )}

            <div className="login-form-group">
              <label htmlFor="login-email">Email Address</label>
              <input
                id="login-email"
                type="email"
                value={email}
                onChange={(e) => { setEmail(e.target.value); setError(""); }}
                placeholder="you@company.com"
                autoComplete="email"
                aria-invalid={error ? "true" : undefined}
              />
            </div>

            <div className="login-form-group">
              <div className="login-label-row">
                <label htmlFor="login-password">Password</label>
                {mode === "login" && (
                  <button type="button" className="login-forgot" onClick={() => setError("Password reset is not available in this build. Contact your workspace admin.")}>Forgot password?</button>
                )}
              </div>
              <input
                id="login-password"
                type="password"
                value={password}
                onChange={(e) => { setPassword(e.target.value); setError(""); }}
                placeholder="Enter your password"
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                aria-invalid={error ? "true" : undefined}
              />
            </div>

            {mode === "register" && (
              <div className="login-form-group">
                <label htmlFor="login-confirm">Confirm Password</label>
                <input
                  id="login-confirm"
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Confirm your password"
                  autoComplete="new-password"
                />
              </div>
            )}

            <button
              type="submit"
              className={`login-button ${loading ? "is-loading" : ""}`}
              disabled={loading}
            >
              {loading ? (
                <span className="login-button-loading">
                  <span className="login-button-spinner" aria-hidden="true" />
                  {mode === "login" ? "Signing in…" : "Creating account…"}
                </span>
              ) : (
                <span>
                  {mode === "login" ? "Access Intelligence Hub" : "Create Account"}{" "}
                  <span aria-hidden="true">→</span>
                </span>
              )}
            </button>
          </form>

          <div className="login-footer">
            <span>{mode === "login" ? "New here?" : "Already have an account?"}</span>
            <button
              type="button"
              className="register-login-link"
              onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }}
            >
              {mode === "login" ? "Create Account" : "Sign In"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
