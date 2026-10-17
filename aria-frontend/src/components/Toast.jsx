import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";

const ToastContext = createContext(null);

const TOAST_ICONS = {
  success: "✓",
  error: "✕",
  info: "✦",
};

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  const nextIdRef = useRef(1);

  const dismiss = useCallback((id) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const show = useCallback(
    (type, message) => {
      const id = `toast-${nextIdRef.current++}`;

      setToasts((current) => [...current, { id, type, message }]);

      window.setTimeout(() => dismiss(id), 3600);
    },
    [dismiss],
  );

  const api = useMemo(
    () => ({
      show,
      success: (message) => show("success", message),
      info: (message) => show("info", message),
      error: (message) => show("error", message),
    }),
    [show],
  );

  return (
    <ToastContext.Provider value={api}>
      {children}

      <div className="toast-region" aria-live="polite" aria-atomic="false">
        {toasts.map((toast) => (
          <div className={`toast toast-${toast.type}`} key={toast.id}>
            <span className="toast-icon" aria-hidden="true">
              {TOAST_ICONS[toast.type] || "✦"}
            </span>

            <span className="toast-message">{toast.message}</span>

            <button
              type="button"
              className="toast-close"
              aria-label="Dismiss notification"
              onClick={() => dismiss(toast.id)}
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useToast() {
  const context = useContext(ToastContext);

  if (!context) {
    throw new Error("useToast must be used inside a <ToastProvider>.");
  }

  return context;
}