import React from "react";

// Catches render crashes per page so one bad chart never blanks the whole app.
// t.txt section 55: real error states, never silently fail / blank screen.
interface Props {
  children: React.ReactNode;
  page?: string;
}

interface State {
  hasError: boolean;
  message: string;
}

export default class ErrorBoundary extends React.Component<Props, State> {
  state: State = { hasError: false, message: "" };

  static getDerivedStateFromError(err: unknown): State {
    return {
      hasError: true,
      message: err instanceof Error ? err.message : "Something went wrong.",
    };
  }

  componentDidCatch(err: unknown) {
    // eslint-disable-next-line no-console
    console.error(`[ARIA:${this.props.page ?? "app"}] render error:`, err);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 24, maxWidth: 560 }}>
          <h2>Something went wrong on this page.</h2>
          <p style={{ opacity: 0.75 }}>{this.state.message}</p>
          <p style={{ opacity: 0.75 }}>
            Your connection and history are intact — try reloading this view or
            picking another page.
          </p>
          <button onClick={() => this.setState({ hasError: false, message: "" })}>
            Retry this view
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
