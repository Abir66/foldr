import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./ui/tokens.css";
import "./ui/app.css";
import { App } from "./app/App";
import { loadExamples } from "./lib/examples";

const root = createRoot(document.getElementById("root")!);

loadExamples().then(
  () =>
    root.render(
      <StrictMode>
        <App />
      </StrictMode>,
    ),
  (err) => {
    console.error(err);
    root.render(<p style={{ padding: 24 }}>The examples could not be loaded. Try reloading the page.</p>);
  },
);
