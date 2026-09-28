import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server runs on 5173 by default, which backend.py's CORS
// allow_origins list already includes. Change the port here (and in
// backend.py) together if you ever need to.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});
