/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import "dotenv/config";

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Middleware to parse JSON request bodies
  app.use(express.json());

  // 1. API: Check configuration status
  app.get("/api/config", async (req, res) => {
    if (process.env.OPENAI_API_KEY) {
      return res.json({
        hasSystemKey: true,
      });
    }
    // Forward to backend config
    try {
      const response = await fetch("http://127.0.0.1:8000/api/config");
      const data = await response.json();
      res.json(data);
    } catch (e) {
      res.json({ hasSystemKey: false });
    }
  });

  // 2. API: Verify custom or system API key
  app.post("/api/verify-key", async (req, res) => {
    try {
      const { apiKey } = req.body;
      const keyToUse = apiKey || process.env.OPENAI_API_KEY;

      if (!keyToUse) {
        // Forward to backend
        try {
          const response = await fetch("http://127.0.0.1:8000/api/verify-key", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(req.body),
          });
          const data = await response.json();
          return res.json(data);
        } catch (e: any) {
          return res.status(400).json({ valid: false, message: `Backend connection error: ${e.message}` });
        }
      }

      // Initialize client and run a minimal test query against OpenAI
      const response = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${keyToUse}`
        },
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "test" }],
          max_tokens: 5
        })
      });

      if (response.ok) {
        return res.json({ valid: true, message: "API Key is valid!" });
      } else {
        const errData = await response.json().catch(() => ({}));
        return res.status(400).json({ 
          valid: false, 
          message: errData?.error?.message || "No response from OpenAI API." 
        });
      }
    } catch (error: any) {
      console.error("API Key validation error:", error);
      return res.status(400).json({ 
        valid: false, 
        message: error.message || "An error occurred while validating the API Key." 
      });
    }
  });

  // 3. API: Stream Chat Response (SSE)
  app.post("/api/chat/stream", async (req, res) => {
    try {
      // Forward all chat stream requests to the Python backend to run Graph RAG
      try {
        const response = await fetch("http://127.0.0.1:8000/api/chat/stream", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(req.body),
        });

        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");

        if (!response.body) {
          res.write(`data: ${JSON.stringify({ error: "No response body from backend." })}\n\n`);
          return res.end();
        }

        // Forward stream bytes
        const reader = response.body.getReader();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          res.write(value);
        }
        return res.end();
      } catch (error: any) {
        console.error("Backend proxy error:", error);
        res.setHeader("Content-Type", "text/event-stream");
        res.write(`data: ${JSON.stringify({ error: `Backend connection error: ${error.message}` })}\n\n`);
        return res.end();
      }
    } catch (error: any) {
      console.error("Streaming error:", error);
      res.write(`data: ${JSON.stringify({ error: error.message || "An unexpected error occurred." })}\n\n`);
      res.end();
    }
  });

  // 4. API: Get Documents List
  app.get("/api/documents", async (req, res) => {
    try {
      const response = await fetch("http://127.0.0.1:8000/api/documents");
      const data = await response.json();
      res.json(data);
    } catch (e: any) {
      res.status(500).json({ error: e.message });
    }
  });

  // 5. API: Delete Document
  app.delete("/api/documents/:id", async (req, res) => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/documents/${req.params.id}`, {
        method: "DELETE"
      });
      const data = await response.json();
      res.json(data);
    } catch (e: any) {
      res.status(500).json({ error: e.message });
    }
  });

  // 6. API: Upload Document (Proxy multipart/form-data)
  app.post("/api/documents/upload", async (req, res) => {
    try {
      const headers: Record<string, string> = {};
      if (req.headers["content-type"]) {
        headers["content-type"] = req.headers["content-type"];
      }

      const response = await fetch("http://127.0.0.1:8000/api/documents/upload", {
        method: "POST",
        headers,
        body: req as any,
        // @ts-ignore
        duplex: 'half'
      });

      const data = await response.json();
      res.status(response.status).json(data);
    } catch (e: any) {
      console.error("Upload proxy error:", e);
      res.status(500).json({ error: e.message });
    }
  });

  // 7. API: Community Detection Endpoints Proxy
  app.use("/api/community", async (req, res) => {
    try {
      const queryParams = new URLSearchParams(req.query as any).toString();
      const targetPath = req.path || "";
      const targetUrl = queryParams
        ? `http://127.0.0.1:8000/api/community${targetPath}?${queryParams}`
        : `http://127.0.0.1:8000/api/community${targetPath}`;

      const options: RequestInit = {
        method: req.method,
        headers: { "Content-Type": "application/json" },
      };
      if (req.method !== "GET" && req.method !== "HEAD") {
        options.body = JSON.stringify(req.body);
      }

      const response = await fetch(targetUrl, options);
      const data = await response.json();
      res.status(response.status).json(data);
    } catch (e: any) {
      console.error("Community proxy error:", e);
      res.status(500).json({ error: e.message });
    }
  });

  // Vite middleware for dev or static asset serving for prod
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
