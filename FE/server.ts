/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import "dotenv/config";

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Middleware to parse JSON request bodies
  app.use(express.json());

  // 1. API: Check configuration status
  app.get("/api/config", async (req, res) => {
    if (process.env.GEMINI_API_KEY) {
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
      const keyToUse = apiKey || process.env.GEMINI_API_KEY;

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

      // Initialize client and run a minimal test query
      const ai = new GoogleGenAI({
        apiKey: keyToUse,
        httpOptions: {
          headers: {
            'User-Agent': 'aistudio-build',
          }
        }
      });

      const response = await ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents: "test",
        config: {
          maxOutputTokens: 5,
        }
      });

      if (response && response.text) {
        return res.json({ valid: true, message: "API Key is valid!" });
      } else {
        return res.status(400).json({ valid: false, message: "No response from Gemini API." });
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
      const { messages, model, config, apiKey } = req.body;
      const keyToUse = apiKey || process.env.GEMINI_API_KEY;

      if (!keyToUse) {
        // Forward to Python backend
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
      }

      if (!messages || !Array.isArray(messages) || messages.length === 0) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.write(`data: ${JSON.stringify({ error: "No messages provided." })}\n\n`);
        return res.end();
      }

      // Format messages correctly for the SDK
      const contents = messages.map((msg: any) => ({
        role: msg.role === 'user' ? 'user' : 'model',
        parts: [{ text: msg.content }],
      }));

      const ai = new GoogleGenAI({
        apiKey: keyToUse,
        httpOptions: {
          headers: {
            'User-Agent': 'aistudio-build',
          }
        }
      });

      // Stream completion
      const responseStream = await ai.models.generateContentStream({
        model: model || "gemini-3.5-flash",
        contents,
        config: {
          systemInstruction: config?.systemInstruction || undefined,
          temperature: typeof config?.temperature === 'number' ? config.temperature : 0.7,
          topP: typeof config?.topP === 'number' ? config.topP : undefined,
          topK: typeof config?.topK === 'number' ? config.topK : undefined,
        },
      });

      for await (const chunk of responseStream) {
        if (chunk.text) {
          res.write(`data: ${JSON.stringify({ text: chunk.text })}\n\n`);
        }
      }

      // Signal completion
      res.write("data: [DONE]\n\n");
      res.end();
    } catch (error: any) {
      console.error("Streaming error:", error);
      res.write(`data: ${JSON.stringify({ error: error.message || "An unexpected error occurred." })}\n\n`);
      res.end();
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
