/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Middleware to parse JSON request bodies
  app.use(express.json());

  // 1. API: Check configuration status
  app.get("/api/config", (req, res) => {
    res.json({
      hasSystemKey: !!process.env.GEMINI_API_KEY,
    });
  });

  // 2. API: Verify custom or system API key
  app.post("/api/verify-key", async (req, res) => {
    try {
      const { apiKey } = req.body;
      const keyToUse = apiKey || process.env.GEMINI_API_KEY;

      if (!keyToUse) {
        return res.status(400).json({ 
          valid: false, 
          message: "API Key is missing. Please configure one." 
        });
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
    // Set headers for Server-Sent Events
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    try {
      const { messages, model, config, apiKey } = req.body;
      const keyToUse = apiKey || process.env.GEMINI_API_KEY;

      if (!keyToUse) {
        res.write(`data: ${JSON.stringify({ error: "Missing API Key. Please provide one." })}\n\n`);
        return res.end();
      }

      if (!messages || !Array.isArray(messages) || messages.length === 0) {
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
