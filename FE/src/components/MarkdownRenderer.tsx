/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Check, Copy } from 'lucide-react';

interface MarkdownRendererProps {
  content: string;
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="markdown-body text-slate-800 leading-relaxed text-sm space-y-3 prose prose-slate max-w-none">
      <ReactMarkdown
        components={{
          // Styled paragraph
          p: ({ children }) => <p className="mb-3 last:mb-0 break-words">{children}</p>,
          
          // Styled links
          a: ({ href, children }) => (
            <a 
              href={href} 
              target="_blank" 
              rel="noopener noreferrer" 
              className="text-indigo-600 hover:text-indigo-500 underline font-medium transition"
            >
              {children}
            </a>
          ),
          
          // Styled list items
          ul: ({ children }) => <ul className="list-disc pl-5 mb-3 space-y-1">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal pl-5 mb-3 space-y-1">{children}</ol>,
          li: ({ children }) => <li className="mb-0.5">{children}</li>,
          
          // Styled headers
          h1: ({ children }) => <h1 className="text-xl font-semibold text-slate-950 mt-4 mb-2 first:mt-0">{children}</h1>,
          h2: ({ children }) => <h2 className="text-lg font-semibold text-slate-950 mt-3 mb-2 first:mt-0">{children}</h2>,
          h3: ({ children }) => <h3 className="text-md font-semibold text-slate-950 mt-2 mb-1 first:mt-0">{children}</h3>,
          
          // Styled blockquotes
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-indigo-600 bg-slate-50 pl-3 py-1 pr-1 my-3 rounded-r text-slate-600 italic">
              {children}
            </blockquote>
          ),
          
          // Custom preformatted code blocks
          pre: ({ children }) => {
            return <div className="my-4 overflow-hidden rounded-xl border border-slate-200 bg-slate-50 shadow-sm">{children}</div>;
          },
          
          // Code block rendering with language header and copy button
          code: ({ inline, className, children, ...props }: any) => {
            const match = /language-(\w+)/.exec(className || '');
            const language = match ? match[1] : '';
            const codeString = String(children).replace(/\n$/, '');
 
            if (!inline && language) {
              return (
                <CodeBlock language={language} code={codeString} />
              );
            }
 
            if (!inline && codeString.includes('\n')) {
              return (
                <CodeBlock language="code" code={codeString} />
              );
            }
 
            return (
              <code className="bg-slate-100 text-indigo-600 px-1.5 py-0.5 border border-slate-200/50 rounded font-mono text-xs" {...props}>
                {children}
              </code>
            );
          },
          
          // Styled tables
          table: ({ children }) => (
            <div className="overflow-x-auto my-4 border border-slate-200 rounded-xl shadow-sm">
              <table className="min-w-full divide-y divide-slate-200 text-left text-xs text-slate-600">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-slate-50">{children}</thead>,
          tbody: ({ children }) => <tbody className="divide-y divide-slate-200 bg-transparent">{children}</tbody>,
          tr: ({ children }) => <tr>{children}</tr>,
          th: ({ children }) => <th className="px-4 py-2 font-semibold text-slate-800">{children}</th>,
          td: ({ children }) => <td className="px-4 py-2">{children}</td>,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
 
// Inner component to handle copy state isolated from parent renders
interface CodeBlockProps {
  language: string;
  code: string;
}
 
function CodeBlock({ language, code }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
 
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text', err);
    }
  };
 
  return (
    <div className="flex flex-col">
      <div className="flex items-center justify-between px-4 py-1.5 bg-slate-100 border-b border-slate-200 text-xs text-slate-500 font-mono">
        <span>{language.toLowerCase()}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 hover:text-slate-850 transition cursor-pointer"
          title="Sao chép mã"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-indigo-600" />
              <span className="text-indigo-600 font-medium">Đã chép</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              <span>Sao chép</span>
            </>
          )}
        </button>
      </div>
      <div className="p-4 overflow-x-auto font-mono text-xs leading-relaxed text-slate-750 bg-slate-50/50">
        <pre className="m-0 bg-transparent border-0 p-0 rounded-none overflow-visible whitespace-pre">
          <code className="bg-transparent text-slate-700 p-0">{code}</code>
        </pre>
      </div>
    </div>
  );
}
