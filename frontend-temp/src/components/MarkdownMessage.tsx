import React from 'react';
import ReactMarkdown from 'react-markdown';
import { cn } from "@/lib/utils";

interface MarkdownMessageProps {
  content: string;
  isUser?: boolean;
  className?: string;
}

const MarkdownMessage: React.FC<MarkdownMessageProps> = ({ 
  content, 
  isUser = false, 
  className 
}) => {
  // Custom components for consistent styling
  const components = {
    // Headings
    h1: ({ children }: any) => (
      <h1 className="text-lg font-bold text-slate-900 mb-2">{children}</h1>
    ),
    h2: ({ children }: any) => (
      <h2 className="text-base font-semibold text-slate-800 mb-2">{children}</h2>
    ),
    h3: ({ children }: any) => (
      <h3 className="text-sm font-semibold text-slate-700 mb-1">{children}</h3>
    ),
    
    // Paragraphs
    p: ({ children }: any) => (
      <p className="text-sm leading-relaxed mb-2 last:mb-0">{children}</p>
    ),
    
    // Lists
    ul: ({ children }: any) => (
      <ul className="list-disc list-inside text-sm space-y-1 mb-2 pl-2">{children}</ul>
    ),
    ol: ({ children }: any) => (
      <ol className="list-decimal list-inside text-sm space-y-1 mb-2 pl-2">{children}</ol>
    ),
    li: ({ children }: any) => (
      <li className="text-sm leading-relaxed">{children}</li>
    ),
    
    // Emphasis
    strong: ({ children }: any) => (
      <strong className="font-semibold text-slate-900">{children}</strong>
    ),
    em: ({ children }: any) => (
      <em className="italic text-slate-700">{children}</em>
    ),
    
    // Code
    code: ({ children, inline }: any) => 
      inline ? (
        <code className="bg-slate-100 text-slate-800 px-1 py-0.5 rounded text-xs font-mono">
          {children}
        </code>
      ) : (
        <pre className="bg-slate-100 text-slate-800 p-2 rounded text-xs font-mono overflow-x-auto mb-2">
          <code>{children}</code>
        </pre>
      ),
    
    // Blockquotes
    blockquote: ({ children }: any) => (
      <blockquote className="border-l-4 border-blue-200 pl-3 italic text-slate-600 mb-2">
        {children}
      </blockquote>
    ),
    
    // Links
    a: ({ children, href }: any) => (
      <a 
        href={href}
        className="text-blue-600 hover:text-blue-800 underline"
        target="_blank"
        rel="noopener noreferrer"
      >
        {children}
      </a>
    ),
  };

  return (
    <div className={cn("prose prose-sm max-w-none", className)}>
      <ReactMarkdown components={components}>
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownMessage;


