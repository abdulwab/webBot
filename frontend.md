# RAG Chatbot Frontend Integration Guide

## Overview

This guide provides instructions for integrating the RAG (Retrieval-Augmented Generation) chatbot API into a Next.js frontend application. The chatbot allows users to process websites and ask questions about their content using an elegant, modern UI.

## API Endpoints

The RAG chatbot API is available at: `https://webbot-production-c4e6.up.railway.app`

### Available Endpoints:

1. **Process Website**: `POST /process-website`
   - Processes a website by crawling, chunking, and storing content in a vector database
   - Parameters: `url`, `max_pages`, `max_depth`

2. **Query**: `POST /query`
   - Queries the RAG system with a user question
   - Parameters: `query`

## Setup Instructions

### Step 1: Set Up Next.js Project

If you don't have a Next.js project yet:

```bash
npx create-next-app@latest rag-chatbot-frontend
cd rag-chatbot-frontend
```

### Step 2: Install Required Dependencies

```bash
npm install axios @chatscope/chat-ui-kit-react react-icons
# Or with yarn
yarn add axios @chatscope/chat-ui-kit-react react-icons
```

### Step 3: Configure Environment Variables

Create a `.env.local` file in your project root:

```
NEXT_PUBLIC_API_URL=https://webbot-production-c4e6.up.railway.app
```

## Implementation Guide

### Step 1: Create API Client

Create a file `lib/api-client.js`:

```javascript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://webbot-production-c4e6.up.railway.app';

// Process a website
export async function processWebsite(url, maxPages = 5, maxDepth = 1) {
  const response = await fetch(`${API_BASE_URL}/process-website`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, max_pages: maxPages, max_depth: maxDepth }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to process website');
  }
  
  return response.json();
}

// Query the RAG system
export async function queryRag(query) {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to query RAG system');
  }
  
  return response.json();
}
```

### Step 2: Create Website Processing Component

Create a component for processing websites:

```javascript
// components/WebsiteProcessor.jsx
'use client';

import { useState } from 'react';
import { processWebsite } from '../lib/api-client';

export default function WebsiteProcessor({ onProcessed }) {
  const [url, setUrl] = useState('');
  const [maxPages, setMaxPages] = useState(5);
  const [maxDepth, setMaxDepth] = useState(1);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsProcessing(true);
    setError(null);
    setResult(null);

    try {
      const response = await processWebsite(url, maxPages, maxDepth);
      setResult(response);
      if (onProcessed) onProcessed(url);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  // Render form with URL input, max pages, max depth, and submit button
  // Include loading state, error display, and success message
}
```

### Step 3: Create Chat Interface Component

Create a chat interface using @chatscope/chat-ui-kit-react:

```javascript
// components/ChatInterface.jsx
'use client';

import { useState, useEffect } from 'react';
import { queryRag } from '../lib/api-client';
import { 
  MainContainer, ChatContainer, MessageList, Message, 
  MessageInput, TypingIndicator 
} from '@chatscope/chat-ui-kit-react';
import '@chatscope/chat-ui-kit-styles/dist/default/styles.min.css';

export default function ChatInterface({ processedUrl }) {
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);

  // Add welcome message when a new URL is processed
  useEffect(() => {
    if (processedUrl) {
      setMessages([
        {
          message: `Website processed: ${processedUrl}. What would you like to know about it?`,
          sender: 'system',
          direction: 'incoming',
          position: 'single'
        }
      ]);
    }
  }, [processedUrl]);

  const handleSend = async (query) => {
    // Add user message
    const userMessage = {
      message: query,
      sender: 'user',
      direction: 'outgoing',
      position: 'single'
    };
    setMessages(prev => [...prev, userMessage]);
    setIsTyping(true);
    
    try {
      // Query RAG API
      const response = await queryRag(query);
      
      // Add AI response
      const botMessage = {
        message: response.answer,
        sender: 'assistant',
        direction: 'incoming',
        position: 'single'
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      // Add error message
      const errorMessage = {
        message: `Error: ${error.message}`,
        sender: 'system',
        direction: 'incoming',
        position: 'single'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  // Render chat UI with message history, typing indicator, and input field
}
```

### Step 4: Create Main Page

Combine the components in your main page:

```javascript
// app/page.jsx
'use client';

import { useState } from 'react';
import WebsiteProcessor from '../components/WebsiteProcessor';
import ChatInterface from '../components/ChatInterface';

export default function Home() {
  const [processedUrl, setProcessedUrl] = useState('');

  const handleProcessed = (url) => {
    setProcessedUrl(url);
  };

  return (
    <main className="container mx-auto p-4">
      <h1 className="text-3xl font-bold text-center mb-8">
        Website RAG Chatbot
      </h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">
            Step 1: Process a Website
          </h2>
          <WebsiteProcessor onProcessed={handleProcessed} />
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-md h-[600px]">
          <h2 className="text-xl font-semibold mb-4">
            Step 2: Ask Questions
          </h2>
          <ChatInterface processedUrl={processedUrl} />
        </div>
      </div>
    </main>
  );
}
```

## Styling Guide

### Option 1: Using Tailwind CSS

If your project uses Tailwind CSS, add these styles to your components for a clean, modern look:

- Use `bg-white`, `rounded-lg`, and `shadow-md` for card containers
- Use `text-3xl font-bold` for main headings
- Use `text-xl font-semibold` for section headings
- Use `bg-blue-600 text-white rounded px-4 py-2` for primary buttons
- Use `bg-gray-200 rounded px-4 py-2` for secondary buttons
- Add hover states with `hover:bg-blue-700` for primary buttons

### Option 2: Using ChatScope Default Styles

The ChatScope UI kit comes with default styles that provide a professional chat interface:

- Import the styles: `import '@chatscope/chat-ui-kit-styles/dist/default/styles.min.css';`
- Customize colors using CSS variables in your global CSS file

## Advanced Customization

### Persistent Chat History

To maintain chat history between page refreshes:

1. Use `localStorage` to save and retrieve messages
2. Implement in the ChatInterface component:

```javascript
// Save messages to localStorage
useEffect(() => {
  if (messages.length > 0) {
    localStorage.setItem('chatHistory', JSON.stringify(messages));
  }
}, [messages]);

// Load messages from localStorage on component mount
useEffect(() => {
  const savedMessages = localStorage.getItem('chatHistory');
  if (savedMessages) {
    setMessages(JSON.parse(savedMessages));
  }
}, []);
```

### Message Attachments

To enhance the chat with file attachments or rich media:

1. Use the `Attachment` component from ChatScope
2. Implement file upload functionality
3. Display images, PDFs, or other media inline with messages

### User Authentication

To add user authentication:

1. Implement authentication using NextAuth.js
2. Store user-specific chat histories
3. Personalize the chat experience based on user preferences

## Deployment

1. Build your Next.js application:
   ```bash
   npm run build
   ```

2. Deploy to your preferred hosting platform:
   - Vercel (recommended for Next.js)
   - Netlify
   - AWS Amplify
   - GitHub Pages

3. Set environment variables in your deployment platform's dashboard

## Troubleshooting

### Common Issues and Solutions

1. **CORS Issues**
   - The API has CORS enabled for specific origins including:
     - https://rag-forntend.vercel.app
     - https://rag-frontend.vercel.app
     - http://localhost:3000
     - All Vercel app domains (via regex pattern)
   - If you're getting CORS errors, ensure your frontend domain is included in the allowed origins
   - For development, make sure you're using http://localhost:3000
   - If deploying to a different domain, you'll need to update the CORS configuration in the backend API

2. **API Connection Errors**
   - Check if the API URL is correct
   - Verify network connectivity
   - Ensure CORS is properly configured

3. **Slow Response Times**
   - Implement loading states
   - Consider adding a timeout for API requests
   - Optimize the number of requests

4. **UI Rendering Issues**
   - Check browser compatibility
   - Verify CSS imports
   - Test on different screen sizes

## Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [ChatScope UI Kit Documentation](https://chatscope.io/docs/)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [RAG API Documentation](https://webbot-production-c4e6.up.railway.app/docs) 