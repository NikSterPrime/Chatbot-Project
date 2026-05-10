import { useEffect, useMemo, useRef, useState } from 'react';

const SESSION_KEY = 'chatbot_session_id';

function getSessionId() {
  const existing = window.localStorage.getItem(SESSION_KEY);
  return existing || '';
}

function saveSessionId(sessionId) {
  window.localStorage.setItem(SESSION_KEY, sessionId);
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: 'bot',
      text: 'Hi. Tell me your name and I will remember it for this session.'
    }
  ]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef(null);
  const chatLogRef = useRef(null);

  useEffect(() => {
    if (!chatLogRef.current) return;
    chatLogRef.current.scrollTo({
      top: chatLogRef.current.scrollHeight,
      behavior: 'smooth'
    });
  }, [messages]);

  const canSend = useMemo(() => input.trim().length > 0 && !busy, [input, busy]);

  async function sendMessage() {
    const text = input.trim();
    if (!text || busy) return;

    setError('');
    setBusy(true);
    setInput('');

    setMessages((prev) => [...prev, { role: 'user', text }]);

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          session_id: getSessionId() || null
        })
      });

      if (!resp.ok) {
        throw new Error(`Request failed: ${resp.status}`);
      }

      const data = await resp.json();
      if (data.session_id) {
        saveSessionId(data.session_id);
      }

      setMessages((prev) => [
        ...prev,
        {
          role: 'bot',
          text: data.response,
          meta: `${data.intent} | confidence ${Number(data.confidence).toFixed(2)}`
        }
      ]);
    } catch (e) {
      setError('Could not reach chatbot API. Make sure backend is running on port 8000.');
      setMessages((prev) => [
        ...prev,
        { role: 'bot', text: 'I am offline right now. Please try again in a moment.' }
      ]);
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  }

  async function resetSession() {
    const sessionId = getSessionId();
    if (!sessionId) return;

    try {
      await fetch(`/api/reset/${sessionId}`, { method: 'POST' });
    } catch (_e) {
      // Best-effort reset
    }

    window.localStorage.removeItem(SESSION_KEY);
    setMessages([
      {
        role: 'bot',
        text: 'Session reset. Tell me your name again if you want me to remember it.'
      }
    ]);
    inputRef.current?.focus();
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function downloadChat() {
    if (messages.length === 0) return;

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const content = messages
      .map((msg) => {
        const label = msg.role === 'user' ? 'User' : 'Bot';
        return `${label}: ${msg.text}${msg.meta ? `\nMeta: ${msg.meta}` : ''}`;
      })
      .join('\n\n');

    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `chat-export-${timestamp}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  }

  return (
    <div className="app-shell">
      <div className="halo" aria-hidden="true" />
      <main className="chat-card">
        <header className="chat-header">
          <div>
            <p className="eyebrow">Context-Aware Bot</p>
            <h1>Chatbot Console</h1>
          </div>
          <div>
            <button className="ghost" onClick={downloadChat} type="button">
              Download Chat
            </button>
            <button className="ghost" onClick={resetSession} type="button">
              Reset Session
            </button>
          </div>
        </header>

        <section ref={chatLogRef} className="chat-log" aria-live="polite">
          {messages.map((msg, idx) => (
            <article key={`${msg.role}-${idx}`} className={`bubble ${msg.role}`}>
              <p>{msg.text}</p>
              {msg.meta ? <span>{msg.meta}</span> : null}
            </article>
          ))}
        </section>

        <footer className="chat-input-row">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type your message..."
            rows={2}
          />
          <button onClick={sendMessage} disabled={!canSend} type="button">
            {busy ? 'Sending...' : 'Send'}
          </button>
        </footer>

        {error ? <p className="error">{error}</p> : null}
      </main>
    </div>
  );
}
