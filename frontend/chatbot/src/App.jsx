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
  const [filterOptions, setFilterOptions] = useState({ genres: [], days: [], times: [] });
  const [selectedGenre, setSelectedGenre] = useState('');
  const [selectedDay, setSelectedDay] = useState('');
  const [selectedTime, setSelectedTime] = useState('');
  const inputRef = useRef(null);
  const chatLogRef = useRef(null);

  useEffect(() => {
    fetch('/api/podcast-filters')
      .then((resp) => resp.json())
      .then((data) => setFilterOptions(data))
      .catch((_err) => {
        /* best-effort */
      });
  }, []);

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
          meta: [
            `${data.intent} | confidence ${Number(data.confidence).toFixed(2)}`,
            data.source ? `source ${data.source}` : null
          ]
            .filter(Boolean)
            .join(' | ')
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

  async function applyPodcastFilters() {
    const filters = [];
    if (selectedGenre) filters.push(selectedGenre);
    if (selectedDay) filters.push(selectedDay);
    if (selectedTime) filters.push(selectedTime);

    const query = filters.length > 0
      ? `recommend ${filters.join(' ')}`
      : 'recommend podcasts';

    setInput(query);
    await new Promise((resolve) => setTimeout(resolve, 0));
    inputRef.current?.focus();
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

        <section className="podcast-filters">
          <div className="filter-group">
            <label htmlFor="genre-select">Genre:</label>
            <select
              id="genre-select"
              value={selectedGenre}
              onChange={(e) => setSelectedGenre(e.target.value)}
            >
              <option value="">All Genres</option>
              {filterOptions.genres.map((genre) => (
                <option key={genre} value={genre}>
                  {genre}
                </option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label htmlFor="day-select">Day:</label>
            <select
              id="day-select"
              value={selectedDay}
              onChange={(e) => setSelectedDay(e.target.value)}
            >
              <option value="">Any Day</option>
              {filterOptions.days.map((day) => (
                <option key={day} value={day}>
                  {day}
                </option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label htmlFor="time-select">Time:</label>
            <select
              id="time-select"
              value={selectedTime}
              onChange={(e) => setSelectedTime(e.target.value)}
            >
              <option value="">Any Time</option>
              {filterOptions.times.map((time) => (
                <option key={time} value={time}>
                  {time}
                </option>
              ))}
            </select>
          </div>

          <button
            className="filter-button"
            onClick={applyPodcastFilters}
            disabled={busy}
            type="button"
          >
            Get Recommendations
          </button>
        </section>

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
