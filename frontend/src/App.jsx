import { useState } from "react";
import "./App.css";

function App() {
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!message.trim()) return;

    setLoading(true);
    setResponse("");

    try {
      const res = await fetch("http://127.0.0.1:8000/agent", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          input_type: "text",
          text: message,
          student_type: "general",
          subject: "general",
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Something went wrong.");
      }

      setResponse(data.response);
    } catch (error) {
      setResponse(
        "Unable to connect to BASERA backend. Make sure the backend is running."
      );
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>BASERA</h1>
          <p>AI Accessibility Assistant</p>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          Online
        </div>
      </header>

      <main className="main">
        <section className="welcome">
          <h2>How can BASERA help you?</h2>
          <p>
            Ask a question, use your voice, or use the camera to get
            accessible educational assistance.
          </p>
        </section>

        <section className="features">
          <button className="feature-card">
            <span className="feature-icon">🎤</span>
            <strong>Voice</strong>
            <span>Talk to BASERA</span>
          </button>

          <button className="feature-card">
            <span className="feature-icon">📷</span>
            <strong>Camera</strong>
            <span>Use visual assistance</span>
          </button>

          <button className="feature-card">
            <span className="feature-icon">📚</span>
            <strong>Study</strong>
            <span>Get educational help</span>
          </button>
        </section>

        <section className="chat-container">
          <div className="chat-header">
            <span>💬</span>
            <h3>Chat with BASERA</h3>
          </div>

          <div className="response-box">
            {loading ? (
              <div className="loading">BASERA is thinking...</div>
            ) : response ? (
              <p>{response}</p>
            ) : (
              <p className="placeholder">
                Your response will appear here.
              </p>
            )}
          </div>

          <div className="input-area">
            <input
              type="text"
              placeholder="Type your question..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  sendMessage();
                }
              }}
            />

            <button
              className="send-button"
              onClick={sendMessage}
              disabled={loading}
            >
              {loading ? "..." : "Send"}
            </button>
          </div>
        </section>
      </main>

      <footer className="footer">
        <p>Designed for accessible education ♿</p>
      </footer>
    </div>
  );
}

export default App;