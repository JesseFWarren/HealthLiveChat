import React, { useState, useEffect } from "react";
import axios from "axios";
import "./styles.css";

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const BACKEND_URL = "https://healthlivechatbackend.onrender.com";

  // Add a welcome message when the chat loads
  useEffect(() => {
    setMessages([
      { text: "Welcome to Health Live Chat! How can I assist you today?", sender: "bot" }
    ]);
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    const userMessage = { text: question, sender: "user" };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");
    setIsTyping(true); // Show typing indicator

    try {
      const res = await axios.post(`${BACKEND_URL}/ask`, { query: question });
      const botMessage = { text: res.data.response, sender: "bot" };
      setTimeout(() => {
        setMessages((prev) => [...prev, botMessage]);
        setIsTyping(false);
      }, 1500); // Simulating bot delay for a smoother experience
    } catch (error) {
      setIsTyping(false);
      setMessages((prev) => [
        ...prev,
        { text: "Error fetching response. Try again later.", sender: "bot" }
      ]);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-box">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            {msg.text}
          </div>
        ))}
        {isTyping && <div className="message bot typing">...</div>}
      </div>
      <form className="chat-form" onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Describe your symptoms..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
};

export default Chatbot;
