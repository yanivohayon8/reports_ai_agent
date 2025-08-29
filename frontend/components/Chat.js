import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const Chat = ({ timestamp }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm your AI Reports Assistant. Ask me anything about your documents! 👋",
      sender: 'ai',
      timestamp: new Date(),
      agent: 'Welcome',
      reasoning: 'Initial greeting'
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [selectedMessage, setSelectedMessage] = useState(null);
  const [showFloatingWindow, setShowFloatingWindow] = useState(false);
  const [isDarkTheme, setIsDarkTheme] = useState(true);
  const chatContainerRef = useRef(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      hour12: false 
    });
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const scrollToTop = () => {
    chatContainerRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleScroll = () => {
    if (chatContainerRef.current) {
      const { scrollTop } = chatContainerRef.current;
      setShowScrollTop(scrollTop > 100);
    }
  };

  const openFloatingWindow = (message) => {
    setSelectedMessage(message);
    setShowFloatingWindow(true);
  };

  const closeFloatingWindow = () => {
    setShowFloatingWindow(false);
    setSelectedMessage(null);
  };

  const toggleTheme = () => {
    setIsDarkTheme(!isDarkTheme);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: input,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input })
      });

      if (response.ok) {
        const data = await response.json();
        const aiMessage = {
          id: Date.now() + 1,
          text: data.answer || 'I apologize, but I couldn\'t process your request.',
          sender: 'ai',
          timestamp: new Date(),
          agent: data.agent || 'AI Assistant',
          reasoning: data.reasoning || 'Response generated',
          responseType: data.response_type || 'text',
          tableData: data.table_data || null
        };
        setMessages(prev => [...prev, aiMessage]);
      } else {
        throw new Error('Failed to get response');
      }
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'ai',
        timestamp: new Date(),
        agent: 'Error Handler',
        reasoning: 'Error occurred during processing'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const container = chatContainerRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll);
      return () => container.removeEventListener('scroll', handleScroll);
    }
  }, []);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Theme-based styling
  const themeStyles = {
    background: isDarkTheme 
      ? "bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900"
      : "bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100",
    floatingElements: isDarkTheme 
      ? ["bg-blue-600/20", "bg-purple-600/20", "bg-indigo-600/20"]
      : ["bg-blue-200", "bg-purple-200", "bg-pink-200"],
    titleGradient: isDarkTheme
      ? "from-blue-300 via-purple-300 to-pink-300"
      : "from-slate-900 via-blue-800 to-purple-800",
    taglineColor: isDarkTheme ? "text-blue-200" : "text-slate-700",
    statusBadge: isDarkTheme
      ? "bg-gray-800/80 border-gray-600/50"
      : "bg-white/80 border-white/30",
    statusText: isDarkTheme ? "text-gray-200" : "text-slate-700",
    chatContainer: isDarkTheme
      ? "bg-gray-800/90 border-gray-600/50"
      : "bg-white/90 border-white/30",
    chatHeader: isDarkTheme
      ? "from-gray-800 via-blue-800 to-purple-800 border-gray-600/50"
      : "from-slate-800 via-blue-800 to-purple-800 border-slate-200",
    messageCounter: isDarkTheme
      ? "bg-gray-700/50 border-gray-500/50 text-gray-200"
      : "bg-white/25 border-white/30 text-white",
    messagesArea: isDarkTheme ? "bg-gray-800" : "bg-white",
    aiMessage: isDarkTheme
      ? "from-gray-700 to-gray-800 text-gray-200 border-gray-600/50"
      : "from-slate-50 to-blue-50 text-slate-800 border-slate-200",
    userMessage: isDarkTheme
      ? "from-blue-600 to-purple-600 text-white border-blue-400/30"
      : "from-blue-500 to-blue-600 text-white border-blue-300/30",
    aiAvatar: isDarkTheme
      ? "border-blue-400/30"
      : "border-blue-300/30",
    agentBadge: isDarkTheme
      ? "text-gray-300 bg-gray-700/50 border-gray-600/50"
      : "text-slate-600 bg-slate-100 border-slate-200",
    timestamp: isDarkTheme
      ? "text-gray-400"
      : "text-slate-500",
    loadingIndicator: isDarkTheme
      ? "from-gray-700 to-gray-800 border-gray-600/50 text-gray-300"
      : "from-slate-50 to-blue-50 border-slate-200 text-slate-700",
    inputArea: isDarkTheme
      ? "from-gray-700/50 to-gray-800/50 border-gray-600/50"
      : "from-slate-50 to-blue-50/30 border-slate-200",
    inputField: isDarkTheme
      ? "bg-gray-700 text-gray-200 placeholder-gray-400 border-gray-600"
      : "bg-white text-slate-800 placeholder-slate-500 border-slate-200",
    clearButton: isDarkTheme
      ? "text-gray-400 hover:text-gray-300"
      : "text-slate-400 hover:text-slate-600",
    sendButton: isDarkTheme
      ? "from-blue-600 to-purple-600 border-blue-400/30"
      : "from-slate-700 via-blue-600 to-purple-600 border-blue-300/30",
    footer: isDarkTheme
      ? "bg-gray-800/80 border-gray-600/50 text-gray-300"
      : "bg-white/80 border-white/30 text-slate-700",
    footerDivider: isDarkTheme ? "bg-gray-600" : "bg-slate-300"
  };

  return (
    <div className={`min-h-screen ${themeStyles.background} p-6 relative overflow-hidden`}>
      {/* Beautiful Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className={`absolute -top-40 -right-40 w-80 h-80 ${themeStyles.floatingElements[0]} rounded-full mix-blend-multiply filter blur-xl opacity-30 animate-pulse`}></div>
        <div className={`absolute -bottom-40 -left-40 w-80 h-80 ${themeStyles.floatingElements[1]} rounded-full mix-blend-multiply filter blur-xl opacity-30 animate-pulse`} style={{ animationDelay: '2s' }}></div>
        <div className={`absolute top-40 left-40 w-80 h-80 ${themeStyles.floatingElements[2]} rounded-full mix-blend-multiply filter blur-xl opacity-30 animate-pulse`} style={{ animationDelay: '4s' }}></div>
      </div>

      <div className="max-w-5xl mx-auto relative z-10 h-screen flex flex-col">
        {/* Beautiful Header */}
        <motion.div 
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center py-6 flex-shrink-0"
        >
          <h1 className={`text-4xl font-bold bg-gradient-to-r ${themeStyles.titleGradient} bg-clip-text text-transparent mb-3 tracking-tight`} style={{ fontFamily: 'cursive, serif' }}>
            AI REPORTS ASSISTANT
          </h1>
          <p className={`text-lg ${themeStyles.taglineColor} font-medium mb-3`}>
            Your intelligent document analysis companion
          </p>
          
          {/* Status Badge and Theme Toggle */}
          <div className="flex items-center justify-center space-x-4">
            <div className={`inline-flex items-center space-x-2 ${themeStyles.statusBadge} px-4 py-2 rounded-full shadow-lg`}>
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className={`text-sm ${themeStyles.statusText} font-medium`}>
                Ready to assist
              </span>
            </div>
            
            {/* Theme Toggle Button */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={toggleTheme}
              className={`p-3 rounded-full shadow-lg border transition-all duration-300 ${
                isDarkTheme 
                  ? 'bg-gray-800/80 border-gray-600/50 text-yellow-400 hover:bg-gray-700/80' 
                  : 'bg-white/80 border-white/30 text-blue-600 hover:bg-white/90'
              }`}
              title={`Switch to ${isDarkTheme ? 'light' : 'dark'} theme`}
            >
              {isDarkTheme ? (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M3 12h1m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              ) : (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              )}
            </motion.button>
          </div>
        </motion.div>

        {/* Beautiful Chat Container */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          className={`${themeStyles.chatContainer} backdrop-blur-xl rounded-3xl shadow-2xl overflow-hidden flex-1 flex flex-col`}
        >
          {/* Enhanced Chat Header */}
          <div className={`bg-gradient-to-r ${themeStyles.chatHeader} px-8 py-6 relative overflow-hidden border-b`}>
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 to-purple-600/10"></div>
            <div className="relative z-10 flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="flex space-x-2">
                  <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse shadow-sm"></div>
                  <div className="w-3 h-3 bg-yellow-400 rounded-full animate-pulse shadow-sm" style={{ animationDelay: '1s' }}></div>
                  <div className="w-3 h-3 bg-red-400 rounded-full animate-pulse shadow-sm" style={{ animationDelay: '2s' }}></div>
                </div>
                <span className="text-white font-semibold text-lg">AI Assistant Online</span>
              </div>
              <div className={`${themeStyles.messageCounter} backdrop-blur-sm px-4 py-2 rounded-full text-sm font-medium shadow-sm`}>
                {messages.length - 1} messages
              </div>
            </div>
          </div>

          {/* Enhanced Messages Area */}
          <div 
            ref={chatContainerRef}
            className={`flex-1 overflow-y-auto p-8 space-y-6 scroll-smooth max-h-96 custom-scrollbar ${themeStyles.messagesArea}`}
          >
            <AnimatePresence>
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 20, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -20, scale: 0.95 }}
                  transition={{ duration: 0.4, ease: "easeOut" }}
                  className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`max-w-sm lg:max-w-md xl:max-w-lg ${message.sender === 'user' ? 'order-2' : 'order-1'}`}>
                    {message.sender === 'ai' && (
                      <motion.div 
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="flex items-center space-x-3 mb-3"
                      >
                        <div className={`w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center text-white text-sm font-bold shadow-lg ${themeStyles.aiAvatar}`}>
                          AI
                        </div>
                        <div className={`text-xs px-3 py-1.5 rounded-full border shadow-sm ${themeStyles.agentBadge}`}>
                          {message.agent && <span className="font-medium">{message.agent}</span>}
                          {message.reasoning && <span className="ml-2">• {message.reasoning}</span>}
                        </div>
                      </motion.div>
                    )}
                    
                    <motion.div
                      whileHover={{ scale: 1.02, y: -2 }}
                      className={`rounded-2xl px-5 py-4 shadow-lg border ${
                        message.sender === 'user'
                          ? `bg-gradient-to-r ${themeStyles.userMessage} ml-auto shadow-blue-500/25`
                          : `bg-gradient-to-r ${themeStyles.aiMessage} shadow-gray-500/10`
                      }`}
                    >
                      <div className="whitespace-pre-wrap break-words text-sm leading-relaxed font-medium">
                        {message.text}
                      </div>
                      
                      {/* Render table if this is a table response */}
                      {message.responseType === 'table' && message.tableData && (
                        <div className="mt-4">
                          <TableRenderer tableData={message.tableData} isDarkTheme={isDarkTheme} />
                        </div>
                      )}
                      <div className={`text-xs mt-3 font-medium ${themeStyles.timestamp}`}>
                        {formatTime(message.timestamp)}
                      </div>
                      
                      {/* Click to Expand Button for AI Messages */}
                      {message.sender === 'ai' && (
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={() => openFloatingWindow(message)}
                          className="mt-3 w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white text-xs font-medium py-2 px-4 rounded-xl hover:from-blue-600 hover:to-purple-700 transition-all duration-200 shadow-md hover:shadow-lg flex items-center justify-center space-x-2 group border border-blue-400/30"
                        >
                          <svg className="w-4 h-4 group-hover:animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                          </svg>
                          <span>Click to expand</span>
                        </motion.button>
                      )}
                    </motion.div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {/* Enhanced Loading Indicator */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex justify-start"
              >
                <div className={`rounded-2xl px-5 py-4 border shadow-lg ${themeStyles.loadingIndicator}`}>
                  <div className="flex items-center space-x-4">
                    <div className="flex space-x-2">
                      <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce"></div>
                      <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                    <div className="text-sm font-medium">AI is analyzing your documents...</div>
                  </div>
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Beautiful Input Area */}
          <div className={`p-6 border-t ${themeStyles.inputArea}`}>
            <form onSubmit={handleSubmit} className="flex space-x-4">
              <div className="flex-1 relative">
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask me about your documents..."
                  className={`w-full px-5 py-4 rounded-2xl border focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300 shadow-lg hover:shadow-xl focus:shadow-2xl font-medium ${themeStyles.inputField}`}
                  disabled={isLoading}
                />
                {input && (
                  <motion.button
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{ opacity: 1, scale: 1 }}
                    type="button"
                    onClick={() => setInput('')}
                    className={`absolute right-4 top-1/2 transform -translate-y-1/2 transition-colors hover:scale-110 ${themeStyles.clearButton}`}
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </motion.button>
                )}
              </div>
              <motion.button
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                type="submit"
                disabled={!input.trim() || isLoading}
                className={`px-8 py-4 bg-gradient-to-r ${themeStyles.sendButton} text-white rounded-2xl font-semibold shadow-xl hover:shadow-2xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-3 group`}
              >
                <svg className="w-5 h-5 group-hover:animate-bounce" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
                <span>Send</span>
              </motion.button>
            </form>
          </div>
        </motion.div>

        {/* Beautiful Footer */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="text-center py-6 flex-shrink-0"
        >
          <div className={`inline-flex items-center space-x-6 backdrop-blur-sm px-6 py-3 rounded-2xl border shadow-lg ${themeStyles.footer}`}>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
              <span className="text-sm font-medium">Powered by AI</span>
            </div>
            <div className={`w-px h-4 ${themeStyles.footerDivider}`}></div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
              <span className="text-sm font-medium">Built with Next.js</span>
            </div>
            <div className={`w-px h-4 ${themeStyles.footerDivider}`}></div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-sm font-medium">Styled with Tailwind</span>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Floating Window Modal */}
      <AnimatePresence>
        {showFloatingWindow && selectedMessage && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-6"
            onClick={closeFloatingWindow}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.8, y: 50 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.8, y: 50 }}
              className={`${isDarkTheme ? 'bg-gray-800 border-gray-600' : 'bg-white border-gray-200'} rounded-3xl shadow-2xl border max-w-5xl w-full max-h-[85vh] overflow-hidden`}
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div className={`bg-gradient-to-r ${isDarkTheme ? 'from-gray-900 via-blue-900 to-purple-900 border-gray-600' : 'from-slate-800 via-blue-800 to-purple-800 border-slate-200'} px-8 py-6 relative border-b`}>
                <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 to-purple-600/10"></div>
                <div className="relative z-10 flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="w-14 h-14 bg-gradient-to-r from-blue-400 to-purple-500 rounded-2xl flex items-center justify-center text-white text-xl font-bold shadow-2xl border border-blue-300/30">
                      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-white" style={{ fontFamily: 'cursive, serif' }}>
                        AI RESPONSE ANALYSIS
                      </h3>
                      <p className="text-blue-200 text-sm mt-1">
                        {selectedMessage.agent} • {formatTime(selectedMessage.timestamp)}
                      </p>
                    </div>
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.1, rotate: 90 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={closeFloatingWindow}
                    className={`w-12 h-12 backdrop-blur-sm rounded-2xl flex items-center justify-center text-white hover:bg-white/30 transition-all duration-300 border ${
                      isDarkTheme 
                        ? 'bg-gray-700/50 border-gray-500/50 hover:bg-gray-600/50 hover:border-gray-400/50' 
                        : 'bg-white/20 border-white/30 hover:bg-white/30'
                    }`}
                  >
                    <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </motion.button>
                </div>
              </div>

              {/* Modal Content */}
              <div className={`p-8 overflow-y-auto max-h-96 custom-scrollbar ${isDarkTheme ? 'bg-gray-800' : 'bg-white'}`}>
                <div className="space-y-6">
                  {/* Main Response Section */}
                  <div className={`rounded-2xl p-6 border ${
                    isDarkTheme 
                      ? 'bg-gray-700/50 border-gray-600/50' 
                      : 'bg-slate-50/80 border-slate-200/50'
                  }`}>
                    <div className="flex items-center space-x-3 mb-4">
                      <div className="w-10 h-10 bg-gradient-to-r from-blue-400 to-purple-500 rounded-xl flex items-center justify-center text-white text-lg font-bold shadow-lg border border-blue-300/30">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                      </div>
                      <h4 className={`text-lg font-semibold ${isDarkTheme ? 'text-white' : 'text-slate-800'}`} style={{ fontFamily: 'cursive, serif' }}>
                        Response Content
                      </h4>
                    </div>
                    <div className={`leading-relaxed text-base rounded-xl p-4 border ${
                      isDarkTheme 
                        ? 'text-gray-200 bg-gray-800/50 border-gray-600/30' 
                        : 'text-slate-700 bg-white/80 border-slate-200/50'
                    }`}>
                      {selectedMessage.text}
                    </div>
                  </div>
                  
                  {/* AI Reasoning Section */}
                  {selectedMessage.reasoning && (
                    <div className={`rounded-2xl p-6 border ${
                      isDarkTheme 
                        ? 'bg-gray-700/50 border-gray-600/50' 
                        : 'bg-slate-50/80 border-slate-200/50'
                    }`}>
                      <div className="flex items-center space-x-3 mb-4">
                        <div className="w-10 h-10 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-xl flex items-center justify-center text-white text-lg font-bold shadow-lg border border-yellow-300/30">
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                          </svg>
                        </div>
                        <h4 className={`text-lg font-semibold ${isDarkTheme ? 'text-white' : 'text-slate-800'}`} style={{ fontFamily: 'cursive, serif' }}>
                          AI Reasoning
                        </h4>
                      </div>
                      <div className={`rounded-xl p-4 border ${
                        isDarkTheme 
                          ? 'text-gray-200 bg-gray-800/50 border-gray-600/30' 
                          : 'text-slate-700 bg-white/80 border-slate-200/50'
                      }`}>
                        {selectedMessage.reasoning}
                      </div>
                    </div>
                  )}

                  {/* Beautiful Table Visualization (Modal) */}
                  {selectedMessage.responseType === 'table' && selectedMessage.tableData && (
                    <div className={`rounded-2xl p-6 border ${
                      isDarkTheme 
                        ? 'bg-gray-700/50 border-gray-600/50' 
                        : 'bg-slate-50/80 border-slate-200/50'
                    }`}>
                      <div className="flex items-center space-x-3 mb-4">
                        <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center text-white text-lg font-bold shadow-lg border border-blue-300/30">
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
                          </svg>
                        </div>
                        <h4 className={`text-lg font-semibold ${isDarkTheme ? 'text-white' : 'text-slate-800'}`} style={{ fontFamily: 'cursive, serif' }}>
                          Table Visualization
                        </h4>
                      </div>
                      <div className="mt-2">
                        <TableRenderer tableData={selectedMessage.tableData} isDarkTheme={isDarkTheme} />
                      </div>
                    </div>
                  )}

                  {/* Process Flow Visualization */}
                  <div className={`rounded-2xl p-6 border ${
                    isDarkTheme 
                      ? 'bg-gray-700/50 border-gray-600/50' 
                      : 'bg-slate-50/80 border-slate-200/50'
                  }`}>
                    <div className="flex items-center space-x-3 mb-4">
                      <div className="w-10 h-10 bg-gradient-to-r from-green-400 to-teal-500 rounded-xl flex items-center justify-center text-white text-lg font-bold shadow-lg border border-green-300/30">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                        </svg>
                      </div>
                      <h4 className={`text-lg font-semibold ${isDarkTheme ? 'text-white' : 'text-slate-800'}`} style={{ fontFamily: 'cursive, serif' }}>
                        Process Flow
                      </h4>
                    </div>
                    <div className="flex items-center justify-center space-x-4">
                      <div className="flex items-center space-x-2">
                        <div className="w-8 h-8 bg-blue-400 rounded-lg flex items-center justify-center text-white text-sm font-bold">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                        </div>
                        <span className={`text-sm ${isDarkTheme ? 'text-blue-200' : 'text-blue-700'}`} style={{ fontFamily: 'cursive, serif' }}>Query</span>
                      </div>
                      <svg className={`w-6 h-6 ${isDarkTheme ? 'text-gray-400' : 'text-slate-400'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                      </svg>
                      <div className="flex items-center space-x-2">
                        <div className="w-8 h-8 bg-purple-400 rounded-lg flex items-center justify-center text-white text-sm font-bold">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                          </svg>
                        </div>
                        <span className={`text-sm ${isDarkTheme ? 'text-purple-200' : 'text-purple-700'}`} style={{ fontFamily: 'cursive, serif' }}>Analysis</span>
                      </div>
                      <svg className={`w-6 h-6 ${isDarkTheme ? 'text-gray-400' : 'text-slate-400'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                      </svg>
                      <div className="flex items-center space-x-2">
                        <div className="w-8 h-8 bg-green-400 rounded-lg flex items-center justify-center text-white text-sm font-bold">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        </div>
                        <span className={`text-sm ${isDarkTheme ? 'text-green-200' : 'text-green-700'}`} style={{ fontFamily: 'cursive, serif' }}>Response</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Modal Footer */}
              <div className={`px-8 py-6 border-t flex justify-end space-x-4 ${
                isDarkTheme 
                  ? 'bg-gray-900 border-gray-600' 
                  : 'bg-slate-50 border-slate-200'
              }`}>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={closeFloatingWindow}
                  className={`px-6 py-3 rounded-xl font-medium transition-all duration-200 shadow-md hover:shadow-lg border ${
                    isDarkTheme 
                      ? 'bg-gray-600 text-white hover:bg-gray-500 border-gray-500/50' 
                      : 'bg-slate-600 text-white hover:bg-slate-500 border-slate-500/50'
                  }`}
                >
                  Close
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => {
                    navigator.clipboard.writeText(selectedMessage.text);
                    // You could add a toast notification here
                  }}
                  className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl font-medium hover:from-blue-600 hover:to-purple-700 transition-all duration-200 shadow-md hover:shadow-lg flex items-center space-x-2 border border-blue-400/30"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  <span>Copy</span>
                </motion.button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

// Table Renderer Component
const TableRenderer = ({ tableData, isDarkTheme }) => {
  if (tableData.error) {
    return (
      <div className={`rounded-xl p-4 border-2 ${isDarkTheme ? 'bg-yellow-900/10 border-yellow-500/30' : 'bg-yellow-50 border-yellow-200'}`}>
        <div className="flex items-center space-x-3">
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${isDarkTheme ? 'bg-yellow-600/30 text-yellow-300' : 'bg-yellow-100 text-yellow-600'}`}>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div>
            <p className={`text-sm font-medium ${isDarkTheme ? 'text-yellow-200' : 'text-yellow-700'}`}>
              Table data not available
            </p>
            <p className={`text-xs ${isDarkTheme ? 'text-yellow-300/70' : 'text-yellow-600/70'}`}>
              The AI provided a text response instead of structured table data
            </p>
          </div>
        </div>
      </div>
    );
  }

  // If we have parsed tables, render them
  if (tableData.parsed_tables && tableData.parsed_tables.length > 0) {
    return (
      <div className="space-y-6">
        {tableData.parsed_tables.map((table, index) => (
          <div key={index} className={`rounded-2xl border-2 overflow-hidden shadow-xl ${isDarkTheme ? 'border-blue-500/30 bg-gray-800/50' : 'border-blue-200 bg-white/80'}`}>
            {/* Enhanced Table Header */}
            <div className={`px-6 py-4 border-b-2 ${isDarkTheme ? 'bg-gradient-to-r from-blue-600/20 to-purple-600/20 border-blue-500/30' : 'bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200'}`}>
              <div className="flex items-center justify-between">
                <h4 className={`text-lg font-bold ${isDarkTheme ? 'text-white' : 'text-gray-800'}`}>
                  📊 Data Table
                </h4>
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${isDarkTheme ? 'bg-blue-600/30 text-blue-200 border border-blue-500/50' : 'bg-blue-100 text-blue-700 border border-blue-200'}`}>
                  {table.type.toUpperCase()}
                </span>
              </div>
            </div>
            
            {/* Enhanced Table Content */}
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className={`${isDarkTheme ? 'bg-gray-700/50' : 'bg-gray-50'}`}>
                  <tr>
                    {table.headers.map((header, headerIndex) => (
                      <th key={headerIndex} className={`px-6 py-4 text-left text-sm font-bold ${isDarkTheme ? 'text-blue-200 border-gray-600' : 'text-blue-700 border-gray-200'} border-r last:border-r-0 uppercase tracking-wide`}>
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {table.rows.map((row, rowIndex) => (
                    <tr key={rowIndex} className={`${rowIndex % 2 === 0 ? (isDarkTheme ? 'bg-gray-700/30' : 'bg-white') : (isDarkTheme ? 'bg-gray-800/30' : 'bg-gray-50')} hover:${isDarkTheme ? 'bg-blue-600/20' : 'bg-blue-50'} transition-all duration-200 group`}>
                      {table.headers.map((header, headerIndex) => (
                        <td key={headerIndex} className={`px-6 py-4 text-sm font-medium ${isDarkTheme ? 'text-gray-200 border-gray-600' : 'text-gray-700 border-gray-200'} border-r last:border-r-0 group-hover:${isDarkTheme ? 'text-white' : 'text-gray-900'}`}>
                          {row[header] || '-'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            {/* Table Footer */}
            <div className={`px-6 py-3 ${isDarkTheme ? 'bg-gray-800/30 border-t border-gray-600/30' : 'bg-gray-50 border-t border-gray-200'}`}>
              <div className="flex items-center justify-between text-xs">
                <span className={`${isDarkTheme ? 'text-gray-400' : 'text-gray-500'}`}>
                  {table.rows.length} row{table.rows.length !== 1 ? 's' : ''} • {table.headers.length} column{table.headers.length !== 1 ? 's' : ''}
                </span>
                <span className={`${isDarkTheme ? 'text-blue-300' : 'text-blue-600'}`}>
                  ✓ Parsed successfully
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  // Fallback: show raw content
  return (
    <div className={`rounded-lg border ${isDarkTheme ? 'border-gray-600' : 'border-gray-200'}`}>
      <div className={`px-4 py-3 border-b ${isDarkTheme ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'}`}>
        <h4 className={`font-semibold ${isDarkTheme ? 'text-white' : 'text-gray-800'}`}>
          Raw Table Data
        </h4>
      </div>
      <div className={`p-4 ${isDarkTheme ? 'bg-gray-800' : 'bg-white'}`}>
        <pre className={`text-sm whitespace-pre-wrap ${isDarkTheme ? 'text-gray-200' : 'text-gray-700'}`}>
          {tableData.raw_content}
        </pre>
      </div>
    </div>
  );
};

export default Chat;
