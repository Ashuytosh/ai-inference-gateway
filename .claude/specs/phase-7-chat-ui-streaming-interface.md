# Phase 7 — Chat UI & Streaming Interface

## Goal
Build a beautiful, dark-themed ChatGPT-like chat interface served
directly from FastAPI using Jinja2 templates, Tailwind CSS (CDN), and
vanilla JavaScript. After this phase, users interact with the gateway
through a polished browser UI with streaming text, model selection
(Auto + manual), temperature controls, metadata badges, and real-time
connection status. This is what makes the project portfolio-worthy.

IMPORTANT: Add detailed comments explaining every concept, pattern, and
"why" behind each implementation. Comments should teach the developer
what each piece does and why it exists. Treat the codebase as a learning
resource.

---

## 1. Design Language (from CLAUDE.md)

- Dark theme: bg-gray-950 main background, bg-gray-900 sidebar,
  bg-gray-800 for cards/bubbles
- Accent color: Indigo-500 to Blue-500 gradient for primary actions
- Font: Inter from Google Fonts CDN
- Rounded corners: rounded-xl on cards, rounded-2xl on containers
- Subtle glass morphism: backdrop-blur-sm, bg-opacity effects on sidebar
- Smooth transitions: transition-all duration-200
- Professional, clean, minimal — NOT colorful or playful
- Mobile responsive with Tailwind breakpoints (sm, md, lg)

---

## 2. External Dependencies (all via CDN, no npm)

Add these to base.html head:
- Tailwind CSS: https://cdn.tailwindcss.com
- Inter font: https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap
- Lucide Icons: https://unpkg.com/lucide@latest/dist/umd/lucide.min.js

---

## 3. Base Template (templates/base.html)

Standard Jinja2 base layout that all pages extend:

```html
<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}AI Inference Gateway{% endblock %}</title>
    
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'system-ui', 'sans-serif'],
                    }
                }
            }
        }
    </script>
    
    <!-- Inter Font -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Custom CSS -->
    <link rel="stylesheet" href="/static/css/custom.css">
    
    {% block head %}{% endblock %}
</head>
<body class="h-full bg-gray-950 text-gray-100 font-sans">
    {% block body %}{% endblock %}
    
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest/dist/umd/lucide.min.js"></script>
    
    {% block scripts %}{% endblock %}
</body>
</html>
```

---

## 4. Chat Page (templates/chat.html)

Extends base.html. This is the main and only page of the app.

### Overall Layout Structure:

```
┌──────────────────────────────────────────────────────────┐
│ ┌─────────────┐ ┌──────────────────────────────────────┐ │
│ │             │ │          Top Bar                      │ │
│ │             │ │  Logo  "AI Gateway"   Status Dot     │ │
│ │  Sidebar    │ ├──────────────────────────────────────┤ │
│ │             │ │                                      │ │
│ │  Model      │ │         Chat Messages Area           │ │
│ │  Dropdown   │ │                                      │ │
│ │             │ │  ┌──────────────────────────┐        │ │
│ │  Temp       │ │  │ User bubble (right)      │        │ │
│ │  Slider     │ │  └──────────────────────────┘        │ │
│ │             │ │         ┌──────────────────────────┐ │ │
│ │  System     │ │         │ AI bubble (left)         │ │ │
│ │  Prompt     │ │         │ streaming text...        │ │ │
│ │             │ │         │ [model] [2.3s] [225 tok] │ │ │
│ │             │ │         └──────────────────────────┘ │ │
│ │  New Chat   │ │                                      │ │
│ │  Button     │ ├──────────────────────────────────────┤ │
│ │             │ │         Input Area (fixed bottom)    │ │
│ │  Analytics  │ │  ┌────────────────────────┐ ┌─────┐ │ │
│ │  Link       │ │  │ Type a message...      │ │ ➤   │ │ │
│ │             │ │  └────────────────────────┘ └─────┘ │ │
│ │             │ │  Model: Auto (gemma3:4b)             │ │
│ └─────────────┘ └──────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### Sidebar (left panel, 280px wide, collapsible on mobile):

```html
<!-- Sidebar structure -->
<aside class="w-72 bg-gray-900/80 backdrop-blur border-r border-gray-800 
              flex flex-col h-full p-4 space-y-6">
    
    <!-- Logo and Title -->
    <div class="flex items-center gap-3">
        <div class="w-8 h-8 bg-gradient-to-br from-indigo-500 to-blue-500 
                    rounded-lg flex items-center justify-center">
            <i data-lucide="zap" class="w-5 h-5 text-white"></i>
        </div>
        <h1 class="text-lg font-semibold">AI Gateway</h1>
    </div>
    
    <!-- New Chat Button -->
    <button id="new-chat-btn" class="w-full py-2.5 px-4 bg-gradient-to-r 
            from-indigo-600 to-blue-600 hover:from-indigo-500 hover:to-blue-500 
            rounded-xl text-sm font-medium transition-all duration-200 
            flex items-center justify-center gap-2">
        <i data-lucide="plus" class="w-4 h-4"></i>
        New Chat
    </button>
    
    <!-- Model Selector -->
    <div class="space-y-2">
        <label class="text-xs font-medium text-gray-400 uppercase tracking-wider">
            Model
        </label>
        <select id="model-select" class="w-full bg-gray-800 border border-gray-700 
                rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 
                focus:border-transparent outline-none">
            <option value="auto">🟢 Auto (Smart Route)</option>
            <!-- Populated dynamically from /api/models -->
        </select>
    </div>
    
    <!-- Temperature Slider -->
    <div class="space-y-2">
        <div class="flex justify-between items-center">
            <label class="text-xs font-medium text-gray-400 uppercase tracking-wider">
                Temperature
            </label>
            <span id="temp-value" class="text-xs text-indigo-400 font-mono">0.7</span>
        </div>
        <input type="range" id="temp-slider" min="0" max="2" step="0.1" value="0.7"
               class="w-full accent-indigo-500">
        <div class="flex justify-between text-xs text-gray-500">
            <span>Precise</span>
            <span>Creative</span>
        </div>
    </div>
    
    <!-- Max Tokens -->
    <div class="space-y-2">
        <label class="text-xs font-medium text-gray-400 uppercase tracking-wider">
            Max Tokens
        </label>
        <input type="number" id="max-tokens" value="1024" min="1" max="4096"
               class="w-full bg-gray-800 border border-gray-700 rounded-lg 
                      px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 
                      focus:border-transparent outline-none">
    </div>
    
    <!-- System Prompt -->
    <div class="space-y-2">
        <label class="text-xs font-medium text-gray-400 uppercase tracking-wider">
            System Prompt (optional)
        </label>
        <textarea id="system-prompt" rows="3" placeholder="e.g., You are a pirate..."
                  class="w-full bg-gray-800 border border-gray-700 rounded-lg 
                         px-3 py-2 text-sm resize-none focus:ring-2 
                         focus:ring-indigo-500 focus:border-transparent outline-none">
        </textarea>
    </div>
    
    <!-- Output Format -->
    <div class="space-y-2">
        <label class="text-xs font-medium text-gray-400 uppercase tracking-wider">
            Output Format
        </label>
        <select id="output-format" class="w-full bg-gray-800 border border-gray-700 
                rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 
                focus:border-transparent outline-none">
            <option value="text">Text (default)</option>
            <option value="sentiment">Sentiment Analysis</option>
            <option value="summary">Summary</option>
            <option value="code_review">Code Review</option>
            <option value="qa">Q&A</option>
            <option value="json">Custom JSON</option>
        </select>
    </div>
    
    <!-- Connection Status (bottom of sidebar) -->
    <div class="mt-auto space-y-3">
        <!-- Analytics Link -->
        <a href="/api/analytics" target="_blank" 
           class="flex items-center gap-2 text-xs text-gray-500 hover:text-gray-300 transition">
            <i data-lucide="bar-chart-2" class="w-3.5 h-3.5"></i>
            View Analytics
        </a>
        
        <!-- API Docs Link -->
        <a href="/docs" target="_blank" 
           class="flex items-center gap-2 text-xs text-gray-500 hover:text-gray-300 transition">
            <i data-lucide="file-text" class="w-3.5 h-3.5"></i>
            API Documentation
        </a>
        
        <!-- Status Indicator -->
        <div id="connection-status" class="flex items-center gap-2 text-xs">
            <div id="status-dot" class="w-2 h-2 rounded-full bg-green-500"></div>
            <span id="status-text" class="text-gray-400">Connected</span>
        </div>
    </div>
</aside>
```

### Main Chat Area:

```html
<main class="flex-1 flex flex-col h-full">
    
    <!-- Top Bar -->
    <header class="h-14 border-b border-gray-800 flex items-center px-6 
                   bg-gray-900/50 backdrop-blur-sm">
        <!-- Mobile menu toggle (hidden on desktop) -->
        <button id="menu-toggle" class="lg:hidden mr-4">
            <i data-lucide="menu" class="w-5 h-5"></i>
        </button>
        <span class="text-sm font-medium text-gray-300">
            AI Inference Gateway
        </span>
        <span class="ml-auto text-xs text-gray-500" id="active-model-display">
            Model: Auto
        </span>
    </header>
    
    <!-- Messages Container (scrollable) -->
    <div id="messages-container" class="flex-1 overflow-y-auto px-4 md:px-8 
                                        lg:px-16 py-6 space-y-6">
        
        <!-- Empty state (shown when no messages) -->
        <div id="empty-state" class="flex flex-col items-center justify-center 
                                     h-full text-center">
            <div class="w-16 h-16 bg-gradient-to-br from-indigo-500/20 to-blue-500/20 
                        rounded-2xl flex items-center justify-center mb-4">
                <i data-lucide="message-square" class="w-8 h-8 text-indigo-400"></i>
            </div>
            <h2 class="text-lg font-medium text-gray-300 mb-2">
                Start a conversation
            </h2>
            <p class="text-sm text-gray-500 max-w-md">
                Send a message and the gateway will intelligently route it 
                to the best model. Or select a specific model from the sidebar.
            </p>
        </div>
        
        <!-- Messages will be inserted here dynamically -->
    </div>
    
    <!-- Input Area (fixed bottom) -->
    <div class="border-t border-gray-800 bg-gray-900/50 backdrop-blur-sm 
                px-4 md:px-8 lg:px-16 py-4">
        
        <div class="max-w-3xl mx-auto">
            <div class="flex gap-3 items-end">
                <div class="flex-1 relative">
                    <textarea id="message-input" rows="1" 
                              placeholder="Send a message..."
                              class="w-full bg-gray-800 border border-gray-700 rounded-xl 
                                     px-4 py-3 pr-12 text-sm resize-none 
                                     focus:ring-2 focus:ring-indigo-500 
                                     focus:border-transparent outline-none 
                                     max-h-40 overflow-y-auto"
                    ></textarea>
                </div>
                <button id="send-btn" 
                        class="w-10 h-10 bg-gradient-to-r from-indigo-600 to-blue-600 
                               hover:from-indigo-500 hover:to-blue-500 
                               rounded-xl flex items-center justify-center 
                               transition-all duration-200 disabled:opacity-50 
                               disabled:cursor-not-allowed flex-shrink-0">
                    <i data-lucide="arrow-up" class="w-5 h-5"></i>
                </button>
            </div>
            
            <!-- Current model indicator below input -->
            <div class="flex items-center gap-2 mt-2 text-xs text-gray-500">
                <span id="input-model-indicator">Model: Auto</span>
                <span>·</span>
                <span id="input-temp-indicator">Temp: 0.7</span>
            </div>
        </div>
    </div>
</main>
```

### Message Bubble Templates (created dynamically via JS):

User message bubble:
```html
<div class="flex justify-end">
    <div class="max-w-2xl bg-indigo-600/20 border border-indigo-500/30 
                rounded-2xl rounded-tr-sm px-4 py-3">
        <p class="text-sm text-gray-100 whitespace-pre-wrap">{message}</p>
    </div>
</div>
```

Assistant message bubble (with streaming + metadata):
```html
<div class="flex justify-start">
    <div class="max-w-2xl">
        <div class="bg-gray-800/80 border border-gray-700/50 
                    rounded-2xl rounded-tl-sm px-4 py-3">
            <!-- Message text (tokens appended here during streaming) -->
            <div class="text-sm text-gray-100 whitespace-pre-wrap 
                        leading-relaxed" id="msg-{id}">
                {streaming text appears here}
                <span class="inline-block w-1.5 h-4 bg-indigo-400 
                             animate-pulse ml-0.5" id="cursor-{id}">
                </span>
            </div>
        </div>
        
        <!-- Metadata badges (shown after streaming completes) -->
        <div class="flex flex-wrap gap-1.5 mt-2 ml-1" id="meta-{id}">
            <span class="px-2 py-0.5 bg-gray-800 rounded-full text-xs 
                         text-gray-400 border border-gray-700/50">
                gemma3:4b
            </span>
            <span class="px-2 py-0.5 bg-gray-800 rounded-full text-xs 
                         text-gray-400 border border-gray-700/50">
                2.3s
            </span>
            <span class="px-2 py-0.5 bg-gray-800 rounded-full text-xs 
                         text-gray-400 border border-gray-700/50">
                225 tokens
            </span>
            <span class="px-2 py-0.5 bg-gray-800 rounded-full text-xs 
                         text-indigo-400 border border-indigo-500/30">
                technical
            </span>
            <span class="px-2 py-0.5 bg-gray-800 rounded-full text-xs 
                         text-gray-500 border border-gray-700/50">
                chain-of-thought
            </span>
        </div>
    </div>
</div>
```

### Loading/Thinking indicator (shown while waiting for first token):
```html
<div class="flex justify-start" id="thinking-indicator">
    <div class="bg-gray-800/80 border border-gray-700/50 rounded-2xl 
                rounded-tl-sm px-4 py-3">
        <div class="flex items-center gap-2">
            <div class="flex gap-1">
                <div class="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" 
                     style="animation-delay: 0ms"></div>
                <div class="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" 
                     style="animation-delay: 150ms"></div>
                <div class="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" 
                     style="animation-delay: 300ms"></div>
            </div>
            <span class="text-xs text-gray-500">Thinking...</span>
        </div>
    </div>
</div>
```

---

## 5. JavaScript — Chat Logic (static/js/chat.js)

This is the brain of the frontend. All interactivity lives here.

### State Management:
```javascript
const state = {
    isStreaming: false,
    messageCount: 0,
    currentModel: "auto",
    temperature: 0.7,
    maxTokens: 1024,
    systemPrompt: "",
    outputFormat: "text",
};
```

### Core Functions:

#### initApp()
Called on page load (DOMContentLoaded):
1. Load available models from GET /api/models
2. Populate the model dropdown with options
3. Check health from GET /health
4. Update connection status indicator
5. Set up event listeners (send button, enter key, slider, etc.)
6. Start health check polling (every 30 seconds)
7. Initialize lucide icons

#### loadModels()
```javascript
async function loadModels() {
    const response = await fetch("/api/models");
    const data = await response.json();
    const select = document.getElementById("model-select");
    
    // Keep the Auto option, add real models
    data.models.forEach(model => {
        const option = document.createElement("option");
        option.value = model.name;
        // Show size and loaded status
        const loaded = model.loaded ? "●" : "○";
        option.textContent = `${loaded} ${model.name} (${model.size_gb}GB)`;
        select.appendChild(option);
    });
}
```

#### checkHealth()
```javascript
async function checkHealth() {
    try {
        const response = await fetch("/health");
        const data = await response.json();
        
        const dot = document.getElementById("status-dot");
        const text = document.getElementById("status-text");
        
        if (data.ollama_connected) {
            dot.className = "w-2 h-2 rounded-full bg-green-500";
            text.textContent = `Connected · ${data.models_loaded} model(s) loaded`;
        } else {
            dot.className = "w-2 h-2 rounded-full bg-red-500";
            text.textContent = "Ollama disconnected";
        }
    } catch {
        // Server itself is down
        dot.className = "w-2 h-2 rounded-full bg-red-500";
        text.textContent = "Server unreachable";
    }
}
```

#### sendMessage()
The main function. Called when user clicks send or presses Enter:

```javascript
async function sendMessage() {
    const input = document.getElementById("message-input");
    const prompt = input.value.trim();
    if (!prompt || state.isStreaming) return;
    
    state.isStreaming = true;
    input.value = "";
    autoResizeTextarea(input);
    
    // Hide empty state
    document.getElementById("empty-state")?.remove();
    
    // Add user message bubble
    addUserMessage(prompt);
    
    // Add thinking indicator
    showThinkingIndicator();
    
    // Prepare request
    const body = {
        prompt: prompt,
        temperature: state.temperature,
        max_tokens: state.maxTokens,
        stream: true,
    };
    
    // Model selection
    if (state.currentModel !== "auto") {
        body.model = state.currentModel;
    }
    
    // System prompt
    if (state.systemPrompt.trim()) {
        body.system_prompt = state.systemPrompt;
    }
    
    // Output format
    if (state.outputFormat !== "text") {
        body.output_format = state.outputFormat;
        // Non-text formats can't stream, use regular endpoint
        await sendNonStreamingMessage(body);
        return;
    }
    
    // Stream the response
    try {
        const response = await fetch("/api/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        
        if (!response.ok) {
            const error = await response.json();
            removeThinkingIndicator();
            addErrorMessage(error.message || "Request failed");
            state.isStreaming = false;
            return;
        }
        
        removeThinkingIndicator();
        const msgId = addAssistantMessage();
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            // Process complete SSE lines
            const lines = buffer.split("\n\n");
            buffer = lines.pop(); // keep incomplete line in buffer
            
            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;
                const jsonStr = line.slice(6); // remove "data: "
                
                try {
                    const event = JSON.parse(jsonStr);
                    if (event.done) {
                        // Final event with metadata
                        removeCursor(msgId);
                        showMetadata(msgId, event.metadata);
                    } else {
                        // Token event
                        appendToken(msgId, event.token);
                    }
                } catch (e) {
                    console.error("Failed to parse SSE event:", e);
                }
            }
        }
    } catch (error) {
        removeThinkingIndicator();
        addErrorMessage("Connection error: " + error.message);
    }
    
    state.isStreaming = false;
    scrollToBottom();
}
```

#### sendNonStreamingMessage(body)
For structured output formats that can't stream:

```javascript
async function sendNonStreamingMessage(body) {
    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        
        removeThinkingIndicator();
        
        if (!response.ok) {
            const error = await response.json();
            addErrorMessage(error.message || "Request failed");
            state.isStreaming = false;
            return;
        }
        
        const data = await response.json();
        const msgId = addAssistantMessage();
        
        // If structured output, show both raw and parsed
        if (data.parsed) {
            setMessageContent(msgId, formatStructuredOutput(data.parsed));
        } else {
            setMessageContent(msgId, data.response);
        }
        
        removeCursor(msgId);
        showMetadata(msgId, data.metadata);
        
    } catch (error) {
        removeThinkingIndicator();
        addErrorMessage("Connection error: " + error.message);
    }
    
    state.isStreaming = false;
    scrollToBottom();
}
```

#### DOM Helper Functions:

```javascript
function addUserMessage(text) {
    // Create user bubble HTML and append to messages container
}

function addAssistantMessage() {
    // Create empty assistant bubble with cursor, return unique msgId
    state.messageCount++;
    return `msg-${state.messageCount}`;
}

function appendToken(msgId, token) {
    // Append token text to the message element
    // The cursor stays at the end (blinking bar after last character)
    const el = document.getElementById(msgId);
    const cursor = document.getElementById(`cursor-${msgId}`);
    el.insertBefore(document.createTextNode(token), cursor);
    scrollToBottom();
}

function setMessageContent(msgId, text) {
    // Set full message content (for non-streaming responses)
}

function removeCursor(msgId) {
    // Remove the blinking cursor after streaming completes
    document.getElementById(`cursor-${msgId}`)?.remove();
}

function showMetadata(msgId, metadata) {
    // Create and show metadata badges below the message
    const container = document.getElementById(`meta-${msgId}`);
    
    // Model badge
    addBadge(container, metadata.model_used, "text-gray-400");
    
    // Latency badge
    addBadge(container, `${(metadata.latency_ms / 1000).toFixed(1)}s`, "text-gray-400");
    
    // Tokens badge
    addBadge(container, `${metadata.tokens_total} tokens`, "text-gray-400");
    
    // Query type badge (colored)
    addBadge(container, metadata.query_type, "text-indigo-400", "border-indigo-500/30");
    
    // Strategy badge
    if (metadata.prompt_strategy) {
        addBadge(container, metadata.prompt_strategy, "text-gray-500");
    }
    
    // Cached badge
    if (metadata.cached) {
        addBadge(container, "cached", "text-green-400", "border-green-500/30");
    }
    
    // Fallback badge
    if (metadata.fallback_used) {
        addBadge(container, "fallback", "text-amber-400", "border-amber-500/30");
    }
}

function addBadge(container, text, textColor, borderColor = "border-gray-700/50") {
    const badge = document.createElement("span");
    badge.className = `px-2 py-0.5 bg-gray-800 rounded-full text-xs ${textColor} border ${borderColor}`;
    badge.textContent = text;
    container.appendChild(badge);
}

function showThinkingIndicator() {
    // Insert the bouncing dots thinking indicator
}

function removeThinkingIndicator() {
    document.getElementById("thinking-indicator")?.remove();
}

function addErrorMessage(text) {
    // Show error as a red-tinted message bubble
    // Use red/rose colors for the border and background
}

function formatStructuredOutput(parsed) {
    // Format parsed JSON as pretty-printed, syntax-highlighted text
    return JSON.stringify(parsed, null, 2);
}

function scrollToBottom() {
    const container = document.getElementById("messages-container");
    container.scrollTop = container.scrollHeight;
}

function clearChat() {
    // Remove all messages, show empty state again
}
```

#### Event Listeners:

```javascript
// Send on button click
document.getElementById("send-btn").addEventListener("click", sendMessage);

// Send on Enter, newline on Shift+Enter
document.getElementById("message-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-resize textarea as user types
document.getElementById("message-input").addEventListener("input", function() {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 160) + "px";
});

// Temperature slider updates display
document.getElementById("temp-slider").addEventListener("input", function() {
    state.temperature = parseFloat(this.value);
    document.getElementById("temp-value").textContent = this.value;
    document.getElementById("input-temp-indicator").textContent = 
        `Temp: ${this.value}`;
});

// Model selector
document.getElementById("model-select").addEventListener("change", function() {
    state.currentModel = this.value;
    const displayText = this.value === "auto" ? "Auto" : this.value;
    document.getElementById("active-model-display").textContent = 
        `Model: ${displayText}`;
    document.getElementById("input-model-indicator").textContent = 
        `Model: ${displayText}`;
});

// Max tokens
document.getElementById("max-tokens").addEventListener("change", function() {
    state.maxTokens = parseInt(this.value) || 1024;
});

// System prompt
document.getElementById("system-prompt").addEventListener("input", function() {
    state.systemPrompt = this.value;
});

// Output format
document.getElementById("output-format").addEventListener("change", function() {
    state.outputFormat = this.value;
});

// New chat button
document.getElementById("new-chat-btn").addEventListener("click", clearChat);

// Mobile sidebar toggle
document.getElementById("menu-toggle")?.addEventListener("click", () => {
    document.getElementById("sidebar").classList.toggle("-translate-x-full");
});
```

---

## 6. Custom CSS (static/css/custom.css)

Minimal CSS for things Tailwind can't do easily:

```css
/* Hide scrollbar but keep scroll functionality */
#messages-container::-webkit-scrollbar {
    width: 6px;
}
#messages-container::-webkit-scrollbar-track {
    background: transparent;
}
#messages-container::-webkit-scrollbar-thumb {
    background: #374151;
    border-radius: 3px;
}
#messages-container::-webkit-scrollbar-thumb:hover {
    background: #4B5563;
}

/* Blinking cursor animation */
@keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
}
.animate-blink {
    animation: blink 1s infinite;
}

/* Textarea auto-resize smoothing */
#message-input {
    transition: height 0.1s ease;
}

/* Code blocks inside chat messages */
#messages-container pre {
    background: #1a1a2e;
    border: 1px solid #374151;
    border-radius: 0.5rem;
    padding: 0.75rem 1rem;
    overflow-x: auto;
    margin: 0.5rem 0;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

#messages-container code {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.8rem;
}

/* Inline code */
#messages-container p code {
    background: #1f2937;
    padding: 0.1rem 0.3rem;
    border-radius: 0.25rem;
    font-size: 0.8rem;
}

/* Smooth scroll */
#messages-container {
    scroll-behavior: smooth;
}

/* Range slider styling */
input[type="range"] {
    -webkit-appearance: none;
    height: 4px;
    border-radius: 2px;
    background: #374151;
    outline: none;
}
input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #6366f1;
    cursor: pointer;
}

/* Mobile sidebar overlay */
@media (max-width: 1023px) {
    #sidebar {
        position: fixed;
        z-index: 50;
        transition: transform 0.3s ease;
    }
    #sidebar.-translate-x-full {
        transform: translateX(-100%);
    }
}
```

---

## 7. Update Root Route (app/main.py)

Replace the plain text root route with template rendering:

```python
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})
```

---

## 8. Markdown Rendering in Chat Messages

LLM responses often contain markdown (headers, bold, code blocks,
lists). Add simple markdown-to-HTML conversion in JavaScript:

```javascript
function renderMarkdown(text) {
    // Convert markdown to HTML for display
    // Handle: **bold**, *italic*, `inline code`, ```code blocks```,
    // headers (#, ##, ###), lists (- item), links [text](url)
    
    // Code blocks first (before other processing)
    text = text.replace(/```(\w*)\n([\s\S]*?)```/g, 
        '<pre><code class="language-$1">$2</code></pre>');
    
    // Inline code
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Bold
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Italic
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Headers (only h3 and below in chat context)
    text = text.replace(/^### (.*$)/gm, '<h3 class="text-sm font-semibold mt-3 mb-1">$1</h3>');
    text = text.replace(/^## (.*$)/gm, '<h3 class="text-sm font-semibold mt-3 mb-1">$1</h3>');
    
    // Line breaks
    text = text.replace(/\n/g, '<br>');
    
    return text;
}
```

Use this when setting final message content (after streaming completes,
render the full accumulated text through renderMarkdown).

---

## 9. Verification Checklist

After implementation, ALL of these must work:

### Page loads correctly:
Open http://localhost:8000 in browser.
Should see dark theme chat interface with sidebar, empty state message,
and input area. No console errors.

### Models populate in dropdown:
The model dropdown should show "Auto (Smart Route)" plus all 5 installed
Ollama models with their sizes.

### Connection status shows green:
Bottom of sidebar should show green dot with "Connected" text.

### Send a simple message:
Type "hello" and press Enter or click send.
Should see: user bubble (right), thinking dots, then streaming AI
response (left) with tokens appearing one by one, then metadata badges.

### Streaming works visually:
Response text should appear character by character with a blinking
cursor, like ChatGPT. Not all at once.

### Metadata badges show correctly:
After response completes, small badges should appear below showing:
model name, latency, token count, query type, prompt strategy.

### Model selection works:
Select "mistral:7b" from dropdown. Send "hello".
Response should come from mistral:7b (check metadata badge).

### Auto mode shows routing:
Select "Auto" mode. Send a technical question.
Metadata should show it routed to qwen2.5-coder:7b.

### Temperature slider works:
Move temperature slider to 0.1. Value display should update.
Send a message — response should be more deterministic.

### System prompt works:
Type "You are a pirate" in system prompt box.
Send "What is Python?" — response should be in pirate speak.

### New Chat clears everything:
Click "New Chat" button. All messages should disappear,
empty state should return.

### Enter sends, Shift+Enter creates newline:
Enter key should send message.
Shift+Enter should create a new line in the input.

### Textarea auto-resizes:
Type multiple lines — textarea should grow.
After sending, should shrink back to one line.

### Structured output works:
Select "Sentiment Analysis" format. Send "I love this product!"
Should show parsed JSON response (not streaming).

### Error handling in UI:
Stop Ollama, send a message.
Should show a red error bubble, not crash the page.

### Mobile responsive:
Resize browser to mobile width.
Sidebar should collapse. Menu toggle should appear.
Chat should fill full width.

### Scrolling works:
Send multiple messages.
Chat should auto-scroll to newest message.
Manual scroll up should work without forced scroll-down.

### Links work:
"View Analytics" link should open /api/analytics in new tab.
"API Documentation" link should open /docs in new tab.

---

## 10. Files Modified in This Phase

| File                        | Action                    |
|-----------------------------|---------------------------|
| templates/base.html         | Full implementation       |
| templates/chat.html         | Full implementation       |
| static/css/custom.css       | Full implementation       |
| static/js/chat.js           | Full implementation       |
| app/main.py                 | Update root route to serve template |
| app/routers/*               | NOT touched               |
| app/services/*              | NOT touched               |