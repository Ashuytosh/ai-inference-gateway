/**
 * chat.js -- all frontend interactivity for the AI Inference Gateway
 * chat UI. No framework, no build step: plain DOM APIs and `fetch`.
 *
 * The three things this file has to get right:
 *   1. Streaming: read Server-Sent Events off a fetch() ReadableStream
 *      and paint tokens onto the page as they arrive (not all at once).
 *   2. State: keep the sidebar controls (model, temperature, etc.) in
 *      sync with what actually gets sent in each request.
 *   3. Never let a network hiccup or a malformed response crash the
 *      page -- always degrade to a visible error bubble instead.
 */

// ---------------------------------------------------------------------
// State
// ---------------------------------------------------------------------
// Single source of truth for "what would the next request look like".
// Sidebar control listeners mutate this; sendMessage() reads from it.
const state = {
    isStreaming: false,
    messageCount: 0,
    currentModel: "auto",
    temperature: 0.7,
    maxTokens: 1024,
    systemPrompt: "",
    outputFormat: "text",
};

// ---------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", initApp);

function initApp() {
    loadModels();
    checkHealth();
    setInterval(checkHealth, 30000);
    setupEventListeners();
    // Lucide replaces every <i data-lucide="..."> with an inline SVG.
    // Must run after the DOM (including this template's static markup)
    // exists; dynamically-inserted icons (message bubbles) trigger it
    // again themselves.
    if (window.lucide) {
        lucide.createIcons();
    }
}

// ---------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------
async function loadModels() {
    try {
        const response = await fetch("/api/models");
        const data = await response.json();
        const select = document.getElementById("model-select");

        data.models.forEach((model) => {
            const option = document.createElement("option");
            option.value = model.name;
            const loaded = model.loaded ? "●" : "○"; // ● / ○
            const size = model.size_gb != null ? `${model.size_gb}GB` : "?GB";
            option.textContent = `${loaded} ${model.name} (${size})`;
            select.appendChild(option);
        });
    } catch (error) {
        console.error("Failed to load models:", error);
    }
}

async function checkHealth() {
    const dot = document.getElementById("status-dot");
    const text = document.getElementById("status-text");

    try {
        const response = await fetch("/health");
        const data = await response.json();

        if (data.ollama_connected) {
            dot.className = "w-2 h-2 rounded-full bg-green-500";
            text.textContent = `Connected · ${data.models_loaded} model(s) loaded`;
        } else {
            dot.className = "w-2 h-2 rounded-full bg-red-500";
            text.textContent = "Ollama disconnected";
        }
    } catch {
        // The FastAPI server itself didn't respond at all.
        dot.className = "w-2 h-2 rounded-full bg-red-500";
        text.textContent = "Server unreachable";
    }
}

// ---------------------------------------------------------------------
// Sending messages
// ---------------------------------------------------------------------
async function sendMessage() {
    const input = document.getElementById("message-input");
    const prompt = input.value.trim();
    if (!prompt || state.isStreaming) return;

    setSendingUI(true);
    input.value = "";
    autoResizeTextarea(input);

    document.getElementById("empty-state")?.remove();
    addUserMessage(prompt);
    showThinkingIndicator();

    const body = {
        prompt: prompt,
        temperature: state.temperature,
        max_tokens: state.maxTokens,
        stream: true,
    };

    if (state.currentModel !== "auto") {
        body.model = state.currentModel;
    }
    if (state.systemPrompt.trim()) {
        body.system_prompt = state.systemPrompt;
    }

    // Structured output formats can't stream (the backend rejects
    // /api/chat/stream with a 400 when output_format != "text"), so
    // route those through the plain JSON endpoint instead.
    if (state.outputFormat !== "text") {
        body.output_format = state.outputFormat;
        await sendNonStreamingMessage(body);
        return;
    }

    try {
        const response = await fetch("/api/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });

        if (!response.ok) {
            const message = await readErrorMessage(response);
            removeThinkingIndicator();
            addErrorMessage(message);
            setSendingUI(false);
            return;
        }

        removeThinkingIndicator();
        const msgId = addAssistantMessage();
        let fullText = "";

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // SSE events are separated by a blank line ("\n\n"). Keep
            // whatever's left after the last full separator in the
            // buffer -- it's a partial event that hasn't fully arrived.
            const lines = buffer.split("\n\n");
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;
                const jsonStr = line.slice(6);

                try {
                    const event = JSON.parse(jsonStr);
                    if (event.done) {
                        removeCursor(msgId);
                        setMessageContent(msgId, renderMarkdown(fullText));
                        showMetadata(msgId, event.metadata);
                    } else {
                        fullText += event.token;
                        appendToken(msgId, event.token);
                    }
                } catch (e) {
                    console.error("Failed to parse SSE event:", e, jsonStr);
                }
            }
        }
    } catch (error) {
        removeThinkingIndicator();
        addErrorMessage("Connection error: " + error.message);
    }

    setSendingUI(false);
    scrollToBottom();
}

async function sendNonStreamingMessage(body) {
    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });

        removeThinkingIndicator();

        if (!response.ok) {
            const message = await readErrorMessage(response);
            addErrorMessage(message);
            setSendingUI(false);
            return;
        }

        const data = await response.json();
        const msgId = addAssistantMessage();

        if (data.parsed) {
            setMessageContent(msgId, formatStructuredOutput(data.parsed));
        } else {
            setMessageContent(msgId, renderMarkdown(data.response));
        }

        removeCursor(msgId);
        showMetadata(msgId, data.metadata);
    } catch (error) {
        removeThinkingIndicator();
        addErrorMessage("Connection error: " + error.message);
    }

    setSendingUI(false);
    scrollToBottom();
}

/** Extracts a human-readable message from a failed fetch Response,
 *  falling back gracefully if the body isn't the expected ErrorResponse
 *  JSON shape (e.g. a raw 429 from a proxy, or a non-JSON 500 page). */
async function readErrorMessage(response) {
    try {
        const error = await response.json();
        return error.message || `Request failed (${response.status})`;
    } catch {
        return `Request failed (${response.status})`;
    }
}

function setSendingUI(isSending) {
    state.isStreaming = isSending;
    document.getElementById("send-btn").disabled = isSending;
}

// ---------------------------------------------------------------------
// DOM construction helpers
// ---------------------------------------------------------------------
function addUserMessage(text) {
    const container = document.getElementById("messages-container");
    const wrapper = document.createElement("div");
    wrapper.className = "flex justify-end";
    wrapper.innerHTML = `
        <div class="max-w-2xl bg-indigo-600/20 border border-indigo-500/30
                    rounded-2xl rounded-tr-sm px-4 py-3">
            <p class="text-sm text-gray-100 whitespace-pre-wrap"></p>
        </div>`;
    // Set text via textContent (not innerHTML) so user input can never
    // be interpreted as markup -- this is the one bubble type that
    // holds untrusted, unrendered text.
    wrapper.querySelector("p").textContent = text;
    container.appendChild(wrapper);
    scrollToBottom();
}

function addAssistantMessage() {
    state.messageCount++;
    const msgId = `msg-${state.messageCount}`;

    const container = document.getElementById("messages-container");
    const wrapper = document.createElement("div");
    wrapper.className = "flex justify-start";
    wrapper.id = `wrapper-${msgId}`;
    wrapper.innerHTML = `
        <div class="max-w-2xl w-full">
            <div class="bg-gray-800/80 border border-gray-700/50
                        rounded-2xl rounded-tl-sm px-4 py-3">
                <div class="text-sm text-gray-100 whitespace-pre-wrap
                            leading-relaxed" id="${msgId}">
                    <span class="inline-block w-1.5 h-4 bg-indigo-400
                                 animate-blink align-middle" id="cursor-${msgId}"></span>
                </div>
            </div>
            <div class="flex flex-wrap gap-1.5 mt-2 ml-1" id="meta-${msgId}"></div>
        </div>`;
    container.appendChild(wrapper);
    scrollToBottom();
    return msgId;
}

function appendToken(msgId, token) {
    const el = document.getElementById(msgId);
    const cursor = document.getElementById(`cursor-${msgId}`);
    el.insertBefore(document.createTextNode(token), cursor);
    scrollToBottom();
}

function setMessageContent(msgId, html) {
    // Used once streaming/non-streaming generation is fully complete,
    // so the rendered-markdown HTML replaces the raw accumulated text.
    const el = document.getElementById(msgId);
    const cursor = document.getElementById(`cursor-${msgId}`);
    el.innerHTML = html;
    if (cursor) el.appendChild(cursor);
}

function removeCursor(msgId) {
    document.getElementById(`cursor-${msgId}`)?.remove();
}

function showMetadata(msgId, metadata) {
    const container = document.getElementById(`meta-${msgId}`);
    if (!container || !metadata) return;

    addBadge(container, metadata.model_used, "text-gray-400");
    addBadge(container, `${(metadata.latency_ms / 1000).toFixed(1)}s`, "text-gray-400");
    addBadge(container, `${metadata.tokens_total} tokens`, "text-gray-400");
    addBadge(container, metadata.query_type, "text-indigo-400", "border-indigo-500/30");

    if (metadata.prompt_strategy) {
        addBadge(container, metadata.prompt_strategy, "text-gray-500");
    }
    if (metadata.cached) {
        addBadge(container, "cached", "text-green-400", "border-green-500/30");
    }
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
    const container = document.getElementById("messages-container");
    const wrapper = document.createElement("div");
    wrapper.className = "flex justify-start";
    wrapper.id = "thinking-indicator";
    wrapper.innerHTML = `
        <div class="bg-gray-800/80 border border-gray-700/50 rounded-2xl
                    rounded-tl-sm px-4 py-3">
            <div class="flex items-center gap-2">
                <div class="flex gap-1">
                    <div class="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style="animation-delay: 0ms"></div>
                    <div class="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style="animation-delay: 150ms"></div>
                    <div class="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style="animation-delay: 300ms"></div>
                </div>
                <span class="text-xs text-gray-500">Thinking...</span>
            </div>
        </div>`;
    container.appendChild(wrapper);
    scrollToBottom();
}

function removeThinkingIndicator() {
    document.getElementById("thinking-indicator")?.remove();
}

function addErrorMessage(text) {
    const container = document.getElementById("messages-container");
    const wrapper = document.createElement("div");
    wrapper.className = "flex justify-start";
    wrapper.innerHTML = `
        <div class="max-w-2xl bg-rose-950/40 border border-rose-500/30
                    rounded-2xl rounded-tl-sm px-4 py-3 flex items-start gap-2">
            <i data-lucide="alert-triangle" class="w-4 h-4 text-rose-400 mt-0.5 shrink-0"></i>
            <p class="text-sm text-rose-300 whitespace-pre-wrap"></p>
        </div>`;
    wrapper.querySelector("p").textContent = text;
    container.appendChild(wrapper);
    if (window.lucide) lucide.createIcons();
    scrollToBottom();
}

function formatStructuredOutput(parsed) {
    return `<pre><code>${escapeHtml(JSON.stringify(parsed, null, 2))}</code></pre>`;
}

function scrollToBottom() {
    const container = document.getElementById("messages-container");
    container.scrollTop = container.scrollHeight;
}

function clearChat() {
    const container = document.getElementById("messages-container");
    container.innerHTML = `
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
        </div>`;
    state.messageCount = 0;
    if (window.lucide) lucide.createIcons();
}

// ---------------------------------------------------------------------
// Markdown rendering
// ---------------------------------------------------------------------
// LLM output often contains basic markdown. This is a small, dependency-
// free converter -- not a full CommonMark implementation, just enough
// for the constructs models actually produce in chat responses.
function renderMarkdown(text) {
    let html = escapeHtml(text);

    // Fenced code blocks first, before any other rule can mangle their
    // contents (e.g. turning `**` inside a code sample into <strong>).
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_match, lang, code) => {
        const langClass = lang ? ` class="language-${lang}"` : "";
        return `<pre><code${langClass}>${code}</code></pre>`;
    });

    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");

    html = html.replace(/^### (.*$)/gm, '<h3 class="text-sm font-semibold mt-3 mb-1">$1</h3>');
    html = html.replace(/^## (.*$)/gm, '<h3 class="text-sm font-semibold mt-3 mb-1">$1</h3>');
    html = html.replace(/^# (.*$)/gm, '<h3 class="text-base font-semibold mt-3 mb-1">$1</h3>');

    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener" class="text-indigo-400 underline hover:text-indigo-300">$1</a>');

    // List items: turn consecutive "- foo" lines into <li>, and wrap
    // runs of them in a single <ul>.
    html = html.replace(/^- (.*$)/gm, "<li>$1</li>");
    html = html.replace(/(<li>.*<\/li>\n?)+/g, (match) =>
        `<ul class="list-disc list-inside space-y-0.5 my-1">${match}</ul>`);

    html = html.replace(/\n/g, "<br>");
    // <br> right after a block element (pre/ul/h3) reads as an extra
    // blank line -- strip those for tighter spacing.
    html = html.replace(/(<\/(?:pre|ul|h3)>)<br>/g, "$1");

    return html;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ---------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------
function setupEventListeners() {
    document.getElementById("send-btn").addEventListener("click", sendMessage);

    const input = document.getElementById("message-input");
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    input.addEventListener("input", function () {
        autoResizeTextarea(this);
    });

    document.getElementById("temp-slider").addEventListener("input", function () {
        state.temperature = parseFloat(this.value);
        document.getElementById("temp-value").textContent = this.value;
        document.getElementById("input-temp-indicator").textContent = `Temp: ${this.value}`;
    });

    document.getElementById("model-select").addEventListener("change", function () {
        state.currentModel = this.value;
        const displayText = this.value === "auto" ? "Auto" : this.value;
        document.getElementById("active-model-display").textContent = `Model: ${displayText}`;
        document.getElementById("input-model-indicator").textContent = `Model: ${displayText}`;
    });

    document.getElementById("max-tokens").addEventListener("change", function () {
        state.maxTokens = parseInt(this.value, 10) || 1024;
    });

    document.getElementById("system-prompt").addEventListener("input", function () {
        state.systemPrompt = this.value;
    });

    document.getElementById("output-format").addEventListener("change", function () {
        state.outputFormat = this.value;
    });

    document.getElementById("new-chat-btn").addEventListener("click", clearChat);

    // Mobile sidebar: menu button opens it, backdrop click closes it.
    const sidebar = document.getElementById("sidebar");
    const backdrop = document.getElementById("sidebar-backdrop");
    document.getElementById("menu-toggle")?.addEventListener("click", () => {
        sidebar.classList.toggle("-translate-x-full");
        backdrop.classList.toggle("hidden");
    });
    backdrop?.addEventListener("click", () => {
        sidebar.classList.add("-translate-x-full");
        backdrop.classList.add("hidden");
    });
}

function autoResizeTextarea(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
}
