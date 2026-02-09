// DataExtract RAG - Main Application JavaScript

const API_BASE = '';

// State
let activities = [];
let queryCount = 0;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initUpload();
    initQuery();
    initDashboard();
    refreshDashboard();

    // Auto-refresh dashboard every 10 seconds
    setInterval(refreshDashboard, 10000);
});

// Navigation
function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-links li');

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            const page = link.dataset.page;

            // Update active nav
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            // Show page
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.getElementById(`${page}-page`).classList.add('active');

            // Refresh dashboard when switching to it
            if (page === 'dashboard') {
                refreshDashboard();
            }
        });
    });
}

// Upload functionality
function initUpload() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');

    // Click to upload
    dropzone.addEventListener('click', () => fileInput.click());

    // Drag and drop
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    // File input change
    fileInput.addEventListener('change', () => {
        handleFiles(fileInput.files);
    });
}

async function handleFiles(files) {
    const progressContainer = document.getElementById('upload-progress');
    const resultsContainer = document.getElementById('upload-results');

    for (const file of files) {
        // Show progress
        progressContainer.style.display = 'block';
        document.getElementById('upload-filename').textContent = `Uploading ${file.name}...`;
        document.getElementById('upload-status').textContent = 'Processing';
        document.getElementById('progress-fill').style.width = '30%';
        document.getElementById('progress-details').textContent = 'Extracting text...';

        try {
            const formData = new FormData();
            formData.append('file', file);

            // Update progress
            document.getElementById('progress-fill').style.width = '50%';
            document.getElementById('progress-details').textContent = 'Generating embeddings...';

            const response = await fetch(`${API_BASE}/upload/`, {
                method: 'POST',
                body: formData
            });

            document.getElementById('progress-fill').style.width = '100%';

            const result = await response.json();

            if (response.ok) {
                document.getElementById('upload-status').textContent = 'Complete';
                document.getElementById('progress-details').textContent =
                    `Created ${result.chunks_created} chunks`;

                // Add result card
                addResultCard(resultsContainer, {
                    filename: result.file_name,
                    type: result.file_type,
                    chunks: result.chunks_created,
                    status: result.status,
                    preview: result.text_preview
                });

                addActivity(`Uploaded: ${result.file_name} (${result.chunks_created} chunks)`);
            } else {
                document.getElementById('upload-status').textContent = 'Error';
                document.getElementById('progress-details').textContent =
                    result.detail || 'Upload failed';

                addResultCard(resultsContainer, {
                    filename: file.name,
                    status: 'error',
                    error: result.detail
                }, true);

                addActivity(`Upload failed: ${file.name}`);
            }

        } catch (error) {
            document.getElementById('upload-status').textContent = 'Error';
            document.getElementById('progress-details').textContent = error.message;

            addResultCard(resultsContainer, {
                filename: file.name,
                status: 'error',
                error: error.message
            }, true);
        }

        // Hide progress after delay
        setTimeout(() => {
            progressContainer.style.display = 'none';
            document.getElementById('progress-fill').style.width = '0%';
        }, 2000);
    }

    // Refresh dashboard
    refreshDashboard();
}

function addResultCard(container, data, isError = false) {
    const card = document.createElement('div');
    card.className = `result-card ${isError ? 'error' : ''}`;

    card.innerHTML = `
        <div class="result-header">
            <span class="result-filename">${data.filename}</span>
            <span class="result-status">${isError ? 'Failed' : 'Success'}</span>
        </div>
        <div class="result-details">
            ${isError
            ? `Error: ${data.error}`
            : `Type: ${data.type} | Chunks: ${data.chunks}`}
        </div>
    `;

    container.insertBefore(card, container.firstChild);
}

// Query functionality
function initQuery() {
    const input = document.getElementById('query-input');
    const sendBtn = document.getElementById('send-query');

    // Auto-resize textarea
    input.addEventListener('input', () => {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';
    });

    // Send on Enter (without Shift)
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });

    sendBtn.addEventListener('click', sendQuery);
}

async function sendQuery() {
    const input = document.getElementById('query-input');
    const messagesContainer = document.getElementById('chat-messages');
    const question = input.value.trim();

    if (!question) return;

    // Clear input
    input.value = '';
    input.style.height = 'auto';

    // Remove welcome message if present
    const welcome = messagesContainer.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    // Add user message
    addMessage(messagesContainer, question, 'user');

    // Add loading message
    const loadingId = Date.now();
    addMessage(messagesContainer, 'Thinking...', 'assistant', loadingId);

    try {
        const response = await fetch(`${API_BASE}/query/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, top_k: 5 })
        });

        const result = await response.json();

        // Remove loading message
        document.getElementById(`msg-${loadingId}`)?.remove();

        if (response.ok) {
            // Add confidence badge to answer
            const confidenceInfo = {
                score: result.confidence_score,
                source: result.source,
                offerInternet: result.offer_internet
            };

            addMessage(messagesContainer, result.answer, 'assistant', null, result.sources, confidenceInfo, question);
            queryCount++;
            addActivity(`Query: "${question.substring(0, 50)}..." (${result.confidence_score}% confidence)`);
        } else {
            addMessage(messagesContainer, `Error: ${result.detail}`, 'assistant');
        }

    } catch (error) {
        document.getElementById(`msg-${loadingId}`)?.remove();
        addMessage(messagesContainer, `Error: ${error.message}`, 'assistant');
    }

    // Update stats
    document.getElementById('stat-queries').textContent = queryCount;
}

// Internet search when confidence is low
async function searchInternet(question) {
    const messagesContainer = document.getElementById('chat-messages');

    // Add loading message
    const loadingId = Date.now();
    addMessage(messagesContainer, '🌐 Searching with general knowledge...', 'assistant', loadingId);

    try {
        const response = await fetch(`${API_BASE}/query/internet`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, save_if_confident: true })
        });

        const result = await response.json();

        // Remove loading message
        document.getElementById(`msg-${loadingId}`)?.remove();

        if (response.ok) {
            const confidenceInfo = {
                score: result.confidence_score,
                source: 'internet',
                savedToDb: result.saved_to_db
            };

            let answer = result.answer;
            if (result.saved_to_db) {
                answer += '\n\n✅ *This answer has been saved for future reference.*';
            }

            addMessage(messagesContainer, answer, 'assistant', null, null, confidenceInfo, null);
            addActivity(`Internet search: "${question.substring(0, 40)}..." (${result.confidence_score}%${result.saved_to_db ? ', saved' : ''})`);
        } else {
            addMessage(messagesContainer, `Error: ${result.detail}`, 'assistant');
        }

    } catch (error) {
        document.getElementById(`msg-${loadingId}`)?.remove();
        addMessage(messagesContainer, `Error: ${error.message}`, 'assistant');
    }
}

function addMessage(container, text, type, id = null, sources = null, confidenceInfo = null, originalQuestion = null) {
    const div = document.createElement('div');
    div.className = `message message-${type}`;
    if (id) div.id = `msg-${id}`;

    if (type === 'user') {
        div.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
    } else {
        let sourcesHtml = '';
        if (sources && sources.length > 0) {
            const sourcesList = sources.map(s => s.file || s.type).filter((v, i, a) => a.indexOf(v) === i);
            sourcesHtml = `
                <div class="message-sources">
                    📄 Sources: ${sourcesList.join(', ')}
                </div>
            `;
        }

        // Confidence badge
        let confidenceHtml = '';
        if (confidenceInfo && confidenceInfo.score !== undefined) {
            const score = confidenceInfo.score;
            let badgeClass = 'confidence-high';
            let badgeIcon = '✓';

            if (score < 60) {
                badgeClass = 'confidence-low';
                badgeIcon = '⚠️';
            } else if (score < 80) {
                badgeClass = 'confidence-medium';
                badgeIcon = '~';
            }

            const sourceLabel = confidenceInfo.source === 'learned' ? '📚 Learned' :
                confidenceInfo.source === 'internet' ? '🌐 Internet' : '📄 Local DB';

            confidenceHtml = `
                <div class="confidence-badge ${badgeClass}">
                    <span class="confidence-score">${badgeIcon} ${score}% Confidence</span>
                    <span class="confidence-source">${sourceLabel}</span>
                </div>
            `;
        }

        // Internet search button (when confidence is low)
        let internetBtnHtml = '';
        if (confidenceInfo && confidenceInfo.offerInternet && originalQuestion) {
            const escapedQuestion = originalQuestion.replace(/'/g, "\\'").replace(/"/g, '\\"');
            internetBtnHtml = `
                <div class="internet-search-offer">
                    <p>💡 Low confidence answer. Want me to search using general knowledge?</p>
                    <button class="internet-search-btn" onclick="searchInternet('${escapedQuestion}')">
                        🌐 Search Internet
                    </button>
                </div>
            `;
        }

        // Saved indicator
        let savedHtml = '';
        if (confidenceInfo && confidenceInfo.savedToDb) {
            savedHtml = `<div class="saved-indicator">✅ Saved for future reference</div>`;
        }

        // Parse markdown for assistant messages
        const formattedText = parseMarkdown(text);

        div.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                ${confidenceHtml}
                ${formattedText}
                ${sourcesHtml}
                ${savedHtml}
                ${internetBtnHtml}
            </div>
        `;
    }

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

// Simple markdown parser for code blocks, bold, lists, etc.
function parseMarkdown(text) {
    if (!text) return '';

    let html = escapeHtml(text);

    // Code blocks with language (```python ... ```)
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        const language = lang || 'code';
        return `<pre class="code-block"><code class="language-${language}">${code.trim()}</code></pre>`;
    });

    // Inline code (`code`)
    html = html.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

    // Bold (**text**)
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Italic (*text*)
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Headers (## Header)
    html = html.replace(/^### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^# (.+)$/gm, '<h2>$1</h2>');

    // Numbered lists (1. item)
    html = html.replace(/^\d+\. (.+)$/gm, '<li class="numbered">$1</li>');

    // Bullet points (- item or * item)
    html = html.replace(/^[-*] (.+)$/gm, '<li class="bullet">$1</li>');

    // Line breaks
    html = html.replace(/\n/g, '<br>');

    return html;
}

// Dashboard functionality
function initDashboard() {
    // Initial load handled by refreshDashboard
}

async function refreshDashboard() {
    // Check system health
    await checkHealth();

    // Get stats
    await getStats();
}

async function checkHealth() {
    // MongoDB
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            updateHealthCard('mongodb-status', true, 'Connected');
        } else {
            updateHealthCard('mongodb-status', false, 'Disconnected');
        }
    } catch {
        updateHealthCard('mongodb-status', false, 'Unreachable');
    }

    // Ollama
    try {
        const response = await fetch('http://localhost:11434/api/tags');
        if (response.ok) {
            const data = await response.json();
            const models = data.models?.map(m => m.name).join(', ') || 'No models';
            updateHealthCard('ollama-status', true, 'Running');
        } else {
            updateHealthCard('ollama-status', false, 'Not running');
        }
    } catch {
        updateHealthCard('ollama-status', false, 'Offline');
    }

    // Stats also show FAISS and embedding health
    try {
        const response = await fetch(`${API_BASE}/stats`);
        if (response.ok) {
            const data = await response.json();
            if (data.vector_store) {
                updateHealthCard('faiss-status', true,
                    `${data.vector_store.total_vectors} vectors`);
                updateHealthCard('embedding-status', true,
                    `Dim: ${data.vector_store.embedding_dim}`);

                // Update stats
                document.getElementById('stat-vectors').textContent =
                    data.vector_store.total_vectors || 0;
                document.getElementById('stat-embedding-dim').textContent =
                    data.vector_store.embedding_dim || '--';
            }
        }
    } catch {
        updateHealthCard('faiss-status', false, 'Error');
        updateHealthCard('embedding-status', false, 'Error');
    }

    // Update system status indicator
    const statusDot = document.querySelector('.status-dot');
    statusDot.classList.add('online');
}

function updateHealthCard(id, healthy, status) {
    const card = document.getElementById(id);
    if (!card) return;

    card.classList.remove('healthy', 'error', 'loading');
    card.classList.add(healthy ? 'healthy' : 'error');

    const statusEl = card.querySelector('.health-status');
    if (statusEl) {
        statusEl.textContent = status;
        statusEl.classList.remove('healthy', 'error', 'loading');
        statusEl.classList.add(healthy ? 'healthy' : 'error');
    }
}

async function getStats() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        if (response.ok) {
            const data = await response.json();

            if (data.vector_store) {
                document.getElementById('stat-vectors').textContent =
                    data.vector_store.total_vectors || 0;
                document.getElementById('stat-documents').textContent =
                    data.vector_store.total_chunks || 'N/A';
            }
        }
    } catch (error) {
        console.error('Failed to get stats:', error);
    }

    // Update query count from local state
    document.getElementById('stat-queries').textContent = queryCount;
}

// Activity log
function addActivity(message) {
    const now = new Date();
    const time = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });

    activities.unshift({ time, message });
    activities = activities.slice(0, 50); // Keep last 50

    updateActivityLog();
}

function updateActivityLog() {
    const log = document.getElementById('activity-log');

    log.innerHTML = activities.map(a => `
        <div class="activity-item">
            <span class="activity-time">${a.time}</span>
            <span class="activity-message">${escapeHtml(a.message)}</span>
        </div>
    `).join('');
}

// Utility
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add initial activity
addActivity('System initialized');
