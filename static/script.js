let currentConversationId = null;

document.addEventListener('DOMContentLoaded', () => {
    loadConversations();
    fetchModels();

    // Check if there's a last conversation? For now, start fresh or load first.
    // Actually, createNewChat() is better default.
});


async function fetchModels(type = 'text') {
    try {
        const response = await fetch(`/api/models?type=${type}`);
        const models = await response.json();
        const select = document.getElementById('model-select');
        select.innerHTML = '';

        if (models.length === 0) {
            const option = document.createElement('option');
            option.value = "";
            option.innerText = "No models found";
            select.appendChild(option);
        } else {
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.innerText = model;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error fetching models:', error);
    }
}

function onModeChange() {
    const mode = document.getElementById('mode-select').value;
    fetchModels(mode);
}

async function loadConversations() {
    const response = await fetch('/api/conversations');
    const conversations = await response.json();
    const list = document.getElementById('history-list');
    list.innerHTML = '';

    conversations.forEach(conv => {
        const div = document.createElement('div');
        div.className = 'group flex justify-between items-center p-2 hover:bg-gray-700 rounded cursor-pointer text-sm text-gray-300';

        const titleSpan = document.createElement('span');
        titleSpan.className = 'truncate flex-1';
        titleSpan.innerText = conv.title || 'Untitled Chat';
        titleSpan.onclick = () => loadChat(conv.id);

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'ml-2 text-gray-500 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity px-1';
        deleteBtn.innerText = '×';
        deleteBtn.title = 'Delete Chat';
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            if (confirm('Delete this chat?')) {
                deleteConversation(conv.id);
            }
        };

        div.appendChild(titleSpan);
        div.appendChild(deleteBtn);
        list.appendChild(div);
    });
}

async function deleteConversation(id) {
    try {
        const response = await fetch(`/api/conversations/${id}`, { method: 'DELETE' });
        if (response.ok) {
            if (currentConversationId === id) {
                createNewChat();
            } else {
                loadConversations();
            }
        } else {
            console.error('Failed to delete chat');
        }
    } catch (e) {
        console.error(e);
    }
}

async function createNewChat() {
    const response = await fetch('/api/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'New Chat' })
    });
    const data = await response.json();
    currentConversationId = data.id;
    document.getElementById('chat-container').innerHTML = ''; // Clear chat
    loadConversations(); // Refresh list
    appendMessage('system', 'Started a new conversation.');
}

async function loadChat(id) {
    currentConversationId = id;
    const response = await fetch(`/api/conversations/${id}`);
    const data = await response.json();

    const container = document.getElementById('chat-container');
    container.innerHTML = '';

    // Restore mode and model dropdowns
    const modeSelect = document.getElementById('mode-select');
    const savedMode = data.last_mode || 'text';
    modeSelect.value = savedMode;

    // Fetch models for the saved mode, then set the saved model
    await fetchModels(savedMode);
    const modelSelect = document.getElementById('model-select');
    if (data.last_model) {
        modelSelect.value = data.last_model;
    }

    const messages = data.messages || data;
    messages.forEach(msg => {
        appendMessage(msg.role, msg.content, msg.type);
    });
}

function appendMessage(role, content, type = 'text') {
    const container = document.getElementById('chat-container');
    const div = document.createElement('div');
    div.className = `flex ${role === 'user' ? 'justify-end' : 'justify-start'}`;

    const bubble = document.createElement('div');
    bubble.className = `max-w-3xl rounded-lg p-4 prose ${role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-200'}`;

    if (type === 'image') {
        const img = document.createElement('img');
        img.src = content;
        img.className = 'rounded-lg max-w-full h-auto';
        bubble.appendChild(img);
    } else {
        // Render Markdown
        if (role === 'system') {
            bubble.classList.add('italic', 'text-sm', 'bg-transparent', 'text-gray-500', 'p-2');
            bubble.innerText = content;
        } else {
            bubble.innerHTML = marked.parse(content);

            // Syntax Highlighting
            bubble.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });

            // Add Copy Button
            bubble.querySelectorAll('pre').forEach((pre) => {
                // Create a wrapper for the code block
                const wrapper = document.createElement('div');
                wrapper.className = 'relative group'; // 'group' for hover effects if needed

                // Insert wrapper before pre, then move pre into wrapper
                pre.parentNode.insertBefore(wrapper, pre);
                wrapper.appendChild(pre);

                const button = document.createElement('button');
                button.className = 'copy-btn';
                button.innerText = 'Copy';
                button.addEventListener('click', () => {
                    const code = pre.querySelector('code').innerText;
                    navigator.clipboard.writeText(code).then(() => {
                        button.innerText = 'Copied!';
                        setTimeout(() => {
                            button.innerText = 'Copy';
                        }, 2000);
                    }).catch(err => {
                        console.error('Failed to copy: ', err);
                        button.innerText = 'Error';
                    });
                });
                wrapper.appendChild(button);
            });
        }
    }

    div.appendChild(bubble);
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

async function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    if (!message) return;

    if (!currentConversationId) {
        await createNewChat();
    }

    // Clear input
    input.value = '';

    // Append user message immediately
    appendMessage('user', message);

    // Show typing indicator
    const container = document.getElementById('chat-container');
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typing-indicator';
    typingDiv.className = 'flex justify-start';
    typingDiv.innerHTML = '<div class="bg-gray-700 text-gray-200 rounded-lg p-4 typing-dots">Thinking</div>';
    container.appendChild(typingDiv);
    container.scrollTop = container.scrollHeight;

    const modelSelect = document.getElementById('model-select');
    const selectedModel = modelSelect.value || "default";
    const modeSelect = document.getElementById('mode-select');
    const selectedMode = modeSelect.value || "text";

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                conversation_id: currentConversationId,
                message: message,
                model: selectedModel,
                mode: selectedMode
            })
        });

        const data = await response.json();

        // Remove typing indicator
        document.getElementById('typing-indicator').remove();

        if (data.role) {
            appendMessage(data.role, data.content, data.type);

            // Auto-update title if provided
            if (data.new_title) {
                // Refresh list to show new title
                loadConversations();
            }
        } else {
            appendMessage('system', 'Error: Invalid response from server');
        }

    } catch (error) {
        document.getElementById('typing-indicator').remove();
        appendMessage('system', 'Error: Could not connect to server.');
        console.error(error);
    }
}

// Handle Enter key to send
document.getElementById('user-input').addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
