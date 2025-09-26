// Global variables
const API_BASE_URL = 'http://localhost:8000';
let currentCustomerId = null;

// DOM elements
const customerSetup = document.getElementById('customerSetup');
const chatMessages = document.getElementById('chatMessages');
const customerIdInput = document.getElementById('customerIdInput');
const startChatButton = document.getElementById('startChatButton');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const typingIndicator = document.getElementById('typingIndicator');

// Initialize the application
document.addEventListener('DOMContentLoaded', async () => {
    await checkApiStatus();
    setupEventListeners();
});

// Check API status
async function checkApiStatus() {
    console.log('🔍 Checking API status at:', API_BASE_URL);
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        console.log('📡 API Response status:', response.status);
        if (response.ok) {
            const data = await response.json();
            console.log('✅ API Response data:', data);
            updateStatus('online', 'API Connected');
        } else {
            console.log('❌ API Error - Status:', response.status);
            updateStatus('offline', 'API Error');
        }
    } catch (error) {
        console.error('❌ API Status Error:', error);
        updateStatus('offline', 'API Offline');
    }
}

// Update status indicator
function updateStatus(status, text) {
    statusText.textContent = text;
    statusDot.className = `status-dot ${status}`;
}

// Setup event listeners
function setupEventListeners() {
    // Customer ID setup
    startChatButton.addEventListener('click', handleStartChat);
    customerIdInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleStartChat();
        }
    });
    customerIdInput.addEventListener('input', handleCustomerIdInput);

    // Chat functionality
    sendButton.addEventListener('click', handleSendMessage);
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });
    messageInput.addEventListener('input', handleInputChange);
}

// Handle customer ID input
function handleCustomerIdInput() {
    const hasText = customerIdInput.value.trim().length > 0;
    startChatButton.disabled = !hasText;
}

// Handle start chat
async function handleStartChat() {
    const customerId = customerIdInput.value.trim();
    console.log('🆔 Starting chat with customer ID:', customerId);
    
    if (!customerId) {
        alert('Please enter a Customer ID');
        return;
    }

    // Validate customer ID format (optional)
    if (!customerId.startsWith('CUST')) {
        alert('Customer ID should start with "CUST" (e.g., CUST001)');
        return;
    }

    try {
        // Check if customer exists
        const response = await fetch(`${API_BASE_URL}/customers`);
        if (response.ok) {
            const data = await response.json();
            if (!data.customers.includes(customerId)) {
                alert(`Customer ID "${customerId}" not found. Try: ${data.customers.slice(0, 5).join(', ')}`);
                return;
            }
        }

        // Set current customer and switch to chat interface
        currentCustomerId = customerId;
        switchToChatInterface();
        
    } catch (error) {
        console.error('❌ Error validating customer:', error);
        alert('Error validating customer ID. Please try again.');
    }
}

// Switch from setup to chat interface
function switchToChatInterface() {
    console.log('🔄 Switching to chat interface for customer:', currentCustomerId);
    customerSetup.style.display = 'none';
    chatMessages.style.display = 'flex';
    messageInput.focus();
    
    // Update status to show current customer
    updateStatus('online', `Connected as ${currentCustomerId}`);
}

// Handle input change
function handleInputChange() {
    const hasText = messageInput.value.trim().length > 0;
    sendButton.disabled = !hasText;
}

// Handle send message - FULL GRAPH WORKFLOW
async function handleSendMessage() {
    const message = messageInput.value.trim();
    console.log('📤 Sending message:', message);
    if (!message || !currentCustomerId) return;

    // Add user message to chat
    addUserMessage(message);
    messageInput.value = '';
    sendButton.disabled = true;
    
    // Show typing indicator
    showTypingIndicator();

    try {
        console.log('🔄 Making API call to:', `${API_BASE_URL}/chat`);
        console.log('📦 Request payload:', { customer_id: currentCustomerId, message: message });
        
        // Use the full chat endpoint that runs the complete graph
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                customer_id: currentCustomerId,
                message: message
            })
        });
        
        console.log('📡 Chat response status:', response.status);

        if (response.ok) {
            const data = await response.json();
            console.log('✅ Chat response data:', data);
            hideTypingIndicator();
            
            // Add bot response to chat
            if (data.success) {
                addBotMessage(data.response, data);
            } else {
                addBotMessage("I apologize, but I encountered an error while processing your request. Please try again.", { error: data.error });
            }
        } else {
            console.log('❌ Chat API Error - Status:', response.status);
            const errorData = await response.json().catch(() => ({}));
            console.log('❌ Error details:', errorData);
            hideTypingIndicator();
            addBotMessage("I'm sorry, but I'm having trouble processing your request right now. Please try again in a moment.");
        }
    } catch (error) {
        console.error('❌ Chat request error:', error);
        hideTypingIndicator();
        addBotMessage("I'm experiencing connection issues. Please check your internet connection and try again.");
    }
}

// Add user message to chat
function addUserMessage(message) {
    const messageElement = document.createElement('div');
    messageElement.className = 'message user-message';
    messageElement.innerHTML = `
        <div class="message-content">
            <p>${escapeHtml(message)}</p>
        </div>
        <div class="user-avatar">
            <i class="fas fa-user"></i>
        </div>
    `;
    chatMessages.appendChild(messageElement);
    scrollToBottom();
}

// Add bot message to chat with full context
function addBotMessage(message, context = {}) {
    const messageElement = document.createElement('div');
    messageElement.className = 'message bot-message';
    
    let botResponse = `
        <div class="bot-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <div class="message-text">${formatMessage(message)}</div>
    `;

    // Add escalation info if available
    if (context.is_escalated && context.specialist_info) {
        const specialist = context.specialist_info;
        botResponse += `
            <div class="escalation-info">
                <div class="escalation-header">🚨 Escalated to Specialist</div>
                <div class="specialist-details">
                    <p><strong>👤 Specialist:</strong> ${specialist.specialist}</p>
                    <p><strong>📧 Email:</strong> ${specialist.email}</p>
                    <p><strong>⏱️ Response Time:</strong> ${specialist.response_time}</p>
                    <p><strong>🎯 Expertise:</strong> ${specialist.expertise.join(', ')}</p>
                </div>
            </div>
        `;
    }

    // Add search quality info if available
    if (context.search_quality) {
        const quality = context.search_quality;
        botResponse += `
            <div class="search-info">
                <small>📊 Search Quality: KB ${(quality.knowledge_base_score * 100).toFixed(0)}% | Web ${(quality.web_search_score * 100).toFixed(0)}%</small>
            </div>
        `;
    }

    botResponse += `
        </div>
    `;

    messageElement.innerHTML = botResponse;
    chatMessages.appendChild(messageElement);
    scrollToBottom();
}

// Format message content (handle markdown-like formatting)
function formatMessage(message) {
    return message
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
}

// Show typing indicator
function showTypingIndicator() {
    typingIndicator.style.display = 'block';
}

// Hide typing indicator
function hideTypingIndicator() {
    typingIndicator.style.display = 'none';
}

// Scroll to bottom of chat
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add styles for new elements
const additionalStyles = `
    .escalation-info {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
    }

    .escalation-header {
        font-weight: 600;
        color: #856404;
        margin-bottom: 8px;
    }

    .specialist-details p {
        margin: 4px 0;
        color: #856404;
        font-size: 14px;
    }

    .search-info {
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid #e1e5e9;
        color: #666;
    }
`;

// Inject additional styles
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);

console.log('✅ Full chatbot script loaded');