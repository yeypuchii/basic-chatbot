const express = require('express');
const { ChatOpenAI } = require('@langchain/openai');
const { StateGraph, END, START } = require('@langchain/langgraph');
const { HumanMessage, AIMessage } = require('@langchain/core/messages');
const { MemorySaver } = require('@langchain/langgraph');
require('dotenv').config();

// Express setup
const app = express();
app.use(express.json());
app.use(express.static('public'));

// For debugging - store complete conversations
const conversationStore = {};

// Initialize the LLM
const llm = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo',
  temperature: 0.7
});

// Define the chat node that preserves history
const chatNode = async (state) => {
  console.log('ChatNode received state:', JSON.stringify(state, null, 2));
  
  // Convert messages to LangChain format
  const formattedMessages = state.messages.map(msg => 
    msg.role === 'user' 
      ? new HumanMessage(msg.content) 
      : new AIMessage(msg.content)
  );
  
  // Call the language model
  const response = await llm.invoke(formattedMessages);
  console.log('LLM response:', response.content);
  
  // Return the complete conversation history with the new message
  return {
    messages: [
      ...state.messages,
      { role: 'assistant', content: response.content }
    ]
  };
};

// Initialize memory saver
const memorySaver = new MemorySaver();

// Create and compile the graph
function createGraph() {
  const builder = new StateGraph({
    channels: {
      messages: {
        value: [],
        // This reducer is crucial - it ensures we use the complete new state
        reducer: (_, newValue) => newValue 
      }
    }
  });
  
  // Add nodes and edges
  builder.addNode("chat", chatNode);
  builder.addEdge(START, "chat");
  builder.addEdge("chat", END);
  
  // Compile with memorySaver
  return builder.compile({
    checkpointer: memorySaver
  });
}

// Create the graph
const graph = createGraph();

// Debug memory endpoint
app.get('/debug-memory', (req, res) => {
  res.json({
    conversations: conversationStore
  });
});

// Clear memory endpoint
app.post('/clear-memory', async (req, res) => {
  try {
    const { sessionId = 'default' } = req.body;
    
    // Clear from our debug store
    delete conversationStore[sessionId];
    
    // Try to clear from the memory saver (might not work)
    try {
      await memorySaver.delete(sessionId);
    } catch (error) {
      console.log('Error clearing from memorySaver:', error);
    }
    
    res.json({ status: 'Memory cleared' });
  } catch (error) {
    res.status(500).json({ error: 'Error clearing memory' });
  }
});

// Main chat endpoint
app.post('/chat', async (req, res) => {
  try {
    const { message, sessionId = 'default' } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }
    
    console.log(`Processing message for session ${sessionId}: "${message}"`);
    
    // Check our debug store first for complete history
    let initialState;
    if (conversationStore[sessionId]) {
      initialState = conversationStore[sessionId];
      console.log('Found conversation in store:', initialState.messages.length, 'messages');
    } else {
      // Try to get from memorySaver
      try {
        initialState = await memorySaver.get(sessionId);
        console.log('Retrieved from memorySaver:', initialState);
      } catch (error) {
        console.log('No existing conversation found, starting new one');
        initialState = { messages: [] };
      }
    }
    
    // Add user message to the conversation
    const inputState = {
      messages: [
        ...(initialState.messages || []),
        { role: 'user', content: message }
      ]
    };
    
    console.log('Input state to graph:', JSON.stringify(inputState, null, 2));
    
    // Process through the graph
    const result = await graph.invoke(inputState, {
      configurable: {
        thread_id: sessionId
      }
    });
    
    console.log('Result from graph:', JSON.stringify(result, null, 2));
    
    // Store complete conversation in our debug store
    conversationStore[sessionId] = result;
    
    // Try to update the memory saver (might not be needed)
    try {
      await memorySaver.put(sessionId, result);
    } catch (error) {
      console.log('Error updating memorySaver:', error);
    }
    
    // Return the result
    res.json({
      response: result.messages[result.messages.length - 1].content,
      history: result.messages,
      messageCount: result.messages.length,
      sessionId
    });
  } catch (error) {
    console.error('Error handling chat:', error);
    res.status(500).json({ 
      error: 'Error processing your message',
      details: error.message
    });
  }
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});