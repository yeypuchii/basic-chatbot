const express = require('express');
const { ChatOpenAI } = require('@langchain/openai');
const { StateGraph, END, START } = require('@langchain/langgraph');
const { HumanMessage, AIMessage } = require('@langchain/core/messages');
const { MemorySaver } = require('@langchain/langgraph');
const { z } = require('zod');
require('dotenv').config();

// Express setup
const app = express();
app.use(express.json());
app.use(express.static('public'));

// Global memory store for debugging
let memoryStoreContent = {};

// Initialize the LLM
const llm = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo',
  temperature: 0.7
});

// Create state schema
const stateSchema = z.object({
  messages: z.array(
    z.object({
      role: z.enum(['user', 'assistant']),
      content: z.string()
    })
  )
});

// Define the chat node
const chatNode = async (state) => {
  console.log('ChatNode received state:', JSON.stringify(state, null, 2));
  
  // Convert state messages to LangChain format
  const formattedMessages = state.messages.map(msg => 
    msg.role === 'user' 
      ? new HumanMessage(msg.content) 
      : new AIMessage(msg.content)
  );
  
  // Call the language model
  const response = await llm.invoke(formattedMessages);
  console.log('LLM response now:', response.content);
  
  // Create the new state with added assistant message
  return {
    messages: [
      ...state.messages,
      { 
        role: 'assistant', 
        content: response.content 
      }
    ]
  };
};

// Initialize memory
const memorySaver = new MemorySaver();

// Create and compile the graph
function createGraph() {
  const builder = new StateGraph({
    channels: {
      messages: {
        value: [],
        reducer: (_, newValue) => newValue
      }
    }
  });
  
  // Add nodes
  builder.addNode("chat", chatNode);
  
  // Add edges
  builder.addEdge(START, "chat");
  builder.addEdge("chat", END);
  
  // Compile with memorySaver
  return builder.compile({
    checkpointer: memorySaver
  });
}

// Create the graph
const graph = createGraph();

// Memory debugging endpoint
app.get('/debug-memory', async (req, res) => {
  try {
    res.json({
      memory: memoryStoreContent,
      note: "This is a debug view of the memory"
    });
  } catch (error) {
    res.status(500).json({ error: "Error accessing memory debug info" });
  }
});

// Clear memory
app.post('/clear-memory', async (req, res) => {
  try {
    const { sessionId = 'default' } = req.body;
    
    // Reset the session in our tracking object
    memoryStoreContent[sessionId] = undefined;
    
    res.json({ 
      status: 'Memory cleared for session: ' + sessionId,
      note: "The next message will start a new conversation"
    });
  } catch (error) {
    res.status(500).json({ error: 'Error clearing memory' });
  }
});

// Chat endpoint
app.post('/chat', async (req, res) => {
  try {
    const { message, sessionId = 'default' } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }
    
    console.log(`Processing message for session ${sessionId}: "${message}"`);
    
    // Create config with thread_id
    const config = {
      configurable: {
        thread_id: sessionId
      }
    };
    
    // Get existing state or create new one
    let existingState;
    try {
      existingState = await memorySaver.get(sessionId);
      console.log('Loaded existing state:', JSON.stringify(existingState, null, 2));
    } catch (error) {
      console.log('No existing state found, creating new conversation');
      existingState = { messages: [] };
    }
    
    // Create input state with new message
    const inputState = {
      messages: [
        ...(existingState?.messages || []),
        { role: 'user', content: message }
      ]
    };
    
    console.log('Sending to graph:', JSON.stringify(inputState, null, 2));
    
    // Process the message
    const result = await graph.invoke(inputState, config);
    
    console.log('Graph result:', JSON.stringify(result, null, 2));
    
    // Store for tracking/debugging
    memoryStoreContent[sessionId] = result;
    
    // Return the result
    res.json({
      response: result.messages[result.messages.length - 1].content,
      history: result.messages,
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