import { GoogleGenerativeAI } from "@google/generative-ai";

export interface ImageAttachment {
  inlineData: {
    data: string;
    mimeType: string;
  };
}

export const sendMessageToGemini = async (
  message: string,
  images: ImageAttachment[],
  history: { role: string; parts: { text: string }[] }[],
  config: {
    apiKeys: string[];
    systemInstruction: string;
  }
): Promise<string> => {
  
  const tryGenerate = async (retryIdx: number): Promise<string> => {
    if (retryIdx >= config.apiKeys.length) {
      throw new Error("All API keys exhausted. Please update keys in Admin Dashboard.");
    }

    try {
      const apiKey = config.apiKeys[retryIdx];
      const ai = new GoogleGenerativeAI(apiKey);

      // Initialize model with system instruction
      const model = ai.getGenerativeModel({ 
        model: 'gemini-2.0-flash-exp',
        systemInstruction: config.systemInstruction
      });

      // Format history for chat - convert 'model' role to 'model' (keep as is)
      const formattedHistory = history.map(msg => ({
        role: msg.role === 'model' ? 'model' : 'user',
        parts: msg.parts.map(p => ({ text: p.text }))
      }));

      // Build current message parts
      const currentParts: any[] = [];
      
      // Add text if exists
      if (message && message.trim()) {
        currentParts.push({ text: message });
      }

      // Add images if exist
      if (images && images.length > 0) {
        images.forEach(img => {
          currentParts.push({
            inlineData: {
              mimeType: img.inlineData.mimeType,
              data: img.inlineData.data
            }
          });
        });
      }

      // Validate message
      if (currentParts.length === 0) {
        throw new Error("Message cannot be empty");
      }

      // Start chat session with history
      const chat = model.startChat({
        history: formattedHistory,
        generationConfig: {
          temperature: 1.3,
          topK: 40,
          topP: 0.95,
          maxOutputTokens: 8192,
        }
      });

      // Send current message with all parts (text + images)
      const result = await chat.sendMessage(currentParts);
      const response = result.response;
      const text = response.text();

      if (!text || text.trim() === '') {
        throw new Error("Empty response from AI");
      }
      
      return text;

    } catch (error: any) {
      const errorMsg = error.message || error.toString();
      console.warn(`API Key index ${retryIdx} failed:`, errorMsg);
      
      // Retry with next key if quota/permission error
      const shouldRetry = 
        errorMsg.includes("429") || 
        errorMsg.includes("403") || 
        errorMsg.includes("RESOURCE_EXHAUSTED") ||
        errorMsg.includes("quota") ||
        errorMsg.includes("API_KEY_INVALID") ||
        errorMsg.includes("PERMISSION_DENIED");
      
      if (shouldRetry && retryIdx + 1 < config.apiKeys.length) {
        console.log(`Switching to next API key (${retryIdx + 1})...`);
        return tryGenerate(retryIdx + 1);
      }
      
      // If it's the last key or non-retryable error, throw
      throw new Error(`AI Connection Error: ${errorMsg}`);
    }
  };

  return tryGenerate(0);
};
