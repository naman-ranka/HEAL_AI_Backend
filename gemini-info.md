
# Using Gemini with Genkit in this Next.js App

This document provides a guide on how the Gemini API is integrated into this application using the Genkit framework. You can use this as a reference to understand the existing code or to set up a similar environment in a different application.

## 1. Project Setup & Configuration

### Environment Variables

To connect to the Gemini API, you need an API key from Google AI Studio.

1.  Create a file named `.env` in the root of the project.
2.  Add your API key to the `.env` file:
    ```
    GEMINI_API_KEY=your_api_key_here
    ```

### Required Packages

The following packages are essential for the setup. They are already included in `package.json`:

-   `genkit`: The core Genkit framework.
-   `@genkit-ai/googleai`: The Genkit plugin for Google AI services, including Gemini.
-   `@genkit-ai/next`: The Next.js integration for Genkit.
-   `zod`: Used for defining structured input and output schemas for the AI models.

### Genkit Initialization

The core configuration for Genkit is located in `src/ai/genkit.ts`. This is where we initialize the framework and specify the default AI model.

**File: `src/ai/genkit.ts`**
```typescript
import {genkit} from 'genkit';
import {googleAI} from '@genkit-ai/googleai';

export const ai = genkit({
  plugins: [googleAI()],
  model: 'googleai/gemini-2.5-flash',
});
```

-   `plugins: [googleAI()]`: This line loads the Google AI plugin, which automatically uses the `GEMINI_API_KEY` from your `.env` file to authenticate.
-   `model: 'googleai/gemini-2.5-flash'`: This sets **Gemini 2.5 Flash** as the default model for all AI generation calls. This is the primary model used throughout the application for tasks like analysis and chat.

---

## 2. How API Calls Are Made: Genkit Flows

Instead of making direct API calls, we define reusable server-side functions called "Flows." A flow is an abstraction that handles the interaction with the AI model. This makes the code cleaner and easier to maintain.

### Key Concepts:

-   **Schema Definition (Zod):** We use `zod` to define the expected structure of the data we send to the AI (`inputSchema`) and the data we expect back (`outputSchema`). This ensures type safety and helps instruct the model to return structured JSON.
-   **Prompt Definition (`ai.definePrompt`):** This is where the actual prompt template is written. It uses Handlebars syntax (`{{{...}}}`) to insert dynamic data from the input schema.
-   **Flow Definition (`ai.defineFlow`):** This wraps the prompt and any other logic. It takes an input, calls the prompt with that input, and returns the structured output.
-   **Server Action Wrapper:** Each flow is wrapped in an exported `async` function (a Next.js Server Action) so it can be easily called from our client-side React components.

### Example: Medical Bill Analysis Flow

Here’s a breakdown of the flow in `src/ai/flows/medical-bill-analysis.ts`:

1.  **Input/Output Schemas:** We define what the flow accepts (a bill image and policy text) and what it returns (a structured report).

    ```typescript
    const AnalyzeMedicalBillInputSchema = z.object({
      billDataUri: z.string().describe("..."),
      policyData: z.string().describe("..."),
    });

    const AnalyzeMedicalBillOutputSchema = z.object({
      report: z.array(/* ... */),
    });
    ```

2.  **Prompt:** The prompt instructs the AI on its role and what to do with the inputs. The `{{media url=...}}` syntax is used to pass image data.

    ```typescript
    const prompt = ai.definePrompt({
      name: 'analyzeMedicalBillPrompt',
      input: {schema: AnalyzeMedicalBillInputSchema},
      output: {schema: AnalyzeMedicalBillOutputSchema},
      prompt: `You are a health expense analyst...meticulously analyze this image of a medical bill...Photo: {{media url=billDataUri}}`,
    });
    ```

3.  **Flow and Wrapper:** The flow executes the prompt, and the wrapper function makes it available to the rest of the app.

    ```typescript
    const analyzeMedicalBillFlow = ai.defineFlow({ /* ... */ }, async input => {
      const {output} = await prompt(input);
      return output!;
    });

    export async function analyzeMedicalBill(input: AnalyzeMedicalBillInput): Promise<AnalyzeMedicalBillOutput> {
      return analyzeMedicalBillFlow(input);
    }
    ```

---

## 3. How Gemini is Used Everywhere

This application uses Gemini for three core features, each defined in its own flow:

1.  **Insurance Policy Upload & Summary (`src/ai/flows/insurance-policy-upload-and-summary.ts`)**
    -   **Model:** `gemini-2.5-flash`
    -   **Functionality:** Takes an image or PDF of an insurance policy, extracts the full text, and summarizes key details (deductible, copay, etc.) into a structured JSON object.

2.  **Medical Bill Analysis (`src/ai/flows/medical-bill-analysis.ts`)**
    -   **Model:** `gemini-2.5-flash`
    -   **Functionality:** Compares an uploaded medical bill image against the extracted policy text. It identifies discrepancies and generates a line-by-line report.

3.  **Conversational Policy Chatbot (`src/ai/flows/conversational-policy-chatbot.ts`)**
    -   **Model:** `gemini-2.5-flash`
    -   **Functionality:** Powers the chat feature. It answers user questions based *only* on the context of the uploaded policy text, using a RAG (Retrieval-Augmented Generation) approach. It also maintains conversation history.
