import { ChatOpenAI } from 'langchain/chat_models/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_TEMPLATE = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_TEMPLATE = `You are a helpful AI assistant that is an expert at Government Regulations. Use the following pieces of context to answer the question at the end with references to the document's part, section, and page like this [§ 173.185(b)(3)]. Provide 3 strong answerable follow up questions.
If you don't know the answer, attempt to answer the question. Then attempt to provide a list of 5 questions that may be more applicable.
If the question is not related to the context, provide a list of 5 questions that may be better worded.

{context}

Question: {question}
Helpful answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new ChatOpenAI({
    temperature: 0.2, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo-0613', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_TEMPLATE,
      questionGeneratorTemplate: CONDENSE_TEMPLATE,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
