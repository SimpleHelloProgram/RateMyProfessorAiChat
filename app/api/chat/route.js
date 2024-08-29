import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = `
You are an AI assistant designed to help students find the best professors according to their specific queries. Your goal is to provide students with up to the top 3 professors that match their criteria, ranked by relevance. If fewer than 3 professors match the query, you should return only the matching professors. If no professors match the description, inform the user politely.

When responding, consider factors such as teaching style, difficulty, student feedback, and overall ratings. Always aim to give concise, helpful, and accurate information based on the student’s request.

Example queries include:
- "Who are the best Computer Science professors at UC Berkeley?"
- "Can you recommend a professor who is known for being approachable and supportive in the Psychology department?"
- "I need a professor who has great ratings for Math 101 and isn't too difficult."

In these cases:
1. If there are 3 or more professors that match the criteria, list the top 3 with brief explanations.
2. If there are 1-2 professors, provide only those with the relevant explanations.
3. If no professors match, explain that there are no exact matches but offer advice or suggest trying different criteria.

Always be polite, informative, and focused on helping students make informed decisions.
`;

export async function POST(req) {
    try {
        const data = await req.json();

        const pc = new Pinecone({
            apiKey: process.env.PINECONE_API_KEY,
        });
        const index = pc.index('rag').namespace('ns1');
        const openai = new OpenAI();

        const text = data[data.length - 1].content;
        const embedding = await openai.embeddings.create({
            model: 'text-embedding-ada-002',
            input: text,
        });

        const result = await index.query({
            topK: 3,
            includeMetadata: true,
            vector: embedding.data[0].embedding,
        });

        let resultString = 'Returned Result from vector db (done automatically):';
        result.matches.forEach((match) => {
            resultString += `
            Professor: ${match.id}
            Review: ${match.metadata.review}
            Subject: ${match.metadata.subject}
            Star: ${match.metadata.star}
            \n\n
            `;
        });

        const lastMessage = data[data.length - 1];
        const lastMessageContent = lastMessage.content + resultString;
        const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

        const completion = await openai.chat.completions.create({
            messages: [
                { role: 'system', content: systemPrompt },
                ...lastDataWithoutLastMessage,
                { role: 'user', content: lastMessageContent },
            ],
            model: 'gpt-4', // Replace with the actual model name you're using
            stream: true,
        });

        const stream = new ReadableStream({
            async start(controller) {
                const encoder = new TextEncoder();
                try {
                    for await (const chunk of completion) {
                        const content = chunk.choices[0]?.delta?.content;
                        if (content) {
                            const text = encoder.encode(content);
                            controller.enqueue(text);
                        }
                    }
                } catch (err) {
                    controller.error(err);
                } finally {
                    controller.close();
                }
            },
        });

        return new NextResponse(stream);
    } catch (error) {
        console.error("Error handling request:", error);
        return new NextResponse("Internal Server Error", { status: 500 });
    }
}
