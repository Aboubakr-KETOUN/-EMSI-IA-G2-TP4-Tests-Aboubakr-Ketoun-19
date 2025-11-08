package ma.emsi.ketoun.test4;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.store.embedding.EmbeddingStore;
import ma.emsi.ketoun.Interfaces.AssistantPasRAG;
import ma.emsi.ketoun.test3.TestRoutage;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class TestPasRag {

    public static void main(String[] args) {

        //retrait de la cle du env local
        String apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null) {
            System.err.println("No Key!");
            return;
        }

        // Creation du Model avec le builder
        GoogleAiGeminiChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .logRequestsAndResponses(true)
                .temperature(0.7)
                .build();


        //creation embeding model et store
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        EmbeddingStore<TextSegment> embeddingStore1 = TestRoutage.creerEmbeddingStore("/rag.pdf", embeddingModel);


        //content retriever
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore1)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        //classe interne utilise prompt template
        class QueryRouterPourEviterRag implements QueryRouter {

            private final PromptTemplate promptTemplate = PromptTemplate.from(
                    "Est-ce que la requête '{{requete}}' porte sur l'IA ? " +
                            "Réponds seulement par 'oui', 'non' ou 'peut-être'."
            );

            @Override
            public List<ContentRetriever> route(Query query) {

                String prompt = promptTemplate.apply(Map.of("requete", query.text())).text();
                String response = model.chat(prompt);

                if (response.toLowerCase().contains("non")) {
                    return Collections.emptyList();
                } else {
                    return Collections.singletonList(contentRetriever);
                }
            }
        }


        //creation du queryrouter
        QueryRouter router = new QueryRouterPourEviterRag();

        //creation du retrieval augmentor avec builder
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        AssistantPasRAG assistant = AiServices.builder(AssistantPasRAG.class)
                .chatModel(model)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // Questions en temps réel
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question : ");
                String question = scanner.nextLine();
                if (question.isBlank()) {
                    continue;
                }
                System.out.println("==================================================");
                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }
                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }
    }
}