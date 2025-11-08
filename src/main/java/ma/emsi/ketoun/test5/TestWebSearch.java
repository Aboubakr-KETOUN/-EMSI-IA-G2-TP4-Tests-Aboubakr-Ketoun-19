package ma.emsi.ketoun.test5;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import ma.emsi.ketoun.Interfaces.Assistant;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;


import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

public class TestWebSearch {

    public static void main(String[] args) {

        //retrait des cles du env local
        String apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null) {
            System.err.println("gamini Key not found!");
            return;
        }

        String tavilyKey = System.getenv("TAVILY_KEY");
        if (tavilyKey == null) {
            System.err.println("tavily key not found!");
            return;
        }

        // Creation du Model avec le builder
        GoogleAiGeminiChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.7)
                .build();

        //recuperation du fichier
        Path pathRessource = getPathRessource("/rag.pdf");

        //creation du parser pour PDF
        DocumentParser documentParser = new ApacheTikaDocumentParser();

        //chargement du fichier avec le parser en param
        Document document = FileSystemDocumentLoader.loadDocument(pathRessource, documentParser);

        //creation du document splitter pour decouper le fichier 30 pour overlap comme demandee
        DocumentSplitter documentSplitter = DocumentSplitters.recursive(300, 30);

        //list des segments de type textsegment
        List<TextSegment> segments = documentSplitter.split(document);

        //creation du model d'embedding
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        //creation des embeddings pour les segments
        Response<List<Embedding>> embeddingsResponse = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = embeddingsResponse.content();

        //creation du magasin d'embeddings en memoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        //ajout des embeddings avec leurs segments
        embeddingStore.addAll(embeddings, segments);


        //Phase 2 :
        //creation du contentretriever avec des parametres
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // declaration du webseach
        WebSearchEngine tavilyWebSearchEngine = TavilyWebSearchEngine
                .builder()
                .apiKey(tavilyKey)
                .build();

        //creation du content retriever avec le websearch retriever
        ContentRetriever webSearchContentRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(tavilyWebSearchEngine)
                .maxResults(3)
                .build();


        //declaration du queryrouter
        DefaultQueryRouter queryRouter = new DefaultQueryRouter(contentRetriever, webSearchContentRetriever);

        //creation du retrieval augmentor avec builder
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        //creation de memoire de 10 msgs
        MessageWindowChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);


        //creation de lassistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(chatMemory)
                .build();

        //questions en temps reel
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


    private static Path getPathRessource(String cheminRessource) {
        Path pathRessource;
        try {
            URL fileUrl = ma.emsi.ketoun.test1.RagNaif.class.getResource(cheminRessource);
            if (fileUrl == null) {
                throw new RuntimeException("Impossible de trouver le fichier " + cheminRessource);
            }
            pathRessource = Paths.get(fileUrl.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
        return pathRessource;
    }


}
