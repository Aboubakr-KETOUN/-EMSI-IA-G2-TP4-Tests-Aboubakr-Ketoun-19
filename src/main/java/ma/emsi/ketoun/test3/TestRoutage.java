package ma.emsi.ketoun.test3;

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
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.ketoun.Interfaces.Assistant;
import ma.emsi.ketoun.test1.RagNaif;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestRoutage {
    public static void main(String[] args) {
        TestRoutage test = new TestRoutage();
        test.execute();
    }

    private static void configureLogger() {
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // Ajuster niveau
        // Ajouter un handler pour la console pour faire afficher les logs
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    private static Path getPathRessource(String cheminRessource) {
        Path pathRessource;
        try {
            URL fileUrl = RagNaif.class.getResource(cheminRessource);
            if (fileUrl == null) {
                throw new RuntimeException("Impossible de trouver le fichier " + cheminRessource);
            }
            pathRessource = Paths.get(fileUrl.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
        return pathRessource;
    }

    public void execute() {
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

        // creation du modèle d'embedding
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        //creation des embeddins store
        EmbeddingStore<TextSegment> embeddingStore1 = creerEmbeddingStore("/rag.pdf", embeddingModel);
        EmbeddingStore<TextSegment> embeddingStore2 = creerEmbeddingStore("/autre.pdf", embeddingModel);

        //creation des content retrievers
        ContentRetriever contentRetriever1 = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore1)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        ContentRetriever contentRetriever2 = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore2)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        //creation du map des descriptions
        Map<ContentRetriever, String> retrieverDescriptions = new HashMap<>();

        retrieverDescriptions.put(
                contentRetriever1,
                "Ce retriever contient des informations sur l'intelligence artificielle, " +
                        "le RAG (Retrieval Augmented Generation), " +
                        "les embeddings et les modèles de langage."
        );

        retrieverDescriptions.put(
                contentRetriever2,
                "Ce retriever contient des informations sur une certif oracle. "
        );

        //creation du queryrouter
        QueryRouter queryRouter = new LanguageModelQueryRouter(model, retrieverDescriptions);

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

    public static EmbeddingStore<TextSegment> creerEmbeddingStore(String cheminFichier, EmbeddingModel embeddingModel) {
        // Récupération du Path
        Path pathRessource = getPathRessource(cheminFichier);

        // Parser
        DocumentParser documentParser = new ApacheTikaDocumentParser();

        // Chargement du document
        Document document = FileSystemDocumentLoader.loadDocument(pathRessource, documentParser);

        // Splitter
        DocumentSplitter documentSplitter = DocumentSplitters.recursive(300, 30);

        // Découpage en segments
        List<TextSegment> segments = documentSplitter.split(document);

        // Création des embeddings
        Response<List<Embedding>> embeddingsResponse = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = embeddingsResponse.content();

        // Création du store
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // Ajout des embeddings
        embeddingStore.addAll(embeddings, segments);

        return embeddingStore;
    }

}
