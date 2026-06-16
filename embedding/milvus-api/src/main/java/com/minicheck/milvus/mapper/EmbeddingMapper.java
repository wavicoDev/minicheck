package com.minicheck.milvus.mapper;

import com.minicheck.milvus.dto.EmbeddingRequestDTO;
import com.minicheck.milvus.dto.EmbeddingResponseDTO;
import io.milvus.grpc.SearchResults;
import io.milvus.response.SearchResultsWrapper;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

@Component
public class EmbeddingMapper {

    public List<Object> toFieldValues(EmbeddingRequestDTO request) {
        List<Object> fields = new ArrayList<>();
        fields.add(request.getDocumentId());
        fields.add(request.getContent());
        fields.add(request.getMetadata() != null ? request.getMetadata() : "");
        fields.add(request.getEmbedding());
        return fields;
    }

    public List<EmbeddingResponseDTO> toResponseDTOList(SearchResults searchResults) {
        List<EmbeddingResponseDTO> results = new ArrayList<>();

        SearchResultsWrapper wrapper = new SearchResultsWrapper(searchResults.getResults());
        int numQueries = wrapper.getNumQueries();

        for (int i = 0; i < numQueries; i++) {
            List<SearchResultsWrapper.IDScore> idScores = wrapper.getIDScore(i);

            for (SearchResultsWrapper.IDScore idScore : idScores) {
                EmbeddingResponseDTO response = EmbeddingResponseDTO.builder()
                        .id(idScore.getLongID())
                        .score(idScore.getScore())
                        .build();

                Object documentId = wrapper.getFieldData("document_id", i);
                Object content = wrapper.getFieldData("content", i);
                Object metadata = wrapper.getFieldData("metadata", i);

                if (documentId instanceof List<?> docList && !docList.isEmpty()) {
                    int idx = idScores.indexOf(idScore);
                    if (idx < docList.size()) {
                        response.setDocumentId((String) docList.get(idx));
                    }
                }
                if (content instanceof List<?> contentList && !contentList.isEmpty()) {
                    int idx = idScores.indexOf(idScore);
                    if (idx < contentList.size()) {
                        response.setContent((String) contentList.get(idx));
                    }
                }
                if (metadata instanceof List<?> metaList && !metaList.isEmpty()) {
                    int idx = idScores.indexOf(idScore);
                    if (idx < metaList.size()) {
                        response.setMetadata((String) metaList.get(idx));
                    }
                }

                results.add(response);
            }
        }

        return results;
    }
}
